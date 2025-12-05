import gzip
import os
import os.path as osp
import pickle
from typing import Callable, List, Optional

import numpy as np
import torch
from torch_geometric.data import Batch, HeteroData, InMemoryDataset
from torch_sparse import SparseTensor

from solver.linprog import linprog
from tqdm import tqdm


class LPDataset(InMemoryDataset):

    def __init__(
        self,
        root: str,
        extra_path: str,
        upper_bound: Optional[float] = None,
        rand_starts: int = 1,
        transform: Optional[Callable] = None,
        pre_transform: Optional[Callable] = None,
        pre_filter: Optional[Callable] = None,
    ):
        # 隨機啟動次數（每個 instance 可以做多次求解以取得不同的 intermediate 轨迹）
        self.rand_starts = rand_starts
        # 是否把 A/b 視為不等式（A_ub x <= b_ub），預設為 True（常見情況）
        self.using_ineq = True
        # 處理後資料夾的後綴字串（例如 processed_1restarts_...）
        self.extra_path = extra_path
        # x 的上界（box constraint），用於傳給求解器
        self.upper_bound = upper_bound
        # 呼叫父類別建構子；如果 processed 資料不存在會觸發 process()
        super().__init__(root, transform, pre_transform, pre_filter)
        # 讀取處理後合併好的資料（torch 保存的 data.pt）
        path = osp.join(self.processed_dir, 'data.pt')
        self.data, self.slices = torch.load(path)

    @property
    def raw_file_names(self) -> List[str]:
        return ['instance_0.pkl.gz']   # there should be at least one pkg

    @property
    def processed_dir(self) -> str:
        return osp.join(self.root, 'processed_' + self.extra_path)

    @property
    def processed_file_names(self) -> List[str]:
        return ['data.pt']

    def process(self):
        num_instance_pkg = len([n for n in os.listdir(self.raw_dir) if n.endswith('pkl.gz')])

        data_list = []
        for i in range(num_instance_pkg):
            # load instance
            print(f"processing {i}th package, {num_instance_pkg} in total")
            with gzip.open(os.path.join(self.raw_dir, f"instance_{i}.pkl.gz"), "rb") as file:
                ip_pkgs = pickle.load(file)

            for ip_idx in tqdm(range(len(ip_pkgs))):
                (A, b, c) = ip_pkgs[ip_idx]
                # 將密集矩陣 A 轉為 SparseTensor，方便取得 COO-style 的 (row, col, value)
                sp_a = SparseTensor.from_dense(A, has_value=True)

                # 從 SparseTensor 的內部儲存取得非零元素的 row, col, value（長度為 nnz）
                row = sp_a.storage._row
                col = sp_a.storage._col
                val = sp_a.storage._value

                # A_tilde_mask 用來標示哪些非零元素被視為 "tilde"（模型可能會用到）
                # 在不等式情況下我們保留所有非零元素
                if self.using_ineq:
                    tilde_mask = torch.ones(row.shape, dtype=torch.bool)
                else:
                    # 在等式情況的示例（專案中通常不使用此路徑）
                    tilde_mask = col < (A.shape[1] - A.shape[0])

                # 對目標向量 c 做數值正規化，避免數值不穩定（不改變 argmin）
                c = c / (c.abs().max() + 1.e-10)  # does not change the result

                # 根據是否為不等式設定傳給求解器的參數
                if self.using_ineq:
                    A_ub = A.numpy()
                    b_ub = b.numpy()
                    A_eq = None
                    b_eq = None
                else:
                    A_eq = A.numpy()
                    b_eq = b.numpy()
                    A_ub = None
                    b_ub = None

                bounds = (0, self.upper_bound)

                # 如果 rand_starts > 1，會對同一 instance 進行多次求解並各自儲存 intermediate
                for _ in range(self.rand_starts):
                    # 呼叫專案的 linprog（interior-point）求解器，callback 返回 intermediate x
                    sol = linprog(c.numpy(),
                                  A_ub=A_ub,
                                  b_ub=b_ub,
                                  A_eq=A_eq, b_eq=b_eq, bounds=bounds,
                                  method='interior-point', callback=lambda res: res.x)
                    # sol.intermediate 預期為 list/array，堆疊成 (n_vars, n_steps)
                    x = np.stack(sol.intermediate, axis=1)
                    assert not np.isnan(sol['fun'])

                    # 轉為 torch 張量作為 ground-truth primal sequence
                    gt_primals = torch.from_numpy(x).to(torch.float)
                    # gt_duals = torch.from_numpy(l).to(torch.float)
                    # gt_slacks = torch.from_numpy(s).to(torch.float)

                    # 建構 HeteroData：
                    # - cons.x: 每個 constraint 的 (mean, std) -> shape (n_cons, 2)
                    # - vals.x: 每個 variable 的 (mean, std) -> shape (n_vars, 2)
                    # - obj.x: objective 的統計 -> shape (1, 2)
                    # 還會加入 hetero-edge，edge_attr 帶入 A 或 c 或 b 的值
                    data = HeteroData(
                        cons={'x': torch.cat([A.mean(1, keepdims=True),
                                              A.std(1, keepdims=True)], dim=1)},
                        vals={'x': torch.cat([A.mean(0, keepdims=True),
                                              A.std(0, keepdims=True)], dim=0).T},
                        obj={'x': torch.cat([c.mean(0, keepdims=True),
                                             c.std(0, keepdims=True)], dim=0)[None]},

                        # cons -> vals: constraint i 到 variable j 的邊（A 的非零位置）
                        cons__to__vals={'edge_index': torch.vstack(torch.where(A)),
                                        'edge_attr': A[torch.where(A)][:, None]},
                        # vals -> cons: 變數到約束的反向邊（A^T 的非零）
                        vals__to__cons={'edge_index': torch.vstack(torch.where(A.T)),
                                        'edge_attr': A.T[torch.where(A.T)][:, None]},
                        # vals <-> obj: 變數與目標的連接，edge_attr 為 c
                        vals__to__obj={'edge_index': torch.vstack([torch.arange(A.shape[1]),
                                                                   torch.zeros(A.shape[1], dtype=torch.long)]),
                                       'edge_attr': c[:, None]},
                        obj__to__vals={'edge_index': torch.vstack([torch.zeros(A.shape[1], dtype=torch.long),
                                                                   torch.arange(A.shape[1])]),
                                       'edge_attr': c[:, None]},
                        # cons <-> obj: 約束與目標的連接，edge_attr 為 b
                        cons__to__obj={'edge_index': torch.vstack([torch.arange(A.shape[0]),
                                                                   torch.zeros(A.shape[0], dtype=torch.long)]),
                                       'edge_attr': b[:, None]},
                        obj__to__cons={'edge_index': torch.vstack([torch.zeros(A.shape[0], dtype=torch.long),
                                                                   torch.arange(A.shape[0])]),
                                       'edge_attr': b[:, None]},
                        # 存放 ground-truth primals 軌跡 (n_vars, ipm_steps)
                        gt_primals=gt_primals,
                        # 儲存最佳目標值與原始 c
                        obj_value=torch.tensor(sol['fun'].astype(np.float32)),
                        obj_const=c,

                        # 稀疏矩陣 A 的 metadata，Trainer 會使用這些欄位計算 Ax 等
                        A_row=row,
                        A_col=col,
                        A_val=val,
                        A_num_row=A.shape[0],
                        A_num_col=A.shape[1],
                        A_nnz=len(val),
                        A_tilde_mask=tilde_mask,
                        rhs=b)

                    # pre_filter 未實作 (保持原樣)
                    if self.pre_filter is not None:
                        raise NotImplementedError

                    # 若提供 pre_transform，則在加入 data_list 前呼叫
                    if self.pre_transform is not None:
                        data = self.pre_transform(data)

                    data_list.append(data)

            torch.save(Batch.from_data_list(data_list), osp.join(self.processed_dir, f'batch{i}.pt'))
            data_list = []

        data_list = []
        for i in range(num_instance_pkg):
            data_list.extend(Batch.to_data_list(torch.load(osp.join(self.processed_dir, f'batch{i}.pt'))))
        torch.save(self.collate(data_list), osp.join(self.processed_dir, 'data.pt'))