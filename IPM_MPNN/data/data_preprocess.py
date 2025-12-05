import numpy as np
import torch
from torch_geometric.data.hetero_data import to_homogeneous_edge_index
from torch_geometric.transforms import AddLaplacianEigenvectorPE
from data.utils import log_normalize

# 對gt_primals進行log歸一化
class LogNormalize:
    def __init__(self):
        pass

    def __call__(self, data):
        data.gt_primals = log_normalize(data.gt_primals)
        return data

# 為異構圖的每個node type添加拉普拉斯特徵
class HeteroAddLaplacianEigenvectorPE:
    def __init__(self, k, attr_name='laplacian_eigenvector_pe'):
        self.k = k
        self.attr_name = attr_name

    def __call__(self, data):
        if self.k == 0:
            return data
        # 轉為同構圖以計算拉普拉斯特徵
        data_homo = data.to_homogeneous()
        del data_homo.edge_weight
        lap = AddLaplacianEigenvectorPE(k=self.k, attr_name=self.attr_name)(data_homo).laplacian_eigenvector_pe

        # 根據node_slices分配特徵到不同node type
        _, node_slices, _ = to_homogeneous_edge_index(data)
        cons_lap = lap[node_slices['cons'][0]: node_slices['cons'][1], :]
        cons_lap = (cons_lap - cons_lap.mean(0)) / cons_lap.std(0)  # 標準化
        vals_lap = lap[node_slices['vals'][0]: node_slices['vals'][1], :]
        vals_lap = (vals_lap - vals_lap.mean(0)) / vals_lap.std(0)  # 標準化
        obj_lap = lap[node_slices['obj'][0]: node_slices['obj'][1], :]

        data['cons'].laplacian_eigenvector_pe = cons_lap
        data['vals'].laplacian_eigenvector_pe = vals_lap
        data['obj'].laplacian_eigenvector_pe = obj_lap
        return data

# 對序列型標籤進行子採樣或補齊
class SubSample:
    def __init__(self, k):
        self.k = k

    def __call__(self, data):
        len_seq = data.gt_primals.shape[1]
        # 只保留最後一個步驟
        if self.k == 1:
            data.gt_primals = data.gt_primals[:, -1:]
            if hasattr(data, 'gt_duals'):
                data.gt_duals = data.gt_duals[:, -1:]
            if hasattr(data, 'gt_slacks'):
                data.gt_slacks = data.gt_slacks[:, -1:]
        # 不做處理
        elif self.k == len_seq:
            return data
        # 補齊到k步
        elif self.k > len_seq:
            data.gt_primals = torch.cat([data.gt_primals,
                                         data.gt_primals[:, -1:].repeat(1, self.k - len_seq)], dim=1)
            if hasattr(data, 'gt_duals'):
                data.gt_duals = torch.cat([data.gt_duals,
                                           data.gt_duals[:, -1:].repeat(1, self.k - len_seq)], dim=1)
            if hasattr(data, 'gt_slacks'):
                data.gt_slacks = torch.cat([data.gt_slacks,
                                            data.gt_slacks[:, -1:].repeat(1, self.k - len_seq)], dim=1)
        # 均勻子採樣k個步驟
        else:
            data.gt_primals = data.gt_primals[:, np.linspace(1, len_seq - 1, self.k).astype(np.int64)]
            if hasattr(data, 'gt_duals'):
                data.gt_duals = data.gt_duals[:, np.linspace(1, len_seq - 1, self.k).astype(np.int64)]
            if hasattr(data, 'gt_slacks'):
                data.gt_slacks = data.gt_slacks[:, np.linspace(1, len_seq - 1, self.k).astype(np.int64)]
        return data