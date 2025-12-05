import os
import argparse
from ml_collections import ConfigDict
import yaml

import copy
import numpy as np
import torch
from torch import optim
from torch.utils.data import DataLoader
from torch_geometric.transforms import Compose
from tqdm import tqdm
import wandb
import time

from data.data_preprocess import HeteroAddLaplacianEigenvectorPE, SubSample
from data.dataset import LPDataset
from data.utils import args_set_bool, collate_fn_ip
from models.hetero_gnn import TripartiteHeteroGNN, BipartiteHeteroGNN
from trainer import Trainer


def args_parser():
    # 定義命令行參數解析器，用於設置訓練的超參數
    parser = argparse.ArgumentParser(description='hyper params for training graph dataset')
    # admin 管理相關參數
    parser.add_argument('--datapath', type=str, required=True)  # 數據集路徑
    parser.add_argument('--wandbproject', type=str, default='ipm_mpnn')  # wandb項目名稱
    parser.add_argument('--wandbname', type=str, default='')  # wandb運行名稱
    parser.add_argument('--use_wandb', type=str, default='true')  # 是否使用wandb

    # ipm processing
    parser.add_argument('--ipm_restarts', type=int, default=1)  # more does not help
    parser.add_argument('--ipm_steps', type=int, default=8)
    parser.add_argument('--ipm_alpha', type=float, default=0.9)
    parser.add_argument('--upper', type=float, default=1.0)

    # training dynamics
    parser.add_argument('--ckpt', type=str, default='true')
    parser.add_argument('--runs', type=int, default=1)
    parser.add_argument('--lr', type=float, default=1.e-3)
    parser.add_argument('--weight_decay', type=float, default=0.)
    parser.add_argument('--epoch', type=int, default=1000)
    parser.add_argument('--patience', type=int, default=100)
    parser.add_argument('--batchsize', type=int, default=16)
    parser.add_argument('--micro_batch', type=int, default=1)
    parser.add_argument('--dropout', type=float, default=0.)  # must
    parser.add_argument('--use_norm', type=str, default='true')  # must
    parser.add_argument('--use_res', type=str, default='false')  # does not help

    # model related
    parser.add_argument('--bipartite', type=str, default='false')
    parser.add_argument('--conv', type=str, default='genconv')
    parser.add_argument('--lappe', type=int, default=0)
    parser.add_argument('--hidden', type=int, default=128)
    parser.add_argument('--num_conv_layers', type=int, default=8)
    parser.add_argument('--num_pred_layers', type=int, default=2)
    parser.add_argument('--num_mlp_layers', type=int, default=2, help='mlp layers within GENConv')
    parser.add_argument('--share_conv_weight', type=str, default='false')
    parser.add_argument('--share_lin_weight', type=str, default='false')
    parser.add_argument('--conv_sequence', type=str, default='cov')

    # loss related
    parser.add_argument('--loss', type=str, default='primal+objgap+constraint')
    parser.add_argument('--loss_weight_x', type=float, default=1.0)
    parser.add_argument('--loss_weight_obj', type=float, default=1.0)
    parser.add_argument('--loss_weight_cons', type=float, default=1.0)  # does not work
    parser.add_argument('--losstype', type=str, default='l2', choices=['l1', 'l2'])  # no big different
    return parser.parse_args()


if __name__ == '__main__':
    # 解析命令行參數
    args = args_parser()
    args = args_set_bool(vars(args))  # 將布爾參數轉換為布爾值
    args = ConfigDict(args)  # 將參數轉換為ConfigDict格式

    # 如果需要保存檢查點，創建日誌目錄
    if args.ckpt:
        if not os.path.isdir('logs'):
            os.mkdir('logs')
        exist_runs = [d for d in os.listdir('logs') if d.startswith('exp')]
        log_folder_name = f'logs/exp{len(exist_runs)}'
        os.mkdir(log_folder_name)
        with open(os.path.join(log_folder_name, 'config.yaml'), 'w') as outfile:
            yaml.dump(args.to_dict(), outfile, default_flow_style=False)

    # 初始化wandb
    wandb.init(project=args.wandbproject,
               name=args.wandbname if args.wandbname else None,
            #    mode="online" if args.use_wandb else "disabled",
               config=vars(args),
               )  # use your own entity

    # 加載數據集，並進行預處理
    dataset = LPDataset(args.datapath,
                        extra_path=f'{args.ipm_restarts}restarts_'
                                         f'{args.lappe}lap_'
                                         f'{args.ipm_steps}steps'
                                         f'{"_upper_" + str(args.upper) if args.upper is not None else ""}',
                        upper_bound=args.upper,
                        rand_starts=args.ipm_restarts,
                        pre_transform=Compose([HeteroAddLaplacianEigenvectorPE(k=args.lappe),
                                                     SubSample(args.ipm_steps)]))

    # 劃分數據集為訓練、驗證和測試集
    train_loader = DataLoader(dataset[:int(len(dataset) * 0.8)],
                              batch_size=args.batchsize,
                              shuffle=True,
                              num_workers=1,
                              collate_fn=collate_fn_ip)
    val_loader = DataLoader(dataset[int(len(dataset) * 0.8):int(len(dataset) * 0.9)],
                            batch_size=args.batchsize,
                            shuffle=False,
                            num_workers=1,
                            collate_fn=collate_fn_ip)
    test_loader = DataLoader(dataset[int(len(dataset) * 0.9):],
                            batch_size=args.batchsize,
                            shuffle=False,
                            num_workers=1,
                            collate_fn=collate_fn_ip)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'  # 設置設備為GPU或CPU

    # 初始化結果存儲變量
    # best_val_losses = []
    best_val_objgap_mean = []
    best_val_consgap_mean = []
    # test_losses = []
    test_objgap_mean = []
    test_consgap_mean = []

    # 多次運行訓練
    for run in range(args.runs):
        if args.ckpt:
            os.mkdir(os.path.join(log_folder_name, f'run{run}'))
        # 選擇模型類型
        if args.bipartite:
            model = BipartiteHeteroGNN(conv=args.conv,
                                       in_shape=2,
                                       pe_dim=args.lappe,
                                       hid_dim=args.hidden,
                                       num_conv_layers=args.num_conv_layers,
                                       num_pred_layers=args.num_pred_layers,
                                       num_mlp_layers=args.num_mlp_layers,
                                       dropout=args.dropout,
                                       share_conv_weight=args.share_conv_weight,
                                       share_lin_weight=args.share_lin_weight,
                                       use_norm=args.use_norm,
                                       use_res=args.use_res).to(device)
        else:
            model = TripartiteHeteroGNN(conv=args.conv,
                                        in_shape=2,
                                        pe_dim=args.lappe,
                                        hid_dim=args.hidden,
                                        num_conv_layers=args.num_conv_layers,
                                        num_pred_layers=args.num_pred_layers,
                                        num_mlp_layers=args.num_mlp_layers,
                                        dropout=args.dropout,
                                        share_conv_weight=args.share_conv_weight,
                                        share_lin_weight=args.share_lin_weight,
                                        use_norm=args.use_norm,
                                        use_res=args.use_res,
                                        conv_sequence=args.conv_sequence).to(device)
        best_model = copy.deepcopy(model.state_dict())  # 保存最佳模型的狀態字典

        # 設置優化器和學習率調度器
        optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=50, min_lr=1.e-5)

        # 初始化訓練器
        trainer = Trainer(device,
                          args.loss,
                          args.losstype,
                          args.micro_batch,
                          min(args.ipm_steps, args.num_conv_layers),
                          args.ipm_alpha,
                          loss_weight={'primal': args.loss_weight_x,
                                       'objgap': args.loss_weight_obj,
                                       'constraint': args.loss_weight_cons})

        # 訓練過程
        pbar = tqdm(range(args.epoch))
        for epoch in pbar:
            train_loss = trainer.train(train_loader, model, optimizer)  # 訓練模型

            with torch.no_grad():
                # 驗證模型性能
                # start timing validation inference
                if device == 'cuda':
                    torch.cuda.synchronize()
                val_start = time.perf_counter()

                val_gaps, val_constraint_gap = trainer.eval_metrics(val_loader, model)

                if device == 'cuda':
                    torch.cuda.synchronize()
                val_end = time.perf_counter()
                val_infer_time = val_end - val_start
                val_num = val_gaps.shape[0] if val_gaps is not None else 0
                val_avg_infer = val_infer_time / val_num if val_num > 0 else 0.0

                # 根據驗證結果更新最佳模型
                cur_mean_gap = val_gaps[:, -1].mean().item()
                cur_cons_gap_mean = val_constraint_gap[:, -1].mean().item()
                if scheduler is not None:
                    scheduler.step(cur_mean_gap)

                if trainer.best_val_objgap > cur_mean_gap:
                    trainer.patience = 0
                    trainer.best_val_objgap = cur_mean_gap
                    trainer.best_val_consgap = cur_cons_gap_mean
                    best_model = copy.deepcopy(model.state_dict())
                    if args.ckpt:
                        torch.save(model.state_dict(), os.path.join(log_folder_name, f'run{run}', 'best_model.pt'))
                else:
                    trainer.patience += 1

            if trainer.patience > args.patience:  # 提前停止條件
                break

            # 更新進度條和wandb日誌
            pbar.set_postfix({'train_loss': train_loss,
                              # 'val_loss': val_loss,
                              'val_obj': cur_mean_gap,
                              'val_cons': cur_cons_gap_mean,
                              'val_infer_s': f'{val_infer_time:.4f}',
                              'lr': scheduler.optimizer.param_groups[0]["lr"]})
            log_dict = {'train_loss': train_loss,
                       # 'val_loss': val_loss,
                        'val_obj_gap_last_mean': cur_mean_gap,
                        'val_cons_gap_last_mean': cur_cons_gap_mean,
                       'lr': scheduler.optimizer.param_groups[0]["lr"],
                       'val_infer_time_total_s': val_infer_time,
                       'val_infer_time_per_instance_s': val_avg_infer}
            # for gnn_l in range(train_gaps.shape[1]):
            #     log_dict[f'train_obj_gap_l{gnn_l}_mean'] = train_gaps[:, gnn_l].mean()
                # log_dict[f'train_obj_gap_l{gnn_l}'] = wandb.Histogram(train_gaps[:, gnn_l])
            # for gnn_l in range(val_gaps.shape[1]):
            #     log_dict[f'val_obj_gap_l{gnn_l}_mean'] = val_gaps[:, gnn_l].mean()
                # log_dict[f'val_obj_gap_l{gnn_l}'] = wandb.Histogram(val_gaps[:, gnn_l])
            # for gnn_l in range(train_constraint_gap.shape[1]):
            #     log_dict[f'train_cons_gap_l{gnn_l}_mean'] = train_constraint_gap[:, gnn_l].mean()
                # log_dict[f'train_cons_gap_l{gnn_l}'] = wandb.Histogram(train_constraint_gap[:, gnn_l])
            # for gnn_l in range(val_constraint_gap.shape[1]):
            #     log_dict[f'val_cons_gap_l{gnn_l}_mean'] = val_constraint_gap[:, gnn_l].mean()
                # log_dict[f'val_cons_gap_l{gnn_l}'] = wandb.Histogram(val_constraint_gap[:, gnn_l])
            wandb.log(log_dict)
        # 保存最佳驗證結果
        # best_val_losses.append(trainer.best_val_loss)
        best_val_objgap_mean.append(trainer.best_val_objgap)
        best_val_consgap_mean.append(trainer.best_val_consgap)

        # 測試模型性能
        model.load_state_dict(best_model)
        with torch.no_grad():
            # time test inference
            if device == 'cuda':
                torch.cuda.synchronize()
            test_start = time.perf_counter()

            test_gaps, test_cons_gap = trainer.eval_metrics(test_loader, model)

            if device == 'cuda':
                torch.cuda.synchronize()
            test_end = time.perf_counter()
        test_infer_time = test_end - test_start
        test_num = test_gaps.shape[0] if test_gaps is not None else 0
        test_avg_infer = test_infer_time / test_num if test_num > 0 else 0.0

        # test_losses.append(test_loss)
        test_objgap_mean.append(test_gaps[:, -1].mean().item())
        test_consgap_mean.append(test_cons_gap[:, -1].mean().item())

        wandb.log({'test_objgap': test_objgap_mean[-1],
                   'test_infer_time_total_s': test_infer_time,
                   'test_infer_time_per_instance_s': test_avg_infer})
        wandb.log({'test_consgap': test_consgap_mean[-1]})


    # 最終結果記錄到wandb
    wandb.log({
        # 'best_val_loss': np.mean(best_val_losses),
        'best_val_objgap': np.mean(best_val_objgap_mean),
        # 'test_loss_mean': np.mean(test_losses),
        # 'test_loss_std': np.std(test_losses),
        'test_objgap_mean': np.mean(test_objgap_mean),
        'test_objgap_std': np.std(test_objgap_mean),
        'test_consgap_mean': np.mean(test_consgap_mean),
        'test_consgap_std': np.std(test_consgap_mean),
        'test_hybrid_gap': np.mean(test_objgap_mean) + np.mean(test_consgap_mean),
        # add aggregated timing stats
        'test_infer_time_total_s_mean': np.mean([0 if x is None else x for x in [np.nan]]),  # placeholder if needed
    })
    print('Done!')