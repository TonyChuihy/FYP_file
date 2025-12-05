#!/usr/bin/env python3
"""
Small test runner:
- Loads dataset (uses same pre-transforms as training)
- Instantiates model matching checkpoint hyperparams (provide via args)
- Loads checkpoint and runs Trainer.eval_metrics on the whole dataset
- Logs mean/std results to wandb
"""

import os
import argparse
from ml_collections import ConfigDict
import yaml

import numpy as np
import torch
from torch.utils.data import DataLoader
from torch_geometric.transforms import Compose
import wandb
import time

from data.data_preprocess import HeteroAddLaplacianEigenvectorPE, SubSample
from data.dataset import LPDataset
from data.utils import args_set_bool, collate_fn_ip
from models.hetero_gnn import TripartiteHeteroGNN, BipartiteHeteroGNN
from trainer import Trainer


def args_parser():
    parser = argparse.ArgumentParser(description='Test saved model on a dataset and log to wandb')
    parser.add_argument('--ckpt', type=str, default='logs/IndsetGCN/run1/best_model.pt', help='path to checkpoint .pt')
    parser.add_argument('--datapath', type=str, default='d/fac6', help='dataset folder (as used by LPDataset)')
    parser.add_argument('--wandbproject', type=str, default='ipm_mpnn')
    parser.add_argument('--wandbname', type=str, default='test_run')
    parser.add_argument('--use_wandb', type=str, default='true')

    # minimal model/transform options to match training
    parser.add_argument('--bipartite', type=str, default='false')
    parser.add_argument('--conv', type=str, default='genconv')
    parser.add_argument('--lappe', type=int, default=0)
    parser.add_argument('--ipm_steps', type=int, default=8)
    parser.add_argument('--ipm_restarts', type=int, default=1)
    parser.add_argument('--ipm_alpha', type=float, default=0.9)
    parser.add_argument('--upper', type=float, default=1.0)

    parser.add_argument('--hidden', type=int, default=128)
    parser.add_argument('--num_conv_layers', type=int, default=8)
    parser.add_argument('--num_pred_layers', type=int, default=2)
    parser.add_argument('--num_mlp_layers', type=int, default=2)
    parser.add_argument('--dropout', type=float, default=0.)
    parser.add_argument('--share_conv_weight', type=str, default='false')
    parser.add_argument('--share_lin_weight', type=str, default='false')
    parser.add_argument('--use_norm', type=str, default='true')
    parser.add_argument('--use_res', type=str, default='false')
    parser.add_argument('--conv_sequence', type=str, default='cov')
    parser.add_argument('--batchsize', type=int, default=16)

    # loss/trainer params (kept minimal)
    parser.add_argument('--loss', type=str, default='primal+objgap+constraint')
    parser.add_argument('--losstype', type=str, default='l2')
    parser.add_argument('--micro_batch', type=int, default=1)
    parser.add_argument('--loss_weight_x', type=float, default=1.0)
    parser.add_argument('--loss_weight_obj', type=float, default=1.0)
    parser.add_argument('--loss_weight_cons', type=float, default=1.0)
    return parser.parse_args()


if __name__ == '__main__':
    args = args_parser()
    args = args_set_bool(vars(args))
    args = ConfigDict(args)

    if not os.path.isfile(args.ckpt):
        raise FileNotFoundError(f'Checkpoint not found: {args.ckpt}')

    # init wandb (can be disabled by passing use_wandb=false)
    wandb.init(project=args.wandbproject,
               name=args.wandbname if args.wandbname else None,
               config=vars(args),
               mode="online" if args.use_wandb else "disabled")

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # build dataset with same pre-transforms as training
    extra_path = f'{args.ipm_restarts}restarts_{args.lappe}lap_{args.ipm_steps}steps' + (f'_upper_{args.upper}' if args.upper is not None else '')
    dataset = LPDataset(args.datapath,
                        extra_path=extra_path,
                        upper_bound=args.upper,
                        rand_starts=args.ipm_restarts,
                        pre_transform=Compose([HeteroAddLaplacianEigenvectorPE(k=args.lappe),
                                               SubSample(args.ipm_steps)]))

    loader = DataLoader(dataset, batch_size=args.batchsize, shuffle=False, num_workers=1, collate_fn=collate_fn_ip)

    # instantiate model matching the training configuration
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

    # load checkpoint
    state = torch.load(args.ckpt, map_location=device)
    model.load_state_dict(state)
    model.eval()

    # prepare trainer (same interface as training)
    trainer = Trainer(device,
                      args.loss,
                      args.losstype,
                      args.micro_batch,
                      min(args.ipm_steps, args.num_conv_layers),
                      args.ipm_alpha,
                      loss_weight={'primal': args.loss_weight_x,
                                   'objgap': args.loss_weight_obj,
                                   'constraint': args.loss_weight_cons})

    # run evaluation on entire dataset
    with torch.no_grad():
        # start timing model computation
        if device == 'cuda':
            torch.cuda.synchronize()
        start_time = time.perf_counter()

        gaps, cons_gaps = trainer.eval_metrics(loader, model)

        if device == 'cuda':
            torch.cuda.synchronize()
        end_time = time.perf_counter()
    total_infer_time = end_time - start_time
    num_instances = gaps.shape[0] if gaps is not None else 0
    avg_infer_time = total_infer_time / num_instances if num_instances > 0 else 0.0

    # compute summary metrics
    obj_mean = gaps[:, -1].mean().item()
    obj_std = gaps[:, -1].std().item()
    cons_mean = cons_gaps[:, -1].mean().item()
    cons_std = cons_gaps[:, -1].std().item()

    log_dict = {
        'test_objgap_mean': obj_mean,
        'test_objgap_std': obj_std,
        'test_consgap_mean': cons_mean,
        'test_consgap_std': cons_std,
        'test_hybrid_gap': obj_mean + cons_mean,
        'test_infer_time_total_s': total_infer_time,
        'test_infer_time_per_instance_s': avg_infer_time,
        'test_num_instances': num_instances
    }
    wandb.log(log_dict)
    print('Test results:', log_dict)
    print('Number of test instances:', num_instances)
    wandb.finish()