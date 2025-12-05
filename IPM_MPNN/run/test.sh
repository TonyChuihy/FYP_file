#!/bin/bash
# --wandbname IndsetGCN_on_Setcover

# GENConv Testing Commands

## setcover (GEN)
python testing.py --ckpt logs/SetcoverGEN/run1/best_model.pt --ipm_alpha 0.15 --micro_batch 4 --batchsize 4 --hidden 180 --num_pred_layers 4 --num_mlp_layers 4 --share_lin_weight false --conv_sequence cov --loss_weight_x 1.2 --loss_weight_obj 0.8 --loss_weight_cons 0.16 --conv genconv --wandbname SetcGEN

## indset (GEN)
python testing.py --ckpt logs/IndsetGEN/run1/best_model.pt --ipm_alpha 0.15 --micro_batch 4 --batchsize 4 --hidden 180 --num_pred_layers 4 --num_mlp_layers 4 --share_lin_weight false --conv_sequence cov --loss_weight_x 1.2 --loss_weight_obj 0.8 --loss_weight_cons 0.16 --conv genconv --wandbname IndsGEN

## cauction (GEN)
python testing.py --ckpt logs/CaucGEN/run1/best_model.pt --ipm_alpha 0.86 --micro_batch 4 --batchsize 4 --hidden 180 --num_pred_layers 4 --num_mlp_layers 4 --share_lin_weight false --loss_weight_x 1. --loss_weight_obj 4.6 --loss_weight_cons 5.3 --conv genconv --wandbname CaucGEN

## fac (GEN)
python testing.py --ckpt logs/FacGEN/run1/best_model.pt --ipm_alpha 0.8 --micro_batch 4 --batchsize 4 --hidden 180 --num_pred_layers 4 --num_mlp_layers 2 --share_lin_weight false --conv_sequence cov --loss_weight_x 1. --loss_weight_obj 2.99 --loss_weight_cons 8.15 --conv genconv --wandbname FacGEN

## Hrand (GEN)
python testing.py --ckpt logs/Hrand/run1/best_model.pt --ipm_alpha 0.8 --micro_batch 4 --batchsize 4 --hidden 180 --num_pred_layers 4 --num_mlp_layers 2 --share_lin_weight false --conv_sequence cov --loss_weight_x 1. --loss_weight_obj 2.99 --loss_weight_cons 8.15 --conv genconv --wandbname Hrandon

# GCNConv Testing Commands

## small setcover (GCN)
python testing.py --ckpt logs/SetcoverGCN/run1/best_model.pt --ipm_alpha 0.76 --micro_batch 4 --batchsize 4 --hidden 180 --num_pred_layers 3 --num_mlp_layers 4 --share_lin_weight false --conv_sequence cov --loss_weight_x 1. --loss_weight_obj 0.33 --loss_weight_cons 2.2 --conv gcnconv --wandbname SetcGCN_small

## large setcover (GCN)
python testing.py --ckpt logs/SetcoverGCN/run1/best_model.pt --ipm_alpha 0.2 --micro_batch 4 --batchsize 4 --hidden 180 --num_pred_layers 3 --num_mlp_layers 4 --share_lin_weight false --conv_sequence cov --loss_weight_x 1. --loss_weight_obj 2.2 --loss_weight_cons 0.26 --conv gcnconv --wandbname SetcGCN_large

## indset (GCN)
python testing.py --ckpt logs/IndsetGCN/run1/best_model.pt --ipm_alpha 0.5 --micro_batch 4 --batchsize 4 --hidden 180 --num_pred_layers 4 --num_mlp_layers 4 --share_lin_weight false --conv_sequence cov --loss_weight_x 1. --loss_weight_obj 4.5 --loss_weight_cons 9.6 --conv gcnconv --wandbname IndsGCN

## cauction (GCN)
python testing.py --ckpt logs/CaucGCN/run1/best_model.pt --ipm_alpha 0.35 --micro_batch 4 --batchsize 4 --hidden 180 --num_pred_layers 4 --num_mlp_layers 4 --share_lin_weight false --loss_weight_x 1. --loss_weight_obj 3.43 --loss_weight_cons 5.8 --conv gcnconv --wandbname CaucGCN

## fac (GCN)
python testing.py --ckpt logs/FacGCN/run1/best_model.pt --ipm_alpha 0.63 --micro_batch 4 --batchsize 4 --hidden 96 --num_pred_layers 4 --num_mlp_layers 4 --share_lin_weight false --conv_sequence cov --loss_weight_x 1. --loss_weight_obj 8.68 --loss_weight_cons 9.56 --conv gcnconv --wandbname FacGCN

# GINConv Testing Commands

## small setcover (GIN)
python testing.py --ckpt logs/SetcoverGIN/run1/best_model.pt --ipm_alpha 0.73 --micro_batch 4 --batchsize 4 --hidden 180 --num_pred_layers 3 --num_mlp_layers 2 --share_lin_weight false --conv_sequence cov --loss_weight_x 1. --loss_weight_obj 2.4 --loss_weight_cons 7.5 --conv ginconv --wandbname SetcGIN_small

## large setcover (GIN)
python testing.py --ckpt logs/SetcoverGIN/run1/best_model.pt --ipm_alpha 0.7 --micro_batch 4 --batchsize 4 --hidden 180 --num_pred_layers 3 --num_mlp_layers 4 --share_lin_weight false --conv_sequence cov --loss_weight_x 1. --loss_weight_obj 4.5 --loss_weight_cons 2.2 --conv ginconv --wandbname SetcGIN_large

## indset (GIN)
python testing.py --ckpt logs/IndsetGIN/run1/best_model.pt --ipm_alpha 0.73 --micro_batch 4 --batchsize 4 --hidden 180 --num_pred_layers 3 --num_mlp_layers 2 --share_lin_weight false --conv_sequence cov --loss_weight_x 1. --loss_weight_obj 2.4 --loss_weight_cons 7.5 --conv ginconv --wandbname IndsGIN

## cauction (GIN)
python testing.py --ckpt logs/CaucGIN/run1/best_model.pt --ipm_alpha 0.63 --micro_batch 4 --batchsize 4 --hidden 180 --num_pred_layers 4 --num_mlp_layers 4 --share_lin_weight false --loss_weight_x 1. --loss_weight_obj 4.3 --loss_weight_cons 6.26 --conv ginconv --wandbname CaucGIN

## small fac (GIN)
python testing.py --ckpt logs/FacGINSmall/run1/best_model.pt --ipm_alpha 0.8 --micro_batch 4 --batchsize 4 --hidden 180 --num_pred_layers 4 --num_mlp_layers 4 --share_lin_weight false --conv_sequence cov --loss_weight_x 1. --loss_weight_obj 1.3 --loss_weight_cons 4.6 --conv ginconv --wandbname FacGIN_small

## large fac (GIN)
python testing.py --ckpt logs/FacGINLarge/run1/best_model.pt --ipm_alpha 0.9 --micro_batch 4 --batchsize 4 --hidden 128 --num_pred_layers 4 --num_mlp_layers 4 --share_lin_weight false --conv_sequence cov --loss_weight_x 1. --loss_weight_obj 2.5 --loss_weight_cons 4.0 --conv ginconv --wandbname FacGIN_large