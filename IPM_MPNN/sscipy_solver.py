# solve_indset_with_scipy.py
import gzip
import pickle
from typing import List, Tuple, Optional, Dict
import numpy as np
import warnings

# SciPy 的 linprog
from scipy.optimize import linprog
from scipy.linalg import LinAlgWarning
from scipy.optimize._optimize import OptimizeWarning
import time
import wandb

import torch

## python scipy_solver.py --input d/fac6/raw/Setcover.pkl.gz --wandbname scipy_on_Setc


def to_numpy(x):
    """把 numpy/torch 物件轉成 numpy.ndarray"""
    if isinstance(x, np.ndarray):
        return x
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    # 若是其他（例如 Python list），嘗試轉成 ndarray
    return np.asarray(x)

def solve_single_instance(A, b, c, using_ineq=True, upper_bound: Optional[float]=1.0, method='highs', options=None) -> Dict:
    """
    用 SciPy 的 linprog 求解單一 (A, b, c)。
    參數:
      - A: 2D array-like (constraints x vars)
      - b: 1D array-like (constraints)
      - c: 1D array-like (vars)
      - using_ineq: 若 True，視作 A_ub x <= b_ub；否則視作 A_eq x == b_eq
      - upper_bound: 上界（None 表示無上界）
      - method: linprog 的 method（'highs' 推薦）
    回傳 dict 包含 success, fun, x, status, message
    """
    A = to_numpy(A)
    b = to_numpy(b).ravel()
    c = to_numpy(c).ravel()

    # 保險起見，把 c 做簡單正規化（與原 repo 做法相同）
    denom = (np.abs(c).max() + 1e-10)
    c_norm = c / denom

    # 設定 bounds: (0, upper_bound) for each variable
    if upper_bound is None:
        bounds = [(0, None)] * c_norm.size
    else:
        bounds = [(0, float(upper_bound))] * c_norm.size

    # 準備 A_ub / A_eq
    if using_ineq:
        A_ub = A
        b_ub = b
        A_eq = None
        b_eq = None
    else:
        A_eq = A
        b_eq = b
        A_ub = None
        b_ub = None

    # 呼叫 linprog；捕捉潛在警告為例
    try:
        with warnings.catch_warnings():
            # 將數值警告視為錯誤或記錄取決需求；這裡只 suppress 部分警告
            warnings.simplefilter("ignore", LinAlgWarning)
            warnings.simplefilter("ignore", OptimizeWarning)
            res = linprog(c_norm, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq,
                          bounds=bounds, method=method, options=options)
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "fun": None,
            "x": None,
            "status": -999,
            "message": "exception during solve"
        }

    # 注意：我們使用的是正規化後的 c_norm，因此若你需要原始 obj value，需要反正規化：
    obj_val = None if res.fun is None else (res.fun * denom)

    return {
        "success": bool(res.success),
        "fun": float(obj_val) if obj_val is not None else None,
        "x": None if res.x is None else res.x,   # numpy array
        "status": int(res.status) if hasattr(res, 'status') else None,
        "message": str(res.message) if hasattr(res, 'message') else None,
        "raw_result": res  # 若你想要完整的 scipy OptimizeResult
    }

def solve_instances_from_gz(input_gz_path: str,
                             output_gz_path: Optional[str] = None,
                             using_ineq=True,
                             upper_bound: Optional[float]=1.0,
                             method='highs',
                             max_to_process: Optional[int] = None):
    """
    讀取 input_gz_path（gzipped pickle），內含 list of (A,b,c) 或類似格式。
    對每個實例使用 SciPy linprog 求解，並回傳 results list；若指定 output_gz_path，會把結果寫回 gzipped pickle。
    """
    # load
    with gzip.open(input_gz_path, 'rb') as f:
        instances = pickle.load(f)

    results = []
    n = len(instances)
    if max_to_process is not None:
        n = min(n, max_to_process)

    for idx in range(n):
        A, b, c = instances[idx]
        res = solve_single_instance(A, b, c, using_ineq=using_ineq, upper_bound=upper_bound, method=method)
        results.append(res)

    if output_gz_path:
        with gzip.open(output_gz_path, 'wb') as f:
            pickle.dump(results, f, protocol=pickle.HIGHEST_PROTOCOL)

    return results

if __name__ == '__main__':
    # 範例用法
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, required=True, help='input gzipped pickle path, e.g. d/indset/raw/indset.gz')
    parser.add_argument('--output', type=str, default=None, help='optional output gz path to save results')
    parser.add_argument('--upper', type=float, default=1.0)
    parser.add_argument('--method', type=str, default='highs')
    parser.add_argument('--use_wandb', type=str, default='true', help='whether to log to wandb')
    parser.add_argument('--wandbproject', type=str, default='ipm_mpnn')
    parser.add_argument('--wandbname', type=str, default='scipy_solver_run')
    args = parser.parse_args()
    # initialize wandb (can be disabled via --use_wandb=false)
    use_wandb = args.use_wandb.lower() in ('1', 'true', 'yes')
    wandb.init(project=args.wandbproject,
               name=args.wandbname if args.wandbname else None,
               config={'input': args.input, 'method': args.method, 'upper': args.upper},
               mode='online' if use_wandb else 'disabled')

    # load instances and time each solve individually
    with gzip.open(args.input, 'rb') as f:
        instances = pickle.load(f)

    results = []
    times = []
    for idx, inst in enumerate(instances):
        A, b, c = inst
        t0 = time.perf_counter()
        res = solve_single_instance(A, b, c, using_ineq=True, upper_bound=args.upper, method=args.method)
        t1 = time.perf_counter()
        elapsed = t1 - t0
        times.append(elapsed)
        results.append(res)

    total_time = sum(times)
    n = len(times)
    avg_time = total_time / n if n > 0 else 0.0

    # log timing-only metrics to wandb
    wandb.log({
        'test_infer_time_total_s': total_time,
        'test_infer_time_per_instance_s': avg_time,
        'test_num_instances': n
    })

    # optionally save results (keeps previous behavior)
    if args.output:
        with gzip.open(args.output, 'wb') as f:
            pickle.dump(results, f, protocol=pickle.HIGHEST_PROTOCOL)

    # print a short summary
    print(f'Processed {n} instances; total_time={total_time:.6f}s; avg_time={avg_time:.6f}s')
    # print first 3 results for quick check
    for i, r in enumerate(results[:3]):
        print(f'instance {i}: success={r["success"]}, fun={r["fun"]}, status={r["status"]}, msg={r["message"]}')
    wandb.finish()