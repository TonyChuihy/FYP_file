import torch, sys, os
print("python:", sys.executable)
print("torch:", torch.__version__)
print("cuda_available:", torch.cuda.is_available())
print("cuda_device_count:", torch.cuda.device_count())
if torch.cuda.is_available():
    print("current device:", torch.cuda.current_device(), "name:", torch.cuda.get_device_name(0))
    
import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

from solver.linprog import linprog
from tqdm import tqdm

import gzip
import pickle
import torch
from scipy.linalg import LinAlgWarning
from scipy.optimize._optimize import OptimizeWarning
import warnings
import numpy as np
from functools import partial

from pathlib import Path

from generate_instances import generate_random_lp, generate_setcover, Graph, generate_indset, generate_cauctions, generate_capacited_facility_location

rng = np.random.RandomState(1)

bounds = (0., 1.)

root = os.getcwd() + '/d/fac6'
root

######
def surrogate_gen():
    nvars = 1000
    nconstraints = 1000
    density = 0.9
    A, b, c = generate_random_lp(nvars=nvars, nconstraints=nconstraints, density=density, rng=rng)
    return A, b, c
######

warnings.filterwarnings("error")
#cauction
ips = []
pkg_idx = "1k_1k_09_dense"
success_cnt = 0
fail_cnt = 0

max_iter = 15000
num = 100

for i in tqdm(range(max_iter)):
    A, b, c = surrogate_gen()
    
    try:
        A_eq = None
        b_eq = None
        A_ub = A
        b_ub = b
        res = linprog(c, 
                A_ub=A_ub,
                b_ub=b_ub,
                A_eq=A_eq, b_eq=b_eq, bounds=bounds, method='interior-point')
        print("Size of A:", A.shape)
    except (LinAlgWarning, OptimizeWarning, AssertionError):
        fail_cnt += 1
        continue
    else:
        if res.success and not np.isnan(res.fun):
            ips.append((torch.from_numpy(A).to(torch.float).to(device), torch.from_numpy(b).to(torch.float).to(device), torch.from_numpy(c).to(torch.float).to(device)))
            success_cnt += 1

    if len(ips) >= 1000 or success_cnt == num:
        out_path = Path(f'{root}/raw/instance_{pkg_idx}.pkl.gz')
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with gzip.open(out_path, "wb") as file:
            pickle.dump(ips, file)
            pkg_idx = 'used_idx'
        ips = []

    if success_cnt >= num:
        break

warnings.resetwarnings()
