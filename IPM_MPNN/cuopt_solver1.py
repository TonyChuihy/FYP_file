# 使用 NVIDIA cuPY 结合其他求解器
import cupy as cp
from scipy.optimize import linprog

# 在 GPU 上处理数据，用传统求解器计算
A = cp.array([[1, 2], [3, 4]])
b = cp.array([5, 6])
c = cp.array([1, 2])

# 转换到 CPU 使用 scipy
A_cpu = A.get()
b_cpu = b.get() 
c_cpu = c.get()

result = linprog(c_cpu, A_eq=A_cpu, b_eq=b_cpu)