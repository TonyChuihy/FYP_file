import torch
import numpy as np

class TorchLPSolver:
    def __init__(self, device='cuda'):
        self.device = device if torch.cuda.is_available() else 'cpu'
        print(f"使用设备: {self.device}")
    
    def simplex_method(self, c, A, b, max_iter=1000):
        """使用PyTorch实现单纯形法"""
        # 转移到GPU，使用float64提高数值稳定性
        c_tensor = torch.tensor(c, dtype=torch.float64, device=self.device)
        A_tensor = torch.tensor(A, dtype=torch.float64, device=self.device) 
        b_tensor = torch.tensor(b, dtype=torch.float64, device=self.device)
        
        m, n = A_tensor.shape
        
        # 初始基（选择线性无关的列）
        basis = self._select_initial_basis(A_tensor)
        print(f"初始基选择完成: {len(basis)}/{m} 个基变量")
        
        for iteration in range(max_iter):
            try:
                # 基矩阵
                B = A_tensor[:, basis]
                
                # 求解基系统 B x_B = b
                x_B = torch.linalg.solve(B, b_tensor)
                
                # 计算 reduced costs
                reduced_costs = self._compute_reduced_costs(c_tensor, A_tensor, B, basis)
                
                # 检查最优性
                if torch.all(reduced_costs >= -1e-6):
                    print(f"在第 {iteration} 次迭代找到最优解")
                    break
                
                # 选择进基变量
                entering = torch.argmin(reduced_costs).item()
                
                # 计算方向
                d = torch.linalg.solve(B, A_tensor[:, entering])
                
                # 比值测试选择离基变量
                ratios = torch.where(d > 1e-10, x_B / d, torch.tensor(float('inf'), device=self.device, dtype=torch.float64))
                min_ratio, leaving_idx = torch.min(ratios, 0)
                
                if min_ratio == float('inf'):
                    print("问题无界")
                    break
                
                # 更新基
                basis[leaving_idx.item()] = entering
                
                if iteration % 100 == 0:
                    print(f"迭代 {iteration}, 目标值: {torch.dot(c_tensor[basis], x_B).item():.4f}")
                
            except torch.linalg.LinAlgError:
                print(f"迭代 {iteration}: 基矩阵奇异，重新选择基")
                basis = self._repair_basis(A_tensor)
        
        # 构造最终解 - 修复数据类型问题
        x_optimal = torch.zeros(n, device=self.device, dtype=torch.float64)  # 明确指定dtype
        x_B_final = torch.linalg.solve(A_tensor[:, basis], b_tensor)
        x_optimal[basis] = x_B_final
        objective = torch.dot(c_tensor, x_optimal)
        
        return x_optimal.cpu().numpy(), objective.cpu().item()
    
    def _select_initial_basis(self, A):
        """选择初始基 - 修正版本"""
        m, n = A.shape
        
        # 方法1: 使用SVD选择线性无关的列（更稳定）
        try:
            U, S, Vh = torch.linalg.svd(A, full_matrices=False)
            rank = torch.sum(S > 1e-10).item()
            
            if rank < m:
                print(f"警告: 矩阵秩不足 ({rank} < {m})")
                return self._select_initial_basis_simple(A)
            
            # 选择前m个最重要的列
            basis = []
            for i in range(m):
                col_idx = torch.argmax(torch.abs(Vh[i, :])).item()
                if col_idx not in basis:
                    basis.append(col_idx)
                else:
                    # 如果重复，选择下一个最大的
                    for j in range(n):
                        if j not in basis:
                            basis.append(j)
                            break
                
                if len(basis) == m:
                    break
                    
            return basis[:m]
            
        except:
            # 如果SVD失败，使用简单方法
            return self._select_initial_basis_simple(A)
    
    def _select_initial_basis_simple(self, A):
        """简单的初始基选择方法"""
        m, n = A.shape
        basis = []
        
        # 选择前m个线性无关的列
        for i in range(n):
            if len(basis) == m:
                break
                
            candidate_basis = basis + [i]
            B = A[:, candidate_basis]
            
            # 检查矩阵是否满秩
            if torch.linalg.matrix_rank(B) == len(candidate_basis):
                basis = candidate_basis
        
        # 如果找不到足够的线性无关列，用随机列填充
        if len(basis) < m:
            print(f"警告: 只找到 {len(basis)} 个线性无关列，用随机列填充")
            for i in range(n):
                if i not in basis and len(basis) < m:
                    basis.append(i)
        
        return basis
    
    def _compute_reduced_costs(self, c, A, B, basis):
        """计算reduced costs"""
        m, n = A.shape
        
        # 计算对偶变量 y = B^{-T} c_B
        c_B = c[basis]
        y = torch.linalg.solve(B.T, c_B)
        
        # 计算所有变量的reduced costs: r = c - A^T y
        reduced_costs = c - A.T @ y
        
        # 基变量的reduced cost应该为0（数值误差范围内）
        reduced_costs[basis] = 0
        
        return reduced_costs
    
    def _repair_basis(self, A):
        """修复奇异基"""
        print("执行基修复...")
        return self._select_initial_basis_simple(A)

# 使用示例
def test_torch_solver():
    solver = TorchLPSolver()
    
    # 生成更好的测试问题（确保有可行解）
    m, n = 200, 400
    
    # 生成一个满秩矩阵
    A = np.random.randn(m, n)
    # 确保矩阵行满秩
    U, S, Vt = np.linalg.svd(A, full_matrices=False)
    S = np.maximum(S, 0.1)  # 确保所有奇异值都不为0
    A = U @ np.diag(S) @ Vt
    
    # 生成一个可行解
    x_feasible = np.random.rand(n) * 2 + 0.1  # 正数解
    b = A @ x_feasible
    c = np.random.randn(n)
    
    print(f"问题规模: {m} x {n}")
    print(f"矩阵A的秩: {np.linalg.matrix_rank(A)}")
    
    # 求解
    x_opt, obj_value = solver.simplex_method(c, A, b)
    
    print(f"\n=== 结果 ===")
    print(f"最优目标值: {obj_value:.4f}")
    print(f"原始目标值: {c @ x_feasible:.4f}")
    print(f"解向量范数: {np.linalg.norm(x_opt):.4f}")
    print(f"约束违反: {np.linalg.norm(A @ x_opt - b):.6f}")
    
    # 验证基变量的选择
    basis_indices = np.where(np.abs(x_opt) > 1e-6)[0]
    print(f"基变量数量: {len(basis_indices)}")
    
    return x_opt, obj_value

if __name__ == "__main__":
    x_opt, obj_value = test_torch_solver()