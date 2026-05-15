#calculus_tools.py
"""
统一的考研数学工具包
整合了微积分工具（基于sympy）和线性代数工具（基于numpy）。
"""
import numpy as np
import math
from typing import List,Tuple,Union,Optional
from scipy import stats
import warnings
warnings.filterwarnings("ignore")

from sympy import symbols, sympify, diff, integrate, limit, solve, simplify, expand, factor

# --- 第一部分：导入/兼容您现有的微积分工具 ---
# 为保持清晰，这里直接内嵌了您提供的函数，实际中可以从 calculus_tools.py 导入
# 假设您的工具使用 'x', 'y' 作为默认符号
x, y = symbols("x y")

class CalculusTools:
    """微积分工具类（基于您现有代码的封装）"""
    
    @staticmethod
    def diff(expr, var="x"):
        try:
            expr = sympify(expr)
            var_sym = symbols(var)
            result = diff(expr, var_sym)
            return {"success": True, "result": str(result)}
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    @staticmethod
    def integrate(expr, var="x"):
        try:
            expr = sympify(expr)
            var_sym = symbols(var)
            result = integrate(expr, var_sym)
            return {"success": True, "result": str(result)}
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    @staticmethod
    def limit(expr, var="x", point=0):
        try:
            expr = sympify(expr)
            var_sym = symbols(var)
            result = limit(expr, var_sym, point)
            return {"success": True, "result": str(result)}
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    @staticmethod
    def solve(expr, var="x"):
        try:
            expr = sympify(expr)
            var_sym = symbols(var)
            result = solve(expr, var_sym)
            return {"success": True, "result": str(result)}
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    @staticmethod
    def simplify(expr):
        try:
            expr = sympify(expr)
            result = simplify(expr)
            return {"success": True, "result": str(result)}
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    @staticmethod
    def expand(expr):
        try:
            expr = sympify(expr)
            result = expand(expr)
            return {"success": True, "result": str(result)}
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    @staticmethod
    def factor(expr):
        try:
            expr = sympify(expr)
            result = factor(expr)
            return {"success": True, "result": str(result)}
        except Exception as e:
            return {"success": False, "error": str(e)}


# --- 第二部分：线性代数工具（基于我之前提供的代码，调整了返回格式）---
class LinearAlgebraTools:
    """线性代数工具类（返回格式与微积分工具保持一致）"""
    
    def __init__(self):
        self.epsilon = 1e-10
    
    def _create_matrix(self, data):
        """辅助方法：创建矩阵"""
        return np.array(data, dtype=float)
    
    def _format_matrix_result(self, result):
        """格式化矩阵结果为字符串（用于返回）"""
        if isinstance(result, np.ndarray):
            # 如果是小矩阵，直接返回字符串表示
            if result.size <= 25:  # 5x5以内
                return str(result.tolist())
            else:
                return f"Matrix(shape={result.shape})"
        else:
            return str(result)
    
    def determinant(self, matrix_data):
        """计算行列式 |A|"""
        try:
            A = self._create_matrix(matrix_data)
            if A.shape[0] != A.shape[1]:
                return {"success": False, "error": "Matrix must be square to compute determinant."}
            
            if A.shape[0] == 1:
                det = A[0, 0]
            elif A.shape[0] == 2:
                det = A[0, 0] * A[1, 1] - A[0, 1] * A[1, 0]
            elif A.shape[0] == 3:
                det = (A[0,0]*A[1,1]*A[2,2] + A[0,1]*A[1,2]*A[2,0] + A[0,2]*A[1,0]*A[2,1] 
                      - A[0,2]*A[1,1]*A[2,0] - A[0,1]*A[1,0]*A[2,2] - A[0,0]*A[1,2]*A[2,1])
            else:
                det = np.linalg.det(A)
            
            return {"success": True, "result": float(det)}
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def inverse(self, matrix_data):
        """计算逆矩阵 A^{-1}"""
        try:
            A = self._create_matrix(matrix_data)
            if A.shape[0] != A.shape[1]:
                return {"success": False, "error": "Matrix must be square to compute inverse."}
            
            det = np.linalg.det(A)
            if abs(det) < self.epsilon:
                return {"success": False, "error": "Matrix is singular (determinant is 0)."}
            
            inv_A = np.linalg.inv(A)
            return {"success": True, "result": self._format_matrix_result(inv_A)}
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def rank(self, matrix_data):
        """计算矩阵的秩"""
        try:
            A = self._create_matrix(matrix_data)
            rank_val = np.linalg.matrix_rank(A)
            return {"success": True, "result": int(rank_val)}
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def solve_linear_system(self, A_data, b_data):
        """解线性方程组 Ax = b"""
        try:
            A = self._create_matrix(A_data)
            b = self._create_matrix(b_data)
            
            if A.shape[0] != b.shape[0]:
                return {"success": False, "error": f"Shape mismatch: A{A.shape} vs b{b.shape}"}
            
            x = np.linalg.solve(A, b)
            return {"success": True, "result": self._format_matrix_result(x)}
        except np.linalg.LinAlgError as e:
            return {"success": False, "error": f"线性方程组求解失败: {str(e)}"}
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def eigenvalues(self, matrix_data):
        """计算特征值"""
        try:
            A = self._create_matrix(matrix_data)
            if A.shape[0] != A.shape[1]:
                return {"success": False, "error": "Matrix must be square to compute eigenvalues."}
            
            eigvals = np.linalg.eigvals(A)
            # 将复数转换为可读字符串
            eigvals_str = [str(v) for v in eigvals]
            return {"success": True, "result": eigvals_str}
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def matrix_multiply(self, A_data, B_data):
        """矩阵乘法 A × B"""
        try:
            A = self._create_matrix(A_data)
            B = self._create_matrix(B_data)
            
            if A.shape[1] != B.shape[0]:
                return {"success": False, "error": f"Shape mismatch for multiplication: A{A.shape} vs B{B.shape}"}
            
            result = np.dot(A, B)
            return {"success": True, "result": self._format_matrix_result(result)}
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def gram_schmidt(self, vectors_data, normalize=True):
        """施密特正交化"""
        try:
            # 输入是向量列表
            vectors = [self._create_matrix(v) if isinstance(v, list) else v for v in vectors_data]
            
            # 转换为列向量
            vectors = [v.reshape(-1, 1) if len(v.shape) == 1 else v for v in vectors]
            
            ortho_vectors = []
            for v in vectors:
                w = v.copy()
                for u in ortho_vectors:
                    proj = (np.dot(v.T, u) / np.dot(u.T, u)) * u
                    w = w - proj
                
                norm = np.linalg.norm(w)
                if norm > self.epsilon:
                    if normalize:
                        w = w / norm
                    ortho_vectors.append(w)
            
            # 转换回列表格式
            result = [v.flatten().tolist() for v in ortho_vectors]
            return {"success": True, "result": result}
        except Exception as e:
            return {"success": False, "error": str(e)}




class ProbabilityTools:
    """概率论与数理统计工具类"""
    
    def __init__(self):
        # 常用常数
        self.sqrt_pi = math.sqrt(math.pi)
        self.sqrt_2pi = math.sqrt(2 * math.pi)
        
    # ==================== 基础概率计算 ====================
    
    def combination(self, n: int, k: int) -> dict:
        """组合数 C(n, k) = n! / (k! * (n-k)!)"""
        try:
            if n < 0 or k < 0 or k > n:
                return {"success": False, "error": f"Invalid parameters: n={n}, k={k}"}
            
            result = math.comb(n, k) if hasattr(math, 'comb') else self._comb_manual(n, k)
            return {"success": True, "result": result}
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def _comb_manual(self, n: int, k: int) -> int:
        """手动计算组合数（兼容Python < 3.8）"""
        k = min(k, n - k)
        if k == 0:
            return 1
        result = 1
        for i in range(1, k + 1):
            result = result * (n - k + i) // i
        return result
    
    def permutation(self, n: int, k: int) -> dict:
        """排列数 A(n, k) = n! / (n-k)!"""
        try:
            if n < 0 or k < 0 or k > n:
                return {"success": False, "error": f"Invalid parameters: n={n}, k={k}"}
            
            result = math.perm(n, k) if hasattr(math, 'perm') else self._perm_manual(n, k)
            return {"success": True, "result": result}
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def _perm_manual(self, n: int, k: int) -> int:
        """手动计算排列数"""
        result = 1
        for i in range(n, n - k, -1):
            result *= i
        return result
    
    def factorial(self, n: int) -> dict:
        """阶乘 n!"""
        try:
            if n < 0:
                return {"success": False, "error": f"阶乘未定义: n={n}"}
            result = math.factorial(n)
            return {"success": True, "result": result}
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    # ==================== 离散型随机变量分布 ====================
    
    def binomial_pmf(self, k: int, n: int, p: float) -> dict:
        """二项分布概率质量函数 P(X = k) = C(n,k) * p^k * (1-p)^(n-k)"""
        try:
            if not 0 <= p <= 1:
                return {"success": False, "error": f"概率p必须在[0,1]之间: p={p}"}
            if k < 0 or k > n:
                return {"success": True, "result": 0.0}  # 概率为0
            
            comb = math.comb(n, k) if hasattr(math, 'comb') else self._comb_manual(n, k)
            prob = comb * (p ** k) * ((1 - p) ** (n - k))
            return {"success": True, "result": float(prob)}
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def binomial_cdf(self, k: int, n: int, p: float) -> dict:
        """二项分布累积分布函数 P(X ≤ k)"""
        try:
            if not 0 <= p <= 1:
                return {"success": False, "error": f"概率p必须在[0,1]之间: p={p}"}
            
            prob = 0.0
            for i in range(k + 1):
                comb = math.comb(n, i) if hasattr(math, 'comb') else self._comb_manual(n, i)
                prob += comb * (p ** i) * ((1 - p) ** (n - i))
            return {"success": True, "result": float(prob)}
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def poisson_pmf(self, k: int, lam: float) -> dict:
        """泊松分布概率质量函数 P(X = k) = (λ^k * e^{-λ}) / k!"""
        try:
            if lam <= 0:
                return {"success": False, "error": f"λ必须大于0: λ={lam}"}
            if k < 0:
                return {"success": True, "result": 0.0}
            
            prob = (math.exp(-lam) * (lam ** k)) / math.factorial(k)
            return {"success": True, "result": float(prob)}
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def poisson_cdf(self, k: int, lam: float) -> dict:
        """泊松分布累积分布函数 P(X ≤ k)"""
        try:
            if lam <= 0:
                return {"success": False, "error": f"λ必须大于0: λ={lam}"}
            
            prob = 0.0
            for i in range(k + 1):
                prob += (math.exp(-lam) * (lam ** i)) / math.factorial(i)
            return {"success": True, "result": float(prob)}
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def geometric_pmf(self, k: int, p: float) -> dict:
        """几何分布概率质量函数 P(X = k) = (1-p)^(k-1) * p, k=1,2,3,..."""
        try:
            if not 0 < p <= 1:
                return {"success": False, "error": f"概率p必须在(0,1]之间: p={p}"}
            if k < 1:
                return {"success": True, "result": 0.0}
            
            prob = ((1 - p) ** (k - 1)) * p
            return {"success": True, "result": float(prob)}
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    # ==================== 连续型随机变量分布 ====================
    
    def normal_pdf(self, x: float, mu: float = 0, sigma: float = 1) -> dict:
        """正态分布概率密度函数"""
        try:
            if sigma <= 0:
                return {"success": False, "error": f"σ必须大于0: σ={sigma}"}
            
            exponent = -((x - mu) ** 2) / (2 * sigma ** 2)
            density = (1 / (sigma * self.sqrt_2pi)) * math.exp(exponent)
            return {"success": True, "result": float(density)}
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def normal_cdf(self, x: float, mu: float = 0, sigma: float = 1) -> dict:
        """正态分布累积分布函数 P(X ≤ x)"""
        try:
            if sigma <= 0:
                return {"success": False, "error": f"σ必须大于0: σ={sigma}"}
            
            z = (x - mu) / sigma
            prob = 0.5 * (1 + math.erf(z / math.sqrt(2)))  # 使用误差函数
            return {"success": True, "result": float(prob)}
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def normal_quantile(self, p: float, mu: float = 0, sigma: float = 1) -> dict:
        """正态分布分位数函数（逆CDF）"""
        try:
            if not 0 < p < 1:
                return {"success": False, "error": f"概率p必须在(0,1)之间: p={p}"}
            if sigma <= 0:
                return {"success": False, "error": f"σ必须大于0: σ={sigma}"}
            
            # 使用scipy的norm.ppf（更精确）
            from scipy.stats import norm
            quantile = norm.ppf(p, loc=mu, scale=sigma)
            return {"success": True, "result": float(quantile)}
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def exponential_pdf(self, x: float, lam: float) -> dict:
        """指数分布概率密度函数 f(x) = λe^{-λx}, x ≥ 0"""
        try:
            if lam <= 0:
                return {"success": False, "error": f"λ必须大于0: λ={lam}"}
            
            if x < 0:
                density = 0.0
            else:
                density = lam * math.exp(-lam * x)
            return {"success": True, "result": float(density)}
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def exponential_cdf(self, x: float, lam: float) -> dict:
        """指数分布累积分布函数 F(x) = 1 - e^{-λx}, x ≥ 0"""
        try:
            if lam <= 0:
                return {"success": False, "error": f"λ必须大于0: λ={lam}"}
            
            if x < 0:
                prob = 0.0
            else:
                prob = 1 - math.exp(-lam * x)
            return {"success": True, "result": float(prob)}
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def uniform_pdf(self, x: float, a: float, b: float) -> dict:
        """均匀分布概率密度函数 f(x) = 1/(b-a), a ≤ x ≤ b"""
        try:
            if a >= b:
                return {"success": False, "error": f"参数需满足a < b: a={a}, b={b}"}
            
            if a <= x <= b:
                density = 1.0 / (b - a)
            else:
                density = 0.0
            return {"success": True, "result": float(density)}
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def uniform_cdf(self, x: float, a: float, b: float) -> dict:
        """均匀分布累积分布函数"""
        try:
            if a >= b:
                return {"success": False, "error": f"参数需满足a < b: a={a}, b={b}"}
            
            if x < a:
                prob = 0.0
            elif x > b:
                prob = 1.0
            else:
                prob = (x - a) / (b - a)
            return {"success": True, "result": float(prob)}
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    # ==================== 随机变量数字特征 ====================
    
    def expectation(self, values: List[float], probs: Optional[List[float]] = None) -> dict:
        """计算期望 E[X] = Σ x_i * p_i 或 样本均值"""
        try:
            if probs is None:
                # 计算样本均值
                if not values:
                    return {"success": False, "error": "数值列表不能为空"}
                result = sum(values) / len(values)
                return {"success": True, "result": float(result)}
            else:
                # 计算加权期望
                if len(values) != len(probs):
                    return {"success": False, "error": f"数值和概率长度不匹配: {len(values)} != {len(probs)}"}
                if abs(sum(probs) - 1.0) > 1e-10:
                    return {"success": False, "error": f"概率和不等于1: sum(probs)={sum(probs)}"}
                
                result = sum(v * p for v, p in zip(values, probs))
                return {"success": True, "result": float(result)}
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def variance(self, values: List[float], probs: Optional[List[float]] = None) -> dict:
        """计算方差 Var(X) = E[X^2] - (E[X])^2"""
        try:
            if probs is None:
                # 计算样本方差（无偏估计）
                if len(values) < 2:
                    return {"success": False, "error": "样本量至少为2"}
                mean = sum(values) / len(values)
                squared_diff = sum((x - mean) ** 2 for x in values)
                result = squared_diff / (len(values) - 1)  # 无偏估计
                return {"success": True, "result": float(result)}
            else:
                # 计算概率分布的方差
                if len(values) != len(probs):
                    return {"success": False, "error": f"数值和概率长度不匹配: {len(values)} != {len(probs)}"}
                
                # 计算 E[X]
                ex = sum(v * p for v, p in zip(values, probs))
                # 计算 E[X^2]
                ex2 = sum((v ** 2) * p for v, p in zip(values, probs))
                result = ex2 - ex ** 2
                return {"success": True, "result": float(result)}
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def covariance(self, x_values: List[float], y_values: List[float], 
                  x_probs: Optional[List[float]] = None, 
                  y_probs: Optional[List[float]] = None) -> dict:
        """计算协方差 Cov(X,Y) = E[XY] - E[X]E[Y]"""
        try:
            if len(x_values) != len(y_values):
                return {"success": False, "error": f"X和Y数据长度不匹配: {len(x_values)} != {len(y_values)}"}
            
            if x_probs is None and y_probs is None:
                # 计算样本协方差
                n = len(x_values)
                if n < 2:
                    return {"success": False, "error": "样本量至少为2"}
                
                x_mean = sum(x_values) / n
                y_mean = sum(y_values) / n
                
                cov = sum((x - x_mean) * (y - y_mean) for x, y in zip(x_values, y_values)) / (n - 1)
                return {"success": True, "result": float(cov)}
            elif x_probs is not None and y_probs is not None:
                # 联合分布的协方差
                if len(x_probs) != len(y_probs):
                    return {"success": False, "error": f"概率长度不匹配: {len(x_probs)} != {len(y_probs)}"}
                
                # 计算 E[X], E[Y], E[XY]
                ex = sum(x * p for x, p in zip(x_values, x_probs))
                ey = sum(y * p for y, p in zip(y_values, y_probs))
                exy = sum(x * y * px for x, y, px in zip(x_values, y_values, x_probs))
                
                cov = exy - ex * ey
                return {"success": True, "result": float(cov)}
            else:
                return {"success": False, "error": "必须同时提供x_probs和y_probs，或都不提供"}
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def correlation(self, x_values: List[float], y_values: List[float]) -> dict:
        """计算相关系数 ρ = Cov(X,Y) / (σ_X * σ_Y)"""
        try:
            if len(x_values) != len(y_values):
                return {"success": False, "error": f"数据长度不匹配: {len(x_values)} != {len(y_values)}"}
            if len(x_values) < 2:
                return {"success": False, "error": "样本量至少为2"}
            
            # 计算样本协方差
            cov_result = self.covariance(x_values, y_values)
            if not cov_result["success"]:
                return cov_result
            cov = cov_result["result"]
            
            # 计算样本标准差
            x_var_result = self.variance(x_values)
            y_var_result = self.variance(y_values)
            
            if not x_var_result["success"]:
                return x_var_result
            if not y_var_result["success"]:
                return y_var_result
            
            x_std = math.sqrt(x_var_result["result"])
            y_std = math.sqrt(y_var_result["result"])
            
            if x_std == 0 or y_std == 0:
                return {"success": False, "error": "标准差为0，无法计算相关系数"}
            
            corr = cov / (x_std * y_std)
            return {"success": True, "result": float(corr)}
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    # ==================== 统计推断工具 ====================
    
    def confidence_interval_mean(self, data: List[float], confidence: float = 0.95, 
                                sigma_known: Optional[float] = None) -> dict:
        """计算均值的置信区间
        参数：
            data: 样本数据
            confidence: 置信水平（默认0.95）
            sigma_known: 已知总体标准差（如果为None，则使用t分布）
        """
        try:
            if not data:
                return {"success": False, "error": "数据不能为空"}
            if not 0 < confidence < 1:
                return {"success": False, "error": f"置信水平必须在0和1之间: {confidence}"}
            
            n = len(data)
            mean = sum(data) / n
            
            if sigma_known is not None:
                # 总体方差已知，使用z分布
                from scipy.stats import norm
                z_alpha = norm.ppf((1 + confidence) / 2)
                margin = z_alpha * sigma_known / math.sqrt(n)
                lower = mean - margin
                upper = mean + margin
                method = "z-distribution (σ known)"
            else:
                # 总体方差未知，使用t分布
                from scipy.stats import t
                # 计算样本标准差
                s = math.sqrt(sum((x - mean) ** 2 for x in data) / (n - 1))
                t_alpha = t.ppf((1 + confidence) / 2, df=n-1)
                margin = t_alpha * s / math.sqrt(n)
                lower = mean - margin
                upper = mean + margin
                method = "t-distribution (σ unknown)"
            
            return {
                "success": True,
                "result": {
                    "mean": float(mean),
                    "lower_bound": float(lower),
                    "upper_bound": float(upper),
                    "margin_of_error": float(margin),
                    "confidence_level": confidence,
                    "sample_size": n,
                    "method": method
                }
            }
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def hypothesis_test_mean(self, data: List[float], mu0: float, 
                           alternative: str = "two-sided", alpha: float = 0.05,
                           sigma_known: Optional[float] = None) -> dict:
        """均值假设检验
        参数：
            data: 样本数据
            mu0: 原假设的均值
            alternative: 备择假设类型，可选："two-sided", "greater", "less"
            alpha: 显著性水平
            sigma_known: 已知总体标准差
        """
        try:
            if not data:
                return {"success": False, "error": "数据不能为空"}
            if alternative not in ["two-sided", "greater", "less"]:
                return {"success": False, "error": f"alternative必须是'two-sided', 'greater'或'less': {alternative}"}
            if not 0 < alpha < 1:
                return {"success": False, "error": f"α必须在0和1之间: {alpha}"}
            
            n = len(data)
            mean = sum(data) / n
            
            if sigma_known is not None:
                # z检验
                from scipy.stats import norm
                z_stat = (mean - mu0) / (sigma_known / math.sqrt(n))
                
                if alternative == "two-sided":
                    p_value = 2 * (1 - norm.cdf(abs(z_stat)))
                elif alternative == "greater":
                    p_value = 1 - norm.cdf(z_stat)
                else:  # "less"
                    p_value = norm.cdf(z_stat)
                
                test_type = "z-test (σ known)"
                df = None
            else:
                # t检验
                from scipy.stats import t
                # 计算样本标准差
                s = math.sqrt(sum((x - mean) ** 2 for x in data) / (n - 1))
                t_stat = (mean - mu0) / (s / math.sqrt(n))
                df = n - 1
                
                if alternative == "two-sided":
                    p_value = 2 * (1 - t.cdf(abs(t_stat), df))
                elif alternative == "greater":
                    p_value = 1 - t.cdf(t_stat, df)
                else:  # "less"
                    p_value = t.cdf(t_stat, df)
                
                test_type = f"t-test (σ unknown, df={df})"
                z_stat = t_stat  # 为了一致性
            
            reject_null = p_value < alpha
            
            return {
                "success": True,
                "result": {
                    "test_statistic": float(z_stat),
                    "p_value": float(p_value),
                    "reject_null": reject_null,
                    "sample_mean": float(mean),
                    "hypothesized_mean": float(mu0),
                    "alpha": alpha,
                    "alternative": alternative,
                    "test_type": test_type,
                    "degrees_of_freedom": df
                }
            }
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def chi2_independence_test(self, observed: List[List[float]]) -> dict:
        """卡方独立性检验"""
        try:
            from scipy.stats import chi2_contingency
            observed_array = np.array(observed)
            
            chi2, p, dof, expected = chi2_contingency(observed_array)
            
            return {
                "success": True,
                "result": {
                    "chi2_statistic": float(chi2),
                    "p_value": float(p),
                    "degrees_of_freedom": int(dof),
                    "expected_frequencies": expected.tolist()
                }
            }
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    # ==================== 特殊函数和数值计算 ====================
    
    def gamma_function(self, x: float) -> dict:
        """伽马函数 Γ(x)"""
        try:
            from scipy.special import gamma
            result = gamma(x)
            return {"success": True, "result": float(result)}
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def beta_function(self, a: float, b: float) -> dict:
        """贝塔函数 B(a, b) = Γ(a)Γ(b) / Γ(a+b)"""
        try:
            from scipy.special import beta
            result = beta(a, b)
            return {"success": True, "result": float(result)}
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def erf(self, x: float) -> dict:
        """误差函数 erf(x) = 2/√π ∫₀ˣ e^{-t²} dt"""
        try:
            result = math.erf(x)
            return {"success": True, "result": float(result)}
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    # ==================== 分布拟合 ====================
    
    def fit_distribution(self, data: List[float], dist_type: str = "normal") -> dict:
        """拟合分布参数"""
        try:
            data_array = np.array(data)
            
            if dist_type == "normal":
                mu = np.mean(data_array)
                sigma = np.std(data_array, ddof=1)  # 样本标准差
                return {
                    "success": True,
                    "result": {
                        "distribution": "normal",
                        "parameters": {"mu": float(mu), "sigma": float(sigma)},
                        "log_likelihood": None  # 可扩展
                    }
                }
            elif dist_type == "exponential":
                lam = 1 / np.mean(data_array)
                return {
                    "success": True,
                    "result": {
                        "distribution": "exponential",
                        "parameters": {"lambda": float(lam)},
                        "log_likelihood": None
                    }
                }
            elif dist_type == "uniform":
                a = np.min(data_array)
                b = np.max(data_array)
                return {
                    "success": True,
                    "result": {
                        "distribution": "uniform",
                        "parameters": {"a": float(a), "b": float(b)},
                        "log_likelihood": None
                    }
                }
            else:
                return {"success": False, "error": f"不支持的分布类型: {dist_type}"}
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    # ==================== 考研数学特定工具 ====================
    
    def total_probability(self, conditional_probs: List[float], 
                         prior_probs: List[float]) -> dict:
        """全概率公式 P(A) = Σ P(A|B_i)P(B_i)"""
        try:
            if len(conditional_probs) != len(prior_probs):
                return {"success": False, "error": "条件概率和先验概率长度不匹配"}
            
            total = sum(c * p for c, p in zip(conditional_probs, prior_probs))
            return {"success": True, "result": float(total)}
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def bayes_theorem(self, conditional_prob: float, prior_prob: float, 
                     total_prob: float) -> dict:
        """贝叶斯公式 P(B_i|A) = P(A|B_i)P(B_i) / P(A)"""
        try:
            posterior = (conditional_prob * prior_prob) / total_prob
            return {"success": True, "result": float(posterior)}
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def convolution_sum(self, x_probs: List[float], y_probs: List[float]) -> dict:
        """离散卷积（卷积和），用于独立随机变量和的分布"""
        try:
            n = len(x_probs)
            m = len(y_probs)
            result_length = n + m - 1
            convolution = [0.0] * result_length
            
            for i in range(n):
                for j in range(m):
                    convolution[i + j] += x_probs[i] * y_probs[j]
            
            return {"success": True, "result": convolution}
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def moment_generating_function(self, moments: List[float], t: float = 1.0) -> dict:
        """矩母函数近似 M_X(t) ≈ Σ (t^k / k!) * E[X^k]"""
        try:
            mgf = 1.0
            for k, moment in enumerate(moments, 1):
                mgf += (t ** k) * moment / math.factorial(k)
            return {"success": True, "result": float(mgf)}
        except Exception as e:
            return {"success": False, "error": str(e)}