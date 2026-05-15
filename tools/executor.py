import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)  # tools 的父目录
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

try:
    # 尝试从 tools 子目录导入
    from tools.calculus_tools import CalculusTools, LinearAlgebraTools, ProbabilityTools
except ImportError:
    # 如果失败，尝试从当前目录导入（假设文件在同一目录）
    try:
        from calculus_tools import CalculusTools, LinearAlgebraTools, ProbabilityTools
    except ImportError as e:
        print(f"导入错误: {e}")
        print("请确保 calculus_tools.py 文件存在")
        sys.exit(1)

# 初始化工具实例
calculus_tools = CalculusTools()
linear_algebra_tools = LinearAlgebraTools()
probability_tools = ProbabilityTools()

# 工具映射表
TOOLS = {
    # 微积分工具
    "diff": calculus_tools.diff,
    "integrate": calculus_tools.integrate,
    "limit": calculus_tools.limit,
    "solve": calculus_tools.solve,
    "simplify": calculus_tools.simplify,
    "expand": calculus_tools.expand,
    "factor": calculus_tools.factor,
    
    # 线性代数工具
    "determinant": linear_algebra_tools.determinant,
    "inverse": linear_algebra_tools.inverse,
    "rank": linear_algebra_tools.rank,
    "solve_linear_system": linear_algebra_tools.solve_linear_system,
    "eigenvalues": linear_algebra_tools.eigenvalues,
    "matrix_multiply": linear_algebra_tools.matrix_multiply,
    "gram_schmidt": linear_algebra_tools.gram_schmidt,
    
    # 概率论与数理统计工具
    "combination": probability_tools.combination,
    "permutation": probability_tools.permutation,
    "factorial": probability_tools.factorial,
    "binomial_pmf": probability_tools.binomial_pmf,
    "binomial_cdf": probability_tools.binomial_cdf,
    "poisson_pmf": probability_tools.poisson_pmf,
    "poisson_cdf": probability_tools.poisson_cdf,
    "geometric_pmf": probability_tools.geometric_pmf,
    "normal_pdf": probability_tools.normal_pdf,
    "normal_cdf": probability_tools.normal_cdf,
    "normal_quantile": probability_tools.normal_quantile,
    "exponential_pdf": probability_tools.exponential_pdf,
    "exponential_cdf": probability_tools.exponential_cdf,
    "uniform_pdf": probability_tools.uniform_pdf,
    "uniform_cdf": probability_tools.uniform_cdf,
    "expectation": probability_tools.expectation,
    "variance": probability_tools.variance,
    "covariance": probability_tools.covariance,
    "correlation": probability_tools.correlation,
    "confidence_interval_mean": probability_tools.confidence_interval_mean,
    "hypothesis_test_mean": probability_tools.hypothesis_test_mean,
    "chi2_independence_test": probability_tools.chi2_independence_test,
    "gamma_function": probability_tools.gamma_function,
    "beta_function": probability_tools.beta_function,
    "erf": probability_tools.erf,
    "fit_distribution": probability_tools.fit_distribution,
    "total_probability": probability_tools.total_probability,
    "bayes_theorem": probability_tools.bayes_theorem,
    "convolution_sum": probability_tools.convolution_sum,
    "moment_generating_function": probability_tools.moment_generating_function
}


def execute_tool(tool_name, args):
    """
    执行数学工具函数
    
    参数:
        tool_name: 工具名称
        args: 参数字典
        
    返回:
        工具执行结果，格式为 {"success": bool, "result": any, "error": str}
    """
    if tool_name is None:
        return {"success": False, "error": "未指定工具名称"}
    
    if tool_name not in TOOLS:
        return {"success": False, "error": f"未知工具: {tool_name}"}
    
    try:
        # 调用对应的工具函数
        result = TOOLS[tool_name](**args)
        return result
    except TypeError as e:
        # 参数类型错误
        return {"success": False, "error": f"参数错误: {str(e)}"}
    except Exception as e:
        # 其他异常
        return {"success": False, "error": f"工具执行异常: {str(e)}"}


def get_available_tools():
    """获取所有可用工具的信息"""
    return {
        "calculus": [
            "diff(expr, var='x') - 求导",
            "integrate(expr, var='x') - 积分",
            "limit(expr, var='x', point=0) - 求极限",
            "solve(expr, var='x') - 解方程",
            "simplify(expr) - 化简",
            "expand(expr) - 展开",
            "factor(expr) - 因式分解"
        ],
        "linear_algebra": [
            "determinant(matrix_data) - 行列式",
            "inverse(matrix_data) - 逆矩阵",
            "rank(matrix_data) - 矩阵的秩",
            "solve_linear_system(A_data, b_data) - 解线性方程组",
            "eigenvalues(matrix_data) - 特征值",
            "matrix_multiply(A_data, B_data) - 矩阵乘法",
            "gram_schmidt(vectors_data, normalize=True) - 施密特正交化"
        ],
        "probability": [
            "combination(n, k) - 组合数C(n,k)",
            "permutation(n, k) - 排列数A(n,k)",
            "factorial(n) - 阶乘n!",
            "binomial_pmf(k, n, p) - 二项分布概率P(X=k)",
            "binomial_cdf(k, n, p) - 二项分布累积P(X≤k)",
            "normal_pdf(x, mu=0, sigma=1) - 正态分布密度函数",
            "normal_cdf(x, mu=0, sigma=1) - 正态分布函数Φ(x)",
            "expectation(values, probs=None) - 期望/均值",
            "variance(values, probs=None) - 方差",
            "correlation(x_values, y_values) - 相关系数",
            "confidence_interval_mean(data, confidence=0.95, sigma_known=None) - 均值置信区间",
            "hypothesis_test_mean(data, mu0, alternative='two-sided', alpha=0.05, sigma_known=None) - 均值假设检验"
        ]
    }
