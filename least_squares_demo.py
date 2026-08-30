"""
最小二乘拟合 —— 本仓库第一个算法复现
数模中最基础的拟合方法：给定数据点，求 y = ax + b 的最优参数
"""
import numpy as np

# 1. 样本数据
x = np.array([1, 2, 3, 4, 5, 6, 7, 8])
y = np.array([2.1, 3.9, 6.2, 7.8, 10.1, 12.3, 13.9, 16.2])

# 2. 构造设计矩阵 A = [x, 1]
A = np.vstack([x, np.ones(len(x))]).T

# 3. 最小二乘求解（lstsq 内部用 SVD，数值稳定）
coef, residuals, rank, s = np.linalg.lstsq(A, y, rcond=None)
a, b = coef

print(f"拟合直线: y = {a:.4f}x + {b:.4f}")

# 4. 拟合优度 R²
y_pred = a * x + b
ss_res = np.sum((y - y_pred) ** 2)
ss_tot = np.sum((y - np.mean(y)) ** 2)
r2 = 1 - ss_res / ss_tot
print(f"R2 = {r2:.4f}   (越接近 1 拟合越好)")

# 5. 外推预测
x_new = 9.0
print(f"当 x = {x_new} 时，预测 y = {a * x_new + b:.4f}")
