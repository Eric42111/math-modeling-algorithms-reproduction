"""
灰色预测 GM(1,1) —— 本仓库第 2 个算法复现
零依赖纯 Python 实现（不需要 numpy / pandas）

为什么是 GM(1,1)？
    它的参数估计用的就是「最小二乘」——
    和第 1 个算法 least_squares_from_scratch.py 是同一套数学，
    只是把自变量换成了「累加生成序列的紧邻均值」。

适用场景：
    ✅ 小样本（n >= 4 即可）、贫信息、单调趋势明显的序列预测
    ❌ 数据波动剧烈 / 非单调 / 有明显的周期或季节性 —— 不要用

运行方式：
    py gm11_gray_prediction.py
"""

import math


# ============================================================
# 第一步：累加生成（1-AGO, Accumulated Generating Operation）
# ============================================================
def ago(x0):
    """
    一次累加生成：x⁽¹⁾(k) = Σ(i=1..k) x⁽⁰⁾(i)

    原始序列往往杂乱无章，累加后会显露出指数增长的趋势。
    这是灰色系统理论的核心技巧——把"灰色"（不确定）变"白"（有规律）。
    """
    result = []
    s = 0.0
    for v in x0:
        s += v
        result.append(s)
    return result


# ============================================================
# 第二步：紧邻均值生成
# ============================================================
def adjacent_mean(x1):
    """
    紧邻均值生成：z⁽¹⁾(k) = 0.5·x⁽¹⁾(k) + 0.5·x⁽¹⁾(k-1)，k = 2..n

    用前后两点的平均代替 x⁽¹⁾(k)，是为了把微分方程离散化时
    减小误差（相当于用梯形面积代替矩形面积）。
    """
    return [0.5 * (x1[k] + x1[k - 1]) for k in range(1, len(x1))]


# ============================================================
# 第三步：最小二乘估计参数 a, b
# ============================================================
def estimate_params(x0, z1):
    """
    灰微分方程：x⁽⁰⁾(k) + a·z⁽¹⁾(k) = b

    写成矩阵形式 Y = B·[a, b]ᵀ：
        Y = [x⁽⁰⁾(2), x⁽⁰⁾(3), ..., x⁽⁰⁾(n)]ᵀ
        B = [[-z⁽¹⁾(2), 1],
             [-z⁽¹⁾(3), 1],
             ...
             [-z⁽¹⁾(n), 1]]

    正规方程：[a, b]ᵀ = (BᵀB)⁻¹·BᵀY

    ⚠️ 这里就是最小二乘——和 least_squares_from_scratch.py 的
       solve_ols() 完全同一套逻辑，只是从一元推广到了二元。
       本函数直接手解 2×2 线性方程组，避免引入矩阵库。
    """
    n = len(z1)
    Y = x0[1:]                      # x⁽⁰⁾(2..n)

    # 构造 BᵀB 和 BᵀY
    # B 的第 i 行是 [-z1[i], 1]
    sum_z2 = sum(z * z for z in z1)          # Σ z²
    sum_z = sum(z1)                          # Σ z
    m = n                                    # 行数（常数项 1 的个数）

    # BᵀB = [[Σz², -Σz], [-Σz, m]]   （注意第一列是 -z）
    b11 = sum_z2
    b12 = -sum_z
    b21 = -sum_z
    b22 = float(m)

    # BᵀY = [-Σ(z·y), Σy]
    t1 = -sum(z * y for z, y in zip(z1, Y))
    t2 = sum(Y)

    # 解 2×2 方程组（克拉默法则）
    det = b11 * b22 - b12 * b21
    if abs(det) < 1e-12:
        raise ValueError("矩阵奇异，无法估计参数（数据可能全相同）")

    a = (t1 * b22 - b12 * t2) / det
    b = (b11 * t2 - t1 * b21) / det
    return a, b


# ============================================================
# 第四步：时间响应 + 还原
# ============================================================
def predict(x0, a, b, steps):
    """
    白化方程 dx⁽¹⁾/dt + a·x⁽¹⁾ = b 的解：
        x⁽¹⁾(k+1) = (x⁽⁰⁾(1) - b/a)·e^(-a·k) + b/a

    得到的是累加序列的预测值，需要「累减还原」才能得到原始尺度：
        x̂⁽⁰⁾(k+1) = x̂⁽¹⁾(k+1) - x̂⁽¹⁾(k)
    """
    x1_hat = []                     # 累加序列的拟合值
    n = len(x0)
    for k in range(n + steps):
        val = (x0[0] - b / a) * math.exp(-a * k) + b / a
        x1_hat.append(val)

    # 累减还原：x̂⁽⁰⁾(k) = x̂⁽¹⁾(k) - x̂⁽¹⁾(k-1)
    x0_hat = [x1_hat[0]]            # 第 1 项无前项，直接取
    for k in range(1, len(x1_hat)):
        x0_hat.append(x1_hat[k] - x1_hat[k - 1])

    return x0_hat, x1_hat


# ============================================================
# 第五步：精度检验（后验差检验）
# ============================================================
def accuracy_check(x0, x0_hat):
    """
    后验差检验，两个指标：

    C = S₂ / S₁
        S₁ = 原始序列标准差，S₂ = 残差标准差
        C 越小越好（残差相对原始波动越小）

    P = P{ |e(k) - ē| < 0.6745·S₁ }
        小误差概率，P 越大越好

    精度等级参考：
        C < 0.35 且 P > 0.95  -> 一级（好）
        C < 0.50 且 P > 0.80  -> 二级（合格）
        C < 0.65 且 P > 0.70  -> 三级（勉强）
        C >= 0.65             -> 四级（不合格，换模型）
    """
    n = len(x0)
    fitted = x0_hat[:n]
    residuals = [x0[k] - fitted[k] for k in range(n)]

    mean_x = sum(x0) / n
    s1 = math.sqrt(sum((v - mean_x) ** 2 for v in x0) / (n - 1))

    e_bar = sum(residuals) / n
    s2 = math.sqrt(sum((e - e_bar) ** 2 for e in residuals) / (n - 1))

    C = s2 / s1 if s1 != 0 else float('inf')

    threshold = 0.6745 * s1
    count = sum(1 for e in residuals if abs(e - e_bar) < threshold)
    P = count / n

    if C < 0.35 and P > 0.95:
        level = "一级（好）"
    elif C < 0.50 and P > 0.80:
        level = "二级（合格）"
    elif C < 0.65 and P > 0.70:
        level = "三级（勉强）"
    else:
        level = "四级（不合格，建议换模型）"

    return residuals, C, P, level


# ============================================================
# 主程序
# ============================================================
def main():
    print("=" * 62)
    print("灰色预测 GM(1,1) · 零依赖纯 Python 实现")
    print("=" * 62)

    # 示例：某地区 2018-2024 年某种资源消耗量（单调递增，适合 GM(1,1)）
    years = [2018, 2019, 2020, 2021, 2022, 2023, 2024]
    x0 = [2.874, 3.278, 3.337, 3.390, 3.679, 3.850, 4.070]

    print("\n【原始数据】")
    print("-" * 62)
    for y, v in zip(years, x0):
        print(f"  {y} 年: {v:.3f}")

    # --- 第 1 步：累加生成 ---
    x1 = ago(x0)
    print("\n【第 1 步】一次累加生成 1-AGO")
    print("-" * 62)
    print("  " + "  ".join(f"{v:.3f}" for v in x1))
    print("  （杂乱的原始序列累加后，显露出指数增长趋势）")

    # --- 第 2 步：紧邻均值 ---
    z1 = adjacent_mean(x1)
    print("\n【第 2 步】紧邻均值生成")
    print("-" * 62)
    print("  " + "  ".join(f"{v:.3f}" for v in z1))

    # --- 第 3 步：最小二乘估参 ---
    a, b = estimate_params(x0, z1)
    print("\n【第 3 步】最小二乘估计参数（本算法的核心）")
    print("-" * 62)
    print(f"  发展系数 a = {a:.6f}   （-a 反映增长势头）")
    print(f"  灰作用量 b = {b:.6f}")
    print(f"  白化方程: dx⁽¹⁾/dt + ({a:.4f})·x⁽¹⁾ = {b:.4f}")
    print("\n  ⚠️ 这一步用的就是最小二乘——和第 1 个算法同一套数学，")
    print("     只是自变量从 x 换成了紧邻均值 z⁽¹⁾。")

    # --- 第 4 步：拟合与预测 ---
    steps = 3
    x0_hat, x1_hat = predict(x0, a, b, steps)
    print(f"\n【第 4 步】拟合 + 外推 {steps} 期")
    print("-" * 62)
    print(f"{'年份':>8}  {'实际值':>9}  {'拟合值':>9}  {'残差':>9}  {'相对误差':>9}")
    print("-" * 62)
    for k in range(len(x0)):
        err = x0[k] - x0_hat[k]
        rel = abs(err) / x0[k] * 100
        print(f"{years[k]:>8}  {x0[k]:>9.3f}  {x0_hat[k]:>9.3f}  {err:>+9.4f}  {rel:>8.2f}%")
    print("-" * 62)
    for i in range(steps):
        idx = len(x0) + i
        y = years[-1] + i + 1
        print(f"{y:>8}  {'—':>9}  {x0_hat[idx]:>9.3f}  {'（预测）':>9}  {'':>9}")

    # --- 第 5 步：精度检验 ---
    residuals, C, P, level = accuracy_check(x0, x0_hat)
    print("\n【第 5 步】后验差检验")
    print("-" * 62)
    print(f"  后验差比 C = {C:.4f}")
    print(f"  小误差概率 P = {P:.4f}")
    print(f"  精度等级：{level}")

    # --- 适用提醒 ---
    print("\n【什么时候不该用 GM(1,1)】")
    print("-" * 62)
    print("  ❌ 数据波动剧烈、非单调 —— 累加后仍无指数趋势")
    print("  ❌ 有明显周期性 / 季节性 —— 应用时间序列（ARIMA / 季节分解）")
    print("  ❌ 需要长期外推 —— GM(1,1) 短期预测准，越长越不可靠")
    print("  ✅ 小样本（n ≥ 4）、贫信息、单调趋势 —— 这才是它的主场")

    print("\n" + "=" * 62)


if __name__ == "__main__":
    main()
