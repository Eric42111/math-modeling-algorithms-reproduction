"""
最小二乘：从闭式解到模型诊断

对应知识点：
  1. 拟合 vs 插值        -> 见 fit_vs_interp_demo()
  2. 为什么用平方        -> 见 why_square()
  3. 一元闭式解 + 必过(x̄,ȳ) -> 见 solve_ols(), check_passes_through_center()
  4. R² 与残差诊断       -> 见 r_squared(), residual_table()

运行方式：
  py least_squares_from_scratch.py
"""

# ============================================================
# 样本数据：8 个观测点（与 least_squares_demo.py 同一组）
# ============================================================
X = [1, 2, 3, 4, 5, 6, 7, 8]
Y = [2.1, 3.9, 6.2, 7.8, 10.1, 12.3, 13.9, 16.2]


def mean(vals):
    """算术平均值"""
    return sum(vals) / len(vals)


# ============================================================
# 知识点 3：一元线性回归的闭式解
# ============================================================
def solve_ols(x, y):
    """
    一元最小二乘闭式解。

    目标函数：S(a,b) = Σ(yᵢ - a·xᵢ - b)²
    对 a、b 分别求偏导令其为 0，解得：

        a = Σ(xᵢ-x̄)(yᵢ-ȳ) / Σ(xᵢ-x̄)²
        b = ȳ - a·x̄

    返回 (a, b)
    """
    x_bar = mean(x)
    y_bar = mean(y)

    # 分子：Σ(xᵢ-x̄)(yᵢ-ȳ)   —— x 与 y 的协变部分
    numerator = sum((xi - x_bar) * (yi - y_bar) for xi, yi in zip(x, y))
    # 分母：Σ(xᵢ-x̄)²        —— x 的离差平方和
    denominator = sum((xi - x_bar) ** 2 for xi in x)

    if denominator == 0:
        raise ValueError("所有 x 相同，分母为 0，无法拟合直线")

    a = numerator / denominator
    b = y_bar - a * x_bar
    return a, b


def check_passes_through_center(a, b, x, y):
    """
    知识点 3 的重要推论：拟合直线必定经过点 (x̄, ȳ)。

    这是检验计算是否正确的快捷方法——
    算完之后代进去验一遍，不成立就说明算错了。
    """
    x_bar = mean(x)
    y_bar = mean(y)
    y_at_center = a * x_bar + b
    diff = abs(y_at_center - y_bar)
    ok = diff < 1e-9
    return ok, x_bar, y_bar, y_at_center, diff


# ============================================================
# 知识点 4：R² 拟合优度
# ============================================================
def r_squared(x, y, a, b):
    """
    R² = 1 - SS_res / SS_tot

    SS_tot = Σ(yᵢ - ȳ)²     数据本身的总变异（只用均值猜时的误差）
    SS_res = Σ(yᵢ - ŷᵢ)²    模型未能解释的残差变异

    含义：模型解释了 y 的总变异的百分之多少。
    注意：R² 高 ≠ 模型好，也不代表预测新数据准。
    """
    y_bar = mean(y)
    ss_tot = sum((yi - y_bar) ** 2 for yi in y)
    ss_res = sum((yi - (a * xi + b)) ** 2 for xi, yi in zip(x, y))
    return 1 - ss_res / ss_tot, ss_res, ss_tot


def residual_table(x, y, a, b):
    """
    知识点 6（进阶）：残差分析。

    残差 eᵢ = yᵢ - ŷᵢ。
    好模型的残差应该随机正负交替、无系统模式。
    若残差呈 U 型 -> 漏了非线性项；呈漏斗型 -> 异方差。
    """
    rows = []
    for xi, yi in zip(x, y):
        y_hat = a * xi + b
        rows.append((xi, yi, y_hat, yi - y_hat))
    return rows


# ============================================================
# 知识点 2：为什么用平方，而不是别的？
# ============================================================
def why_square(x, y, a, b):
    """
    用三组数字说明"为什么目标函数必须是残差平方和"。
    """
    residuals = [yi - (a * xi + b) for xi, yi in zip(x, y)]

    sum_e = sum(residuals)                      # 直接求和
    sum_abs = sum(abs(e) for e in residuals)    # 绝对值和
    sum_sq = sum(e ** 2 for e in residuals)     # 平方和

    return sum_e, sum_abs, sum_sq


def fit_vs_interp_note():
    """
    知识点 1：拟合 vs 插值（文字说明，无计算）

    插值：曲线必须穿过每一个数据点 -> 会把噪声一起穿进去（过拟合）
    拟合：只要求整体趋势最接近，允许每个点有偏差

    实测数据几乎都带噪声，所以数模 99% 的场景用拟合。
    插值还有 Runge 现象：多项式次数越高，区间端点振荡越剧烈。
    """
    return (
        "插值 = 过点（噪声也穿进去，易过拟合）\n"
        "拟合 = 看整体趋势（允许每点有偏差）\n"
        "实测数据含噪声 -> 用拟合，不用插值"
    )


# ============================================================
# 主程序
# ============================================================
def main():
    print("=" * 58)
    print("最小二乘：从闭式解到模型诊断（零依赖纯 Python）")
    print("=" * 58)

    # --- 知识点 1 ---
    print("\n【知识点 1】拟合 vs 插值")
    print("-" * 58)
    print(fit_vs_interp_note())

    # --- 知识点 3 ---
    print("\n【知识点 3】一元闭式解")
    print("-" * 58)
    a, b = solve_ols(X, Y)
    x_bar, y_bar = mean(X), mean(Y)
    num = sum((xi - x_bar) * (yi - y_bar) for xi, yi in zip(X, Y))
    den = sum((xi - x_bar) ** 2 for xi in X)

    print(f"x̄ = {x_bar:.4f},  ȳ = {y_bar:.4f}")
    print(f"分子 Σ(xᵢ-x̄)(yᵢ-ȳ) = {num:.4f}")
    print(f"分母 Σ(xᵢ-x̄)²     = {den:.4f}")
    sign = "-" if b < 0 else "+"
    print(f"\n拟合直线: y = {a:.4f}x {sign} {abs(b):.4f}")

    # 验证必过 (x̄, ȳ)
    ok, xb, yb, y_at_center, diff = check_passes_through_center(a, b, X, Y)
    print(f"\n[校验] 直线是否经过 (x̄, ȳ) = ({xb:.4f}, {yb:.4f})？")
    print(f"       代入 x = {xb:.4f} 得 ŷ = {y_at_center:.4f}")
    print(f"       偏差 = {diff:.2e}  ->  {'✅ 通过' if ok else '❌ 不成立，算错了'}")

    # --- 知识点 2 ---
    print("\n【知识点 2】为什么目标函数用「平方和」")
    print("-" * 58)
    sum_e, sum_abs, sum_sq = why_square(X, Y, a, b)
    print(f"Σeᵢ    （直接求和） = {sum_e:>10.4f}   <- 正负抵消，接近 0，无法作为目标")
    print(f"Σ|eᵢ|  （绝对值和） = {sum_abs:>10.4f}   <- 可用，但 0 点不可导，只能迭代求解")
    print(f"Σeᵢ²   （平方和）   = {sum_sq:>10.4f}   <- 处处可导 + 对大误差惩罚更重")
    print("\n补充：最小的是竖直方向(y方向)残差，不是点到直线的垂直距离。")
    print("      后者叫正交回归/总最小二乘，解出来是另一条直线。")

    # --- 知识点 4 ---
    print("\n【知识点 4】R² 拟合优度")
    print("-" * 58)
    r2, ss_res, ss_tot = r_squared(X, Y, a, b)
    print(f"SS_res（未解释的变异） = {ss_res:.4f}")
    print(f"SS_tot（总变异）       = {ss_tot:.4f}")
    print(f"R² = 1 - SS_res/SS_tot = {r2:.4f}")
    print(f"\n含义：这条直线解释了 y 变异的 {r2*100:.2f}%，"
          f"还剩 {(1-r2)*100:.2f}% 未解释。")
    print("⚠️  R² 高 ≠ 模型好，也 ≠ 预测新数据准。")

    # --- 残差表 ---
    print("\n【残差诊断】")
    print("-" * 58)
    rows = residual_table(X, Y, a, b)
    print(f"{'x':>4}  {'实际y':>8}  {'拟合ŷ':>8}  {'残差e':>9}")
    print("-" * 58)
    for xi, yi, y_hat, e in rows:
        print(f"{xi:>4}  {yi:>8.2f}  {y_hat:>8.4f}  {e:>+9.4f}")
    print("-" * 58)
    signs = [1 if e > 0 else -1 for _, _, _, e in rows]
    flips = sum(1 for i in range(1, len(signs)) if signs[i] != signs[i-1])
    print(f"残差正负交替 {flips} 次 / {len(rows)-1} 次机会"
          f"  -> {'✅ 随机无模式，线性假设合理' if flips >= 4 else '⚠️  可能存在系统模式'}")

    # --- 外推 ---
    print("\n【外推预测】")
    print("-" * 58)
    for x_new in (9.0, 10.0):
        print(f"x = {x_new:>4.1f}  ->  ŷ = {a*x_new + b:.4f}")
    print("⚠️  外推超出数据范围，R² 再高也不保证准确。")

    print("\n" + "=" * 58)


if __name__ == "__main__":
    main()
