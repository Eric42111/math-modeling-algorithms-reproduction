"""
ARIMA 第 1 步：差分平稳化 + AR(p) 建模 + 外推预测（零依赖，仅需标准库）

本文件对应复现清单 #4「时间序列 ARIMA」的第一阶段。
为什么先做 AR：ARIMA(p,d,q) = d 阶差分 + AR(p) + MA(q)。
其中「d 阶差分」和「AR(p)」都能用最小二乘闭式解搞定，不需要迭代优化（这正是
本仓库第 1 个算法已实现的东西）；而 MA(q) 的参数估计必须用极大似然或迭代法，
留到下一阶段。先做出能跑通、能预测、能评估的最小闭环。

运行：py arima_forecast.py
"""

import math

# ---------------------------------------------------------------- 数据
# 示例：某商品 24 个月的月销量（有明显上升趋势 -> 非平稳，需差分）
SERIES = [120, 135, 148, 160, 175, 190, 205, 220, 240, 262,
          285, 310, 330, 350, 372, 395, 415, 440, 465, 490,
          520, 550, 580, 610]
TRAIN_N = 20          # 前 20 个点拟合，后 4 个点当测试集
MAX_P = 4             # AR 阶数搜索上限


# ---------------------------------------------------------------- 工具函数
def diff(series, d=1):
    """d 阶差分：把带趋势的非平稳序列变成平稳序列。长度减少 d。"""
    out = list(series)
    for _ in range(d):
        out = [out[i] - out[i - 1] for i in range(1, len(out))]
    return out


def inverse_diff(last_values, diff_preds, d=1):
    """差分的逆运算：把差分域的预测值还原回原始尺度。

    last_values: 原始序列末尾的 d 个真实值（作为还原的"起点"）
    diff_preds : 差分域上的预测值
    """
    # 先从 d 阶差分还原到 1 阶差分
    level = list(diff_preds)
    for _ in range(d - 1):
        level = prefix_sum_from(level, 0.0)
    # 再从 1 阶差分还原到原序列：y_t = y_{t-1} + dy_t
    out, prev = [], last_values[-1]
    for v in level:
        prev = prev + v
        out.append(prev)
    return out


def prefix_sum_from(values, start):
    out, acc = [], start
    for v in values:
        acc += v
        out.append(acc)
    return out


def mean(xs):
    return sum(xs) / len(xs)


def acf(series, lags):
    """自相关函数：rho_k = Cov(y_t, y_{t-k}) / Var(y_t)。用来看"隔几步还相关"。"""
    n = len(series)
    m = mean(series)
    denom = sum((x - m) ** 2 for x in series)
    return [sum((series[t] - m) * (series[t - k] - m) for t in range(k, n)) / denom
            for k in range(lags + 1)]


def pacf(series, lags):
    """偏自相关：剔除中间滞后项影响后，y_t 与 y_{t-k} 的"纯净"相关。
    用 Durbin-Levinson 递推求解，等价于对 AR(k) 拟合取最后一个系数。"""
    acf_vals = acf(series, lags)
    pacs, prev_phi = [], []
    for k in range(1, lags + 1):
        num = acf_vals[k] - sum(prev_phi[j - 1] * acf_vals[k - j] for j in range(1, k))
        den = 1 - sum(prev_phi[j - 1] * acf_vals[j] for j in range(1, k))
        phi_kk = num / den
        phi = list(prev_phi)
        for j in range(1, k):
            phi[j - 1] = prev_phi[j - 1] - phi_kk * prev_phi[k - j - 1]
        phi.append(phi_kk)
        pacs.append(phi_kk)
        prev_phi = phi
    return pacs


def solve_linear(A, b):
    """高斯消元解 A x = b（零依赖实现，用于求最小二乘的正规方程）。"""
    n = len(A)
    M = [row[:] + [b[i]] for i, row in enumerate(A)]
    for col in range(n):
        piv = max(range(col, n), key=lambda r: abs(M[r][col]))
        M[col], M[piv] = M[piv], M[col]
        if abs(M[col][col]) < 1e-12:
            raise ValueError("矩阵奇异，无法求解")
        for r in range(n):
            if r == col:
                continue
            f = M[r][col] / M[col][col]
            for c in range(col, n + 1):
                M[r][c] -= f * M[col][c]
    return [M[i][n] / M[i][i] for i in range(n)]


# ---------------------------------------------------------------- AR(p) 建模
def build_design(series, p):
    """构造最小二乘的设计矩阵：用前 p 个时刻预测当前时刻。"""
    X, y = [], []
    for t in range(p, len(series)):
        X.append([series[t - i - 1] for i in range(p)])
        y.append(series[t])
    return X, y


def fit_ar(series, p):
    """AR(p) 参数估计：对正规方程 (X'X) beta = X'y 求闭式解。"""
    X, y = build_design(series, p)
    k = p
    XtX = [[sum(X[r][i] * X[r][j] for r in range(len(X))) for j in range(k)] for i in range(k)]
    Xty = [sum(X[r][i] * y[r] for r in range(len(X))) for i in range(k)]
    beta = solve_linear(XtX, Xty)
    resid = [y[r] - sum(beta[i] * X[r][i] for i in range(k)) for r in range(len(X))]
    rss = sum(e ** 2 for e in resid)
    n = len(y)
    aic = n * math.log(rss / n) + 2 * k        # AIC：拟合好且参数少 -> 值小
    return beta, rss, aic


def forecast_ar(series, beta, steps):
    """多步外推：每预测一步，就把预测值当作"真实值"喂回去继续算。"""
    hist, preds = list(series), []
    for _ in range(steps):
        nxt = sum(beta[i] * hist[-1 - i] for i in range(len(beta)))
        preds.append(nxt)
        hist.append(nxt)
    return preds


# ---------------------------------------------------------------- 评估
def metrics(actual, pred):
    n = len(actual)
    mse = sum((actual[i] - pred[i]) ** 2 for i in range(n)) / n
    mae = sum(abs(actual[i] - pred[i]) for i in range(n)) / n
    mape = sum(abs((actual[i] - pred[i]) / actual[i]) for i in range(n)) / n * 100
    return math.sqrt(mse), mae, mape


def bar(v):
    """把相关系数画成 ASCII 条形图，肉眼判断截尾/拖尾。"""
    n = int(abs(v) * 30)
    return ("+" if v >= 0 else "-") + "#" * n


# ---------------------------------------------------------------- 主流程
def main():
    print("=" * 62)
    print("ARIMA 第 1 步：差分平稳化 + AR(p) 建模 + 外推预测")
    print("=" * 62)

    train = SERIES[:TRAIN_N]
    test = SERIES[TRAIN_N:]

    # 1) 差分：消掉趋势，让序列"围绕一个固定水平波动"
    d = 1
    d_train = diff(train, d)
    m0, m1 = mean(train), mean(d_train)
    sd0 = math.sqrt(sum((x - m0) ** 2 for x in train) / len(train))
    sd1 = math.sqrt(sum((x - m1) ** 2 for x in d_train) / len(d_train))
    print(f"\n[1] {d} 阶差分")
    print(f"    原序列   均值 {m0:8.2f}  标准差 {sd0:8.2f}")
    print(f"    差分后   均值 {m1:8.2f}  标准差 {sd1:8.2f}   <- 均值趋于 0，趋势被消掉")

    # 2) ACF / PACF：判断相关性结构
    print(f"\n[2] 自相关 ACF / 偏自相关 PACF（虚线 |acf|>{1.96/math.sqrt(len(d_train)):.2f} 视为显著）")
    a_vals, p_vals = acf(d_train, 6), pacf(d_train, 6)
    print(f"    {'lag':>4} {'ACF':>8}  {'图':<32} {'PACF':>8}")
    for k in range(1, 7):
        print(f"    {k:>4} {a_vals[k]:>8.3f}  {bar(a_vals[k]):<32} {p_vals[k-1]:>8.3f}")

    # 3) AIC 定阶：在 p=1..MAX_P 里挑 AIC 最小的
    print(f"\n[3] AR 阶数选择（AIC 越小越好）")
    best_p, best_aic, best_beta = None, float("inf"), None
    for p in range(1, MAX_P + 1):
        _, _, aic = fit_ar(d_train, p)
        flag = ""
        if aic < best_aic:
            best_p, best_aic, best_beta = p, aic, fit_ar(d_train, p)[0]
            flag = "  <- 当前最优"
        print(f"    p = {p}   AIC = {aic:8.3f}{flag}")
    print(f"    选定 p = {best_p}")
    print(f"    AR({best_p}) 系数：{[round(c, 4) for c in best_beta]}")

    # 4) 预测：在差分域外推，再逆差分还原到原始尺度
    steps = len(test)
    diff_preds = forecast_ar(d_train, best_beta, steps)
    preds = inverse_diff(train, diff_preds, d)

    print(f"\n[4] 外推 {steps} 步（差分域 -> 还原回原尺度）")
    print(f"    {'月份':>6} {'真实值':>10} {'预测值':>10} {'误差':>9}")
    for i, (a, p) in enumerate(zip(test, preds)):
        print(f"    {TRAIN_N + i + 1:>6} {a:>10.1f} {p:>10.1f} {p - a:>9.1f}")

    rmse, mae, mape = metrics(test, preds)
    print(f"\n[5] 测试集评估")
    print(f"    RMSE = {rmse:.2f}   MAE = {mae:.2f}   MAPE = {mape:.2f}%")

    print("\n说明：本阶段只做 AR(p)。ARIMA 的完整形态还要加 MA(q) 项，")
    print("      MA 的残差项不可直接观测，需用极大似然或迭代估计——下一阶段补上。")
    print("=" * 62)


if __name__ == "__main__":
    main()
