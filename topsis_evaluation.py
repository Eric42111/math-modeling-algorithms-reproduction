# -*- coding: utf-8 -*-
"""
TOPSIS 综合评价法（Technique for Order Preference by Similarity to Ideal Solution）
—— 逼近理想解排序法 · 零依赖实现（仅标准库）

【核心思想】
    在多个方案、多个（往往互相冲突的）指标下，定义一个「虚拟最优方案」和
    「虚拟最劣方案」，然后给每个真实方案打分：
        既离最优方案尽可能近，又离最劣方案尽可能远。
    贴近度 C = D- / (D+ + D-)，C ∈ [0,1]，越大越好。

【为什么要这个模型】
    数模里大量问题是「选哪个最好」而不是「预测多少」：
      - 选哪个检测时点 / 选址 / 选供应商 / 选应急方案 / 评价城市宜居度
    这些问题的指标天生冲突（越早发现风险越低，但检测越容易失败），
    没法用单一指标排序，TOPSIS 就是处理这类权衡的标准工具。

【七步流程】
    1. 指标正向化    —— 统一转成「越大越好」
    2. 标准化        —— 向量归一化，消除量纲
    3. 加权          —— 乘以权重得到加权矩阵
    4. 定理想解      —— 每列取 max 为 V+，min 为 V-
    5. 算距离        —— 各方案到 V+、V- 的欧氏距离
    6. 算贴近度      —— C = D- / (D+ + D-)
    7. 排序 + 敏感性分析




"""

import math

# 第 1 步：指标正向化 —— 把所有指标统一成「极大型」（越大越好）

def positive_transform(values, kind, best=None, low=None, high=None, mode='range'):
    """
    将一列指标转换为极大型。

    参数
    ----
    values : list[float]   原始指标列
    kind   : str
        'max'      极大型（效益型）—— 越大越好，如达标率
        'min'      极小型（成本型）—— 越小越好，如失败率
        'mid'      中间型          —— 越接近某个值越好，如 pH 值
        'interval' 区间型          —— 落在某区间内最好，如体温
    best / low / high : 对应 kind 所需的参数
    mode   : 极小型的转换方式
        'range'      极差变换  x' = max(x) - x      （平移，保持间距）
        'reciprocal' 倒数变换  x' = 1 / x           （改变比例关系）

    返回
    ----
    list[float] 转换后的极大型指标列
    """
    if kind == 'max':
        return [float(v) for v in values]

    if kind == 'min':
        if mode == 'reciprocal':
            # 倒数法：要求所有值 > 0。它保持「倍数关系」，
            # 会把小数值区域的差异放大（例如 0.01 → 100）
            if min(values) <= 0:
                raise ValueError("倒数变换要求所有值 > 0，请改用 mode='range'")
            return [1.0 / v for v in values]
        else:
            # 极差法：x' = max - x。它只是「翻转 + 平移」，
            # 保留了原始数据的间距结构，但把零点平移了，
            # 导致后续向量归一化的结果依赖「最大值是谁」——这是它的隐含缺陷。
            m = max(values)
            return [m - v for v in values]

    if kind == 'mid':
        # 中间型：越接近 best 越好
        #     x' = 1 - |x - best| / max|x - best|
        if best is None:
            raise ValueError("kind='mid' 需要提供 best 参数")
        dev = max(abs(v - best) for v in values)
        if dev == 0:
            return [1.0] * len(values)
        return [1.0 - abs(v - best) / dev for v in values]

    if kind == 'interval':
        # 区间型：落在 [low, high] 内最好，越往外越差
        if low is None or high is None:
            raise ValueError("kind='interval' 需要提供 low / high 参数")
        M = max(low - min(values), max(values) - high)
        if M <= 0:
            return [1.0] * len(values)
        out = []
        for v in values:
            if v < low:
                out.append(1.0 - (low - v) / M)
            elif v > high:
                out.append(1.0 - (v - high) / M)
            else:
                out.append(1.0)
        return out

    raise ValueError(f"未知的指标类型: {kind}")



# 第 2 步：标准化 —— 向量归一化，消除量纲


def normalize(matrix):
    """
    向量归一化： z_ij = x_ij / sqrt( Σ_i x_ij² )

    每一列单独除以该列的范数。这样每列都变成「单位向量」，
    不同量纲（百分比 / 周数 / 无量纲指数）之间才可比。

    ⚠️ 常见错误：把「标准化」和「正向化」搞混。
       正向化改的是方向（越小越好 → 越大越好），
       标准化改的是尺度（消除量纲）。两者都要做，且**先正向化后标准化**。
       如果顺序反了，成本型指标会先被归一化再翻转，结果完全不同。
    """
    n_col = len(matrix[0])
    norm = []
    for j in range(n_col):
        col_sq_sum = sum(row[j] ** 2 for row in matrix)
        denom = math.sqrt(col_sq_sum)
        if denom == 0:
            denom = 1e-12  # 防止除零（整列全 0 的退化情形）
        norm.append(denom)
    return [[row[j] / norm[j] for j in range(n_col)] for row in matrix], norm


# ============================================================
# 第 3 步：加权
# ============================================================

def apply_weights(matrix, weights):
    """加权标准化矩阵： v_ij = w_j * z_ij"""
    if abs(sum(weights) - 1.0) > 1e-9:
        raise ValueError(f"权重之和必须为 1，当前为 {sum(weights):.6f}")
    n_col = len(matrix[0])
    if len(weights) != n_col:
        raise ValueError(f"权重个数 {len(weights)} 与指标个数 {n_col} 不匹配")
    return [[row[j] * weights[j] for j in range(n_col)] for row in matrix]


# ============================================================
# 第 4–6 步：理想解、距离、贴近度
# ============================================================

def ideal_solutions(matrix):
    """正理想解 V+（每列 max）、负理想解 V-（每列 min）"""
    n_col = len(matrix[0])
    v_pos = [max(row[j] for row in matrix) for j in range(n_col)]
    v_neg = [min(row[j] for row in matrix) for j in range(n_col)]
    return v_pos, v_neg


def euclid(a, b):
    """欧氏距离"""
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))


def topsis(raw_matrix, kinds, weights, **kw):
    """
    完整 TOPSIS 流程。

    返回
    ----
    dict: {
        'C'    : list[float]  贴近度（越大越好）
        'D_pos': list[float]  到正理想解的距离
        'D_neg': list[float]  到负理想解的距离
        'rank' : list[int]    方案下标，按 C 降序
        'weighted': 加权标准化矩阵
        'positive': 正向化后的矩阵
    }
    """
    n_row = len(raw_matrix)
    n_col = len(raw_matrix[0])

    # 1. 正向化（逐列）
    positive = []
    for j in range(n_col):
        col = [raw_matrix[i][j] for i in range(n_row)]
        positive.append(positive_transform(col, kinds[j], **kw))
    # 转置回「行为方案、列为指标」
    positive = [[positive[j][i] for j in range(n_col)] for i in range(n_row)]

    # 2. 标准化
    normalized, _ = normalize(positive)

    # 3. 加权
    weighted = apply_weights(normalized, weights)

    # 4. 理想解
    v_pos, v_neg = ideal_solutions(weighted)

    # 5. 距离
    d_pos = [euclid(weighted[i], v_pos) for i in range(n_row)]
    d_neg = [euclid(weighted[i], v_neg) for i in range(n_row)]

    # 6. 贴近度
    scores = []
    for i in range(n_row):
        s = d_pos[i] + d_neg[i]
        scores.append(d_neg[i] / s if s > 0 else 0.0)

    # 7. 排序（C 降序；并列时按 D+ 升序，即离最优更近者在前）
    rank = sorted(range(n_row), key=lambda i: (-scores[i], d_pos[i]))

    return {
        'C': scores, 'D_pos': d_pos, 'D_neg': d_neg, 'rank': rank,
        'weighted': weighted, 'positive': positive,
        'v_pos': v_pos, 'v_neg': v_neg,
    }


# ============================================================
# 敏感性分析：换权重看排名稳不稳
# ============================================================

def sensitivity(raw_matrix, kinds, weight_sets, labels, **kw):
    """
    对同一批数据跑多组权重，输出每组的第一名与完整排名。

    【为什么要做这个】
    TOPSIS 最大的软肋是**权重是主观给的**。如果换一组同样合理的权重，
    最优方案就变了，那你论文里的结论就是脆的，评委一问就倒。
    敏感性分析就是提前自证：我的结论对权重不敏感（或明确指出在哪种偏好下会变）。
    """
    results = []
    for w, lab in zip(weight_sets, labels):
        r = topsis(raw_matrix, kinds, w, **kw)
        results.append((lab, w, r['rank'], r['C']))
    return results


# ============================================================
# 示例：2025 国赛 C 题 —— 男胎孕妇最佳 NIPT 时点选择
# ============================================================

def demo_nipt_timing():
    """
    场景（对应 2025 CUMCM C 题 问题 2/3 的简化版）：
        为某 BMI 分组内的男胎孕妇，从若干候选孕周中选最佳 NIPT 检测时点。

    四个指標，天然两两冲突：
        ① Y 浓度达标率（%）      效益型：越大越好 —— 孕周越大越容易达标
        ② 检测失败率（%）        成本型：越小越好 —— 孕周越大越不容易失败
        ③ 早期发现收益（周）     效益型：越大越好 —— 发现越早，治疗窗口越长
        ④ 晚期发现风险指数       成本型：越小越好 —— 发现越晚，风险急剧上升

    冲突点：①② 要「晚」，③④ 要「早」。这正是 TOPSIS 要解的题。
    """
    print("=" * 72)
    print("  TOPSIS 示例 · 2025 国赛 C 题：男胎孕妇最佳 NIPT 时点选择")
    print("=" * 72)

    # --- 候选方案：6 个候选检测时点（孕周）---
    plans = ["12 周", "14 周", "16 周", "18 周", "20 周", "22 周"]

    # --- 原始指标矩阵：行=方案，列=指标 ---
    raw = [
        [62.0, 18.0, 15.0, 1.0],   # 12 周：达标率低、失败率高，但发现早、风险低
        [74.0, 12.0, 13.0, 1.4],   # 14 周
        [83.0,  8.0, 11.0, 2.2],   # 16 周
        [89.0,  5.0,  9.0, 3.6],   # 18 周
        [93.0,  3.5,  7.0, 5.5],   # 20 周
        [96.0,  2.5,  5.0, 8.0],   # 22 周：达标率最高，但发现最晚、风险极高
    ]

    indicators = ["Y浓度达标率(%)", "检测失败率(%)", "早期发现收益(周)", "晚期风险指数"]
    kinds = ['max', 'min', 'max', 'min']
    weights = [0.30, 0.20, 0.30, 0.20]   # 主观权重，和为 1

    # ---------- 打印原始数据 ----------
    print("【第 0 步】原始数据矩阵")
    print(f"  {'方案':<8}", end="")
    for name in indicators:
        print(f"{name:>16}", end="")
    print()
    for i, p in enumerate(plans):
        print(f"  {p:<10}", end="")
        for v in raw[i]:
            print(f"{v:>16.1f}", end="")
        print()

    # ---------- 第 1 步：正向化 ----------
    print("\n【第 1 步】指标正向化（全部转成「越大越好」）")
    print(f"  指标类型：{[k for k in kinds]}")
    print(f"  说明：成本型用极差变换 x' = max(x) - x")
    pos = []
    for j in range(4):
        col = [raw[i][j] for i in range(6)]
        pos.append(positive_transform(col, kinds[j]))
    print(f"  {'方案':<8}", end="")
    for name in indicators:
        print(f"{name[:14]:>16}", end="")
    print()
    for i, p in enumerate(plans):
        print(f"  {p:<10}", end="")
        for j in range(4):
            print(f"{pos[j][i]:>16.2f}", end="")
        print()
    print("  ↑ 注意第 2、4 列：原来越小越好，现在翻转成了越大越好")

    # ---------- 完整 TOPSIS ----------
    result = topsis(raw, kinds, weights)
    C, Dp, Dn, rank = result['C'], result['D_pos'], result['D_neg'], result['rank']

    print("\n【第 2-3 步】向量归一化 → 加权（权重：" +
          " / ".join(f"{w:.2f}" for w in weights) + "）")
    print(f"  {'方案':<8}{'加权后范数':>14}")
    for i, p in enumerate(plans):
        nv = math.sqrt(sum(x ** 2 for x in result['weighted'][i]))
        print(f"  {p:<10}{nv:>14.4f}")

    print("\n【第 4 步】理想解")
    print("  V+ (正理想) = [" + ", ".join(f"{v:.4f}" for v in result['v_pos']) + "]")
    print("  V- (负理想) = [" + ", ".join(f"{v:.4f}" for v in result['v_neg']) + "]")
    print("  ↑ 这是虚拟方案，现实中不一定存在——这正是 TOPSIS 的巧妙之处")

    # ---------- 第 5-6 步 ----------
    print("\n【第 5-6 步】距离与贴近度")
    print(f"  {'排名':<6}{'方案':<10}{'D+(离最优)':>14}{'D-(离最劣)':>14}{'贴近度 C':>12}")
    print("  " + "-" * 56)
    for pos_idx, i in enumerate(rank, 1):
        print(f"  {pos_idx:<6}{plans[i]:<12}{Dp[i]:>14.4f}{Dn[i]:>14.4f}{C[i]:>12.4f}")

    best = rank[0]
    print(f"\n  >>> 结论：最佳 NIPT 时点为 {plans[best]}（C = {C[best]:.4f}）")
    print(f"      次优：{plans[rank[1]]}（C = {C[rank[1]]:.4f}）")

    # ---------- 自检 ----------
    print("\n【自检】")
    ok = all(0.0 <= c <= 1.0 for c in C)
    print(f"  所有贴近度 C ∈ [0,1]？ {'✅ 通过' if ok else '❌ 失败'}")
    print(f"  D+ 最小者是否第一名？ {'✅ 是' if min(range(6), key=lambda i: Dp[i]) == best else '⚠️ 否（说明 D- 起了主导作用）'}")

    # ---------- 敏感性分析 ----------
    print("\n" + "=" * 72)
    print("  【敏感性分析】换权重，结论还稳吗？")
    print("=" * 72)

    weight_sets = [
        [0.30, 0.20, 0.30, 0.20],   # 基准：均衡
        [0.50, 0.20, 0.20, 0.10],   # 偏「准确性」：最看重达标率
        [0.15, 0.10, 0.45, 0.30],   # 偏「尽早发现」：最看重早期收益与风险
        [0.25, 0.25, 0.25, 0.25],   # 等权
    ]
    labels = ["均衡", "偏准确性", "偏尽早发现", "等权"]

    print(f"\n  {'权重方案':<14}{'第一名':<12}{'第二名':<12}{'第三名':<12}{'稳定性'}")
    print("  " + "-" * 62)
    sens = sensitivity(raw, kinds, weight_sets, labels)
    firsts = []
    for lab, w, rk, sc in sens:
        firsts.append(rk[0])
        stable = "✅ 稳定" if rk[0] == best else "⚠️ 改变"
        top3 = " / ".join(plans[rk[k]] for k in range(3))
        parts = top3.split(" / ")
        print(f"  {lab:<16}{parts[0]:<14}{parts[1]:<14}{parts[2]:<14}{stable}")

    same = len(set(firsts)) == 1
    print(f"\n  结论：{'✅ 4 组权重下第一名完全一致 —— 结论稳健，论文中可放心使用' if same else '⚠️ 第一名随权重变化 —— 论文中必须如实说明结论的适用偏好范围'}")

    # ---------- 方法边界 ----------
    print("\n" + "=" * 72)
    print("  【什么时候不该用 TOPSIS】")
    print("=" * 72)
    print("""
  1. 指标间高度相关 —— 会重复计权。先用 PCA 或相关分析降维，再来 TOPSIS。
  2. 权重全靠拍脑袋且没做敏感性分析 —— 评委一问「为什么这组权重」就倒。
  3. 方案数量很多（>30）且指标很多 —— 区分度会退化，C 值挤在一起难分高下。
  4. 需要给出绝对优劣判断 —— TOPSIS 只给相对排序，C=0.8 不代表「好」，
     只代表「在这批方案里最好」。换批方案，C 值全变。
  5. 成本型指标里有 0 或负数 —— 极差变换会失效，必须先平移或用倒数法。
""")
    print("=" * 72)

    return result


if __name__ == "__main__":
    demo_nipt_timing()
