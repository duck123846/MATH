"""
LeTID 兩步一階動力學：獨立計算程式（只需要 numpy 與 matplotlib）

模型（時間單位：小時）：
    da/dt = -k1*a          a = 未劣化態比例，a(0) = 1
    db/dt =  k1*a - k2*b   b = 劣化態比例，   b(0) = 0
    dc/dt =  k2*b          c = 再生態比例，   c(0) = 0

解析解：
    b(t) = k1/(k2-k1) * ( exp(-k1 t) - exp(-k2 t) )
    t*   = ln(k2/k1) / (k2-k1)               # 最大劣化的時刻
    b_max = (k1/k2) ** (k2/(k2-k1))          # 最大劣化量
    t_theta ≈ (1/k2) * ln( k1 / ((k1-k2)*theta) )   # b 降到 theta 以下所需時間（k1>k2、時間夠長）

使用方式：改下面「可調整的輸入」，然後執行  python letid_standalone.py
"""
import math
import numpy as np
import matplotlib
matplotlib.use("Agg")                      # 只存檔，不跳視窗；想看圖可刪掉這行
import matplotlib.pyplot as plt

# ============ 可調整的輸入 ============
T_C   = 75.0     # 測試溫度 [°C]（IEC TS 63342 為 75°C）
# 暗注入電流 = 2*(Isc - Imp)。預設取元晶太陽能 TSSCN（M10，182x182 mm）規格書的 23.00% 檔：
ISC_A    = 13.796        # 短路電流 [A]
IMP_A    = 13.055        # 最大功率點電流 [A]
AREA_CM2 = 18.2 * 18.2   # 電池面積 [cm²]（未扣倒角，略為高估面積）
T_END = 1500.0   # 圖的時間範圍 [h]
TEST_H = 504.0   # 標準加速測試長度 [h]（3 週）
THETA = 0.10     # 「再生完成」的判準：b 降到 10% 以下
# =====================================
J_APPLIED = 2.0 * (ISC_A - IMP_A) / AREA_CM2 * 1000.0   # 注入電流密度 [mA/cm²]
SUNS_EQ   = 2.0 * (ISC_A - IMP_A) / ISC_A               # 等效 sun 數（只用來顯示）

# ---- 文獻參數 ----
# 來源：Repins 2023 取自論文圖 1 表格；Kwapil、Bredemeier 取自 NREL PVDeg 0.7.4 的
#       DegradationDatabase.json（尚未對照原論文）。
# Ea 單位 eV；x 為載子密度的指數。
# k1_ref、k2_ref 是用 PVDeg 的 calc_dn 與 carrier_factor 在
#   T = 75°C、注入電流密度 4.473 mA/cm²（= 0.1074 sun × 41.65 mA/cm²，
#   由元晶 TSSCN 規格書 23.00% 檔的 Isc、Impp 與面積算出）、
#   PERC 電池（厚 180 µm、後表面複合速度 45 cm/s，取自 NREL 海報範例）
# 算出的速率 [1/h]，壽命取 115 µs（全為未劣化態）與 55 µs（全為劣化態）兩個極端。
PARAMS = {
    # 主模型：論文（Repins 2023 MRS Bulletin 式 5、6）的寫法，載子密度以 n_c = 1e15 cm^-3 = 1e21 m^-3 正規化。
    #   v、Ea、x 取自論文圖 1 表格（A→B：4.67e7 /s、0.827 eV、x=1；B→C：1.99e7 /s、0.871 eV、x=1.2）。
    "Repins 2023":     dict(ea1=0.827, ea2=0.871, x1=1.0, x2=1.2,
                            ref={115: (2.5001e-02, 1.6571e-03), 55: (1.3527e-02, 7.929e-04)}),
    # 對照：同一組 v、Ea，但用 PVDeg 的做法（以資料庫條件算出的 dn_lit = 7.06e20 m^-3 正規化）。
    "Repins（PVDeg）": dict(ea1=0.827, ea2=0.871, x1=1.0, x2=1.2,
                            ref={115: (3.5389e-02, 2.5144e-03), 55: (1.9148e-02, 1.2032e-03)}),
    "Kwapil 2017":     dict(ea1=0.827, ea2=0.871, x1=1.0, x2=1.2,
                            ref={115: (2.5007e-02, 1.6552e-03), 55: (1.3530e-02, 7.9203e-04)}),
    "Bredemeier 2017": dict(ea1=0.940, ea2=0.941, x1=1.0, x2=1.2,
                            ref={115: (7.6246e-03, 2.5849e-04), 55: (4.1254e-03, 1.2369e-04)}),
}
T_REF_C, J_REF = 75.0, 4.473   # 參考條件：75°C、4.473 mA/cm²
KB = 8.617333262e-5   # Boltzmann 常數 [eV/K]


def rates(p, tau, T_C=T_C, j_app=J_APPLIED):
    """把參考條件的 k 換算到指定的溫度與注入量。
    近似：Δn 與注入電流密度成正比，並忽略 Δn 本身隨溫度的小變化。
          與 PVDeg 完整計算相比（同一顆電池、溫度 60–85°C、注入 0.5–2 倍），
          速率差距在 0.5% 以內；若改用別顆電池（Jsc 不同），Bredemeier 那組
          在 PVDeg 內的算法會隨 Jsc 變動，差距可能到幾個百分點。"""
    k1r, k2r = p["ref"][tau]
    inv = 1.0 / (T_C + 273.15) - 1.0 / (T_REF_C + 273.15)
    k1 = k1r * (j_app / J_REF) ** p["x1"] * math.exp(-p["ea1"] / KB * inv)
    k2 = k2r * (j_app / J_REF) ** p["x2"] * math.exp(-p["ea2"] / KB * inv)
    return k1, k2


def b_of_t(t, k1, k2):
    if abs(k1 - k2) < 1e-12 * max(k1, k2):          # k1 = k2 的特例
        return k1 * t * np.exp(-k1 * t)
    return k1 / (k2 - k1) * (np.exp(-k1 * t) - np.exp(-k2 * t))


def t_star(k1, k2):
    return 1.0 / k1 if abs(k1 - k2) < 1e-12 * max(k1, k2) else math.log(k2 / k1) / (k2 - k1)


def b_max(k1, k2):
    return 1.0 / math.e if abs(k1 - k2) < 1e-12 * max(k1, k2) else (k1 / k2) ** (k2 / (k2 - k1))


def t_theta(k1, k2, theta=THETA):
    return math.log(k1 / ((k1 - k2) * theta)) / k2


def t_theta_exact(k1, k2, theta=THETA):
    """用二分法在峰值之後找 b = theta 的時刻，用來檢查近似式。"""
    lo = t_star(k1, k2); hi = lo
    while b_of_t(hi, k1, k2) > theta:
        hi *= 2
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if b_of_t(mid, k1, k2) > theta:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


# ============ 驗證（題目卷要求至少兩項獨立檢查）============
def verify(k1, k2):
    t = np.linspace(0.0, 2000.0, 20001)
    b = b_of_t(t, k1, k2)
    a = np.exp(-k1 * t)
    h = t[1] - t[0]
    db = np.gradient(b, h)
    resid = np.max(np.abs(db - (k1 * a - k2 * b)))             # 檢查 1：代回原方程式
    init = b_of_t(0.0, k1, k2)                                  # 檢查 2：初始條件 b(0)=0
    ts = t_star(k1, k2)
    peak_ok = (b_of_t(ts, k1, k2) >= b_of_t(ts * 0.99, k1, k2) and
               b_of_t(ts, k1, k2) >= b_of_t(ts * 1.01, k1, k2))  # 檢查 3：t* 確實是最大值
    bm_err = abs(b_of_t(ts, k1, k2) - b_max(k1, k2))             # 檢查 4：b_max 公式 = b(t*)
    tt_err = abs(t_theta(k1, k2) - t_theta_exact(k1, k2))        # 檢查 5：近似式 vs 二分法
    return resid, init, peak_ok, bm_err, tt_err


# ============ 計算與列表 ============
print(f"條件：T = {T_C:.0f} °C，注入 = {J_APPLIED:.2f} mA/cm²（等效 {SUNS_EQ:.4f} sun），"
      f"測試長度 = {TEST_H:.0f} h，theta = {THETA}")
print(f"{'參數組':16s}{'壽命µs':>7s}{'k1 [1/h]':>11s}{'k2 [1/h]':>11s}{'t* [h]':>9s}"
      f"{'b_max':>7s}{'b(504h)':>9s}{'t_theta[h]':>12s}")
table = {}
for name, p in PARAMS.items():
    for tau in (115, 55):
        k1, k2 = rates(p, tau)
        table[(name, tau)] = (k1, k2)
        print(f"{name:16s}{tau:7d}{k1:11.3e}{k2:11.3e}{t_star(k1,k2):9.1f}{b_max(k1,k2):7.3f}"
              f"{b_of_t(TEST_H,k1,k2):9.3f}{t_theta(k1,k2):12.0f}")

print("\n驗證（Repins 2023, 壽命 115 µs）：")
resid, init, peak_ok, bm_err, tt_err = verify(*table[("Repins 2023", 115)])
print(f"  代回原方程式最大殘差 = {resid:.2e} [1/h]（有限差分的誤差量級）")
print(f"  b(0) = {init:.1e}")
print(f"  t* 為最大值：{peak_ok}")
print(f"  |b(t*) - b_max| = {bm_err:.1e}")
print(f"  |t_theta 近似 - 二分法| = {tt_err:.2e} h")


# ============ 畫圖（主模型加完整標註；對照組用細線，不加標註）============
PLOT_END = 3200.0                      # 圖的時間範圍 [h]，要涵蓋 t_theta
MAIN = "Repins 2023"
plt.rcParams["font.sans-serif"] = ["Microsoft JhengHei", "PingFang TC", "Noto Sans CJK TC",
                                   "Arial Unicode MS", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False
t = np.linspace(0.0, PLOT_END, 3201)
fig, ax = plt.subplots(figsize=(10, 6))

# 對照組（細線）。Kwapil 資料庫條目與主模型數字相同，不重複畫。
for name, col in (("Repins（PVDeg）", "#7a5cc7"), ("Bredemeier 2017", "#1baf7a")):
    for tau, ls in ((115, "-"), (55, "--")):
        ax.plot(t, b_of_t(t, *table[(name, tau)]), color=col, lw=1.1, ls=ls, alpha=0.8,
                label=f"{name}（壽命 {tau} µs）")

# 主模型（粗線 + 標註）
c_main = "#2a78d6"
y115 = b_of_t(t, *table[(MAIN, 115)]); y55 = b_of_t(t, *table[(MAIN, 55)])
ax.fill_between(t, np.minimum(y115, y55), np.maximum(y115, y55), color=c_main, alpha=0.12, lw=0)
ax.plot(t, y115, color=c_main, lw=2.6, label=f"{MAIN}（論文式 6）壽命 115 µs")
ax.plot(t, y55, color=c_main, lw=2.6, ls="--", label=f"{MAIN}（論文式 6）壽命 55 µs")

# 關鍵點 1：初始值 b(0) = 0
ax.plot(0, 0, "s", color="black", ms=6)
ax.annotate("b(0)=0", (0, 0), (40, 0.04), fontsize=9)

# 關鍵點 2：峰值 (t*, b_max)，標示數字（文字放在圖上方空白帶）
for tau, xt in ((115, 20), (55, 1000)):
    k1, k2 = table[(MAIN, tau)]
    ts, bm = t_star(k1, k2), b_max(k1, k2)
    ax.plot(ts, bm, "o", color=c_main, mec="white", mew=1.5, ms=9, zorder=5)
    ax.annotate(f"壽命 {tau} µs：t* = {ts:.0f} h，b_max = {bm:.2f}", (ts, bm), (xt, 1.05),
                fontsize=9, color=c_main, va="bottom",
                arrowprops=dict(arrowstyle="-", color=c_main, lw=0.8))

# 關鍵點 3：3 週（504 h）時的 b
ax.axvline(TEST_H, color="gray", lw=1.2)
ax.text(TEST_H + 15, 1.16, f"標準測試上限 {TEST_H:.0f} h（3 週）", color="gray", fontsize=9, va="bottom")
for tau in (115, 55):
    k1, k2 = table[(MAIN, tau)]
    bv = b_of_t(TEST_H, k1, k2)
    ax.plot(TEST_H, bv, "D", color=c_main, mec="white", ms=7, zorder=5)
    ax.annotate(f"b(504 h)={bv:.2f}", (TEST_H, bv), (TEST_H + 70, bv + 0.03),
                fontsize=9, color=c_main, bbox=dict(fc="white", ec="none", alpha=0.85, pad=1))

# 關鍵點 4：門檻 b = 10%（視為完整再生）及其到達時間 t_theta
ax.axhline(THETA, color="#d03030", lw=1.2, ls=":", label=f"再生判準 b = {THETA:.0%}")
for tau in (115, 55):
    k1, k2 = table[(MAIN, tau)]
    tt = t_theta(k1, k2)
    if tt <= PLOT_END:
        ax.plot(tt, THETA, "v", color="#d03030", mec="white", ms=9, zorder=5)
        ax.annotate(f"t_θ={tt:.0f} h\n（約 {tt/24:.0f} 天）", (tt, THETA), (tt - 150, THETA + 0.13 + (0.0 if tau == 115 else 0.0)),
                    fontsize=9, color="#d03030", ha="center", arrowprops=dict(arrowstyle="-", color="#d03030", lw=0.8))

ax.set_xlabel("時間 t（小時）"); ax.set_ylabel("劣化態缺陷比例 b（無因次）")
ax.set_title(f"LeTID：{T_C:.0f}°C、暗注入 {J_APPLIED:.2f} mA/cm² 下的 b(t)")
ax.set_xlim(0, PLOT_END); ax.set_ylim(0, 1.22); ax.set_yticks(np.arange(0, 1.01, 0.2))
ax.grid(True, alpha=0.3); ax.legend(fontsize=8, loc="upper right", bbox_to_anchor=(1.0, 0.80))
fig.tight_layout()
fig.savefig("letid_b_of_t.png", dpi=150)
print("\n已存檔：letid_b_of_t.png")
