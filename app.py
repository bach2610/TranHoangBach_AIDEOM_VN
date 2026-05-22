"""
╔══════════════════════════════════════════════════════════════════════════════╗
║         AIDEOM-VN: AI-Driven Economic Optimization Model - Vietnam          ║
║         Dashboard Hỗ Trợ Ra Quyết Định Phát Triển Kinh Tế Số               ║
║         Môn: Các Mô Hình Ra Quyết Định | Viện QTKD - ĐHKT                  ║
╚══════════════════════════════════════════════════════════════════════════════╝

Cấu trúc 6 Module:
  M1 - Dự báo kinh tế (Cobb-Douglas TFP + Growth Accounting)
  M2 - Đánh giá sẵn sàng số (TOPSIS + Entropy Weighting)
  M3 - Tối ưu phân bổ ngân sách (LP + MIP)
  M4 - Mô phỏng tác động lao động (NetJob Optimizer)
  M5 - Tối ưu đa mục tiêu & ngẫu nhiên (NSGA-II + Stochastic LP)
  M6 - Dashboard tổng hợp 5 kịch bản chính sách

Cài đặt: pip install streamlit plotly pulp cvxpy pymoo pyomo numpy pandas scipy
Chạy:    streamlit run app.py
"""

# ─────────────────────────────────────────────────────────────────────────────
# IMPORTS
# ─────────────────────────────────────────────────────────────────────────────
import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings("ignore")

# Optional heavy deps — fallback gracefully
try:
    import pulp
    PULP_OK = True
except ImportError:
    PULP_OK = False

try:
    import cvxpy as cp
    CVXPY_OK = True
except ImportError:
    CVXPY_OK = False

try:
    from pymoo.core.problem import ElementwiseProblem
    from pymoo.algorithms.moo.nsga2 import NSGA2
    from pymoo.optimize import minimize as pymoo_minimize
    PYMOO_OK = True
except ImportError:
    PYMOO_OK = False

try:
    import pyomo.environ as pyo
    PYOMO_OK = True
except ImportError:
    PYOMO_OK = False

# ─────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG & THEME
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AIDEOM-VN Dashboard",
    page_icon="🇻🇳",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS — dark government-data aesthetic with red-gold accent
st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Be+Vietnam+Pro:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

  html, body, [class*="css"] {
    font-family: 'Be Vietnam Pro', sans-serif;
  }
  .main { background: #0d1117; color: #e6edf3; }
  .block-container { padding: 1.5rem 2rem; }

  /* Sidebar */
  section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #161b22 0%, #0d1117 100%);
    border-right: 1px solid #21262d;
  }

  /* Module header cards */
  .module-header {
    background: linear-gradient(135deg, #1a1f2e 0%, #161b22 100%);
    border: 1px solid #30363d;
    border-left: 4px solid #c9a227;
    border-radius: 8px;
    padding: 1.2rem 1.5rem;
    margin-bottom: 1.5rem;
  }
  .module-header h2 { color: #c9a227; margin: 0 0 0.3rem 0; font-size: 1.3rem; font-weight: 700; }
  .module-header p  { color: #8b949e; margin: 0; font-size: 0.88rem; }

  /* KPI metric cards */
  .kpi-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(160px, 1fr)); gap: 1rem; margin: 1rem 0; }
  .kpi-card {
    background: #161b22;
    border: 1px solid #30363d;
    border-radius: 8px;
    padding: 1rem;
    text-align: center;
  }
  .kpi-card .kpi-value { font-size: 1.6rem; font-weight: 700; color: #58a6ff; font-family: 'JetBrains Mono', monospace; }
  .kpi-card .kpi-label { font-size: 0.75rem; color: #8b949e; margin-top: 0.3rem; }
  .kpi-card.gold .kpi-value { color: #c9a227; }
  .kpi-card.green .kpi-value { color: #3fb950; }
  .kpi-card.red   .kpi-value { color: #f85149; }

  /* Section divider */
  .section-title {
    font-size: 0.9rem;
    font-weight: 600;
    color: #58a6ff;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    padding: 0.5rem 0;
    border-bottom: 1px solid #21262d;
    margin: 1.2rem 0 1rem 0;
  }

  /* Info box */
  .info-box {
    background: #0d2137;
    border: 1px solid #1f6feb;
    border-radius: 6px;
    padding: 0.8rem 1rem;
    color: #79c0ff;
    font-size: 0.85rem;
    margin: 0.8rem 0;
  }

  /* Warning box */
  .warn-box {
    background: #2d1a00;
    border: 1px solid #d29922;
    border-radius: 6px;
    padding: 0.8rem 1rem;
    color: #f0c67f;
    font-size: 0.85rem;
    margin: 0.8rem 0;
  }

  /* Streamlit overrides */
  .stSelectbox label, .stSlider label, .stNumberInput label { color: #c9d1d9 !important; font-size: 0.85rem !important; }
  .stButton > button {
    background: linear-gradient(90deg, #c9a227, #e6c84a);
    color: #0d1117;
    font-weight: 600;
    border: none;
    border-radius: 6px;
    padding: 0.5rem 1.5rem;
  }
  .stButton > button:hover { opacity: 0.88; }
  div[data-testid="metric-container"] {
    background: #161b22;
    border: 1px solid #30363d;
    border-radius: 8px;
    padding: 0.8rem;
  }
  div[data-testid="metric-container"] label { color: #8b949e !important; }
  .stDataFrame { border-radius: 8px; overflow: hidden; }
  .stExpander { border: 1px solid #30363d; border-radius: 8px; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTS & BUILT-IN DATA
# ─────────────────────────────────────────────────────────────────────────────
YEARS = [2020, 2021, 2022, 2023, 2024, 2025]
REGIONS = ["Trung du MN phía Bắc", "Đồng bằng sông Hồng", "Bắc Trung Bộ & DH Trung Bộ",
           "Tây Nguyên", "Đông Nam Bộ", "Đồng bằng sông Cửu Long"]
REGIONS_SHORT = ["NMM", "RRD", "NCC", "CH", "SE", "MD"]
ITEMS = ["Hạ tầng số (I)", "Chuyển đổi số DN (D)", "Trí tuệ AI (AI)", "Nhân lực số (H)"]
ITEMS_SHORT = ["I", "D", "AI", "H"]
SECTORS = ["Nông-Lâm-Thủy sản", "CN Chế biến-Chế tạo", "Xây dựng",
           "Bán buôn-Bán lẻ", "Tài chính-Ngân hàng", "Logistics-Vận tải",
           "CNTT-Truyền thông", "Giáo dục-Đào tạo"]
PROJECTS = {i: n for i, n in enumerate([
    "P1: TTDL Quốc gia Hòa Lạc", "P2: TTDL Quốc gia phía Nam", "P3: 5G toàn quốc",
    "P4: VNeID 2.0", "P5: Cổng DVC Quốc gia v3", "P6: Y tế số quốc gia",
    "P7: Giáo dục số K-12", "P8: Trung tâm AI Quốc gia", "P9: Sandbox Fintech",
    "P10: Logistics thông minh", "P11: Nông nghiệp số ĐBSCL", "P12: 50k kỹ sư AI",
    "P13: Khu CN bán dẫn Bắc Ninh", "P14: An ninh mạng SOC", "P15: Open Data"
], start=1)}

MACRO_DATA = {
    "year": YEARS,
    "GDP_trillion_VND": [8044.4, 8487.5, 9513.3, 10221.8, 11511.9, 12847.6],
    "K": [16500, 17800, 19600, 21300, 23500, 25900],
    "L": [53.6, 50.5, 51.7, 52.4, 52.9, 53.4],
    "D_pct": [12.0, 12.7, 14.3, 16.5, 18.3, 19.5],
    "AI_k": [55.6, 60.2, 65.4, 67.0, 73.8, 80.1],
    "H_pct": [24.1, 26.1, 26.2, 27.0, 28.4, 29.2],
}

REGIONS_DATA = {
    "region": REGIONS,
    "grdp_per_capita": [57.0, 152.3, 87.5, 68.9, 158.9, 80.5],
    "fdi": [3.5, 20.0, 8.2, 0.8, 18.5, 2.1],
    "digital_index": [38, 78, 55, 32, 82, 48],
    "ai_readiness": [22, 68, 40, 18, 75, 30],
    "trained_labor": [21.5, 36.8, 27.5, 18.2, 42.5, 16.8],
    "rd_intensity": [0.18, 0.85, 0.32, 0.15, 0.78, 0.22],
    "internet": [72, 92, 84, 68, 94, 78],
    "gini": [0.405, 0.358, 0.372, 0.412, 0.385, 0.392],
}

SECTOR_DATA = {
    "sector": SECTORS,
    "growth": [3.27, 9.64, 7.45, -1.20, 7.10, 7.36, 9.93, 7.85],
    "productivity": [103.4, 241.2, 168.8, 1290.5, 145.3, 1072.4, 321.4, 713.8],
    "spillover": [0.35, 0.78, 0.42, 0.30, 0.55, 0.85, 0.72, 0.92],
    "export": [40.5, 290.9, 2.5, 8.2, 5.5, 1.2, 3.1, 178.0],
    "labor": [13.20, 11.50, 4.80, 0.30, 7.80, 0.55, 1.95, 0.62],
    "ai_readiness": [15, 55, 20, 30, 48, 72, 42, 88],
    "automation_risk": [18, 42, 25, 55, 38, 52, 35, 28],
}

PLOTLY_THEME = dict(
    template="plotly_dark",
    paper_bgcolor="#0d1117",
    plot_bgcolor="#161b22",
    font_color="#c9d1d9",
    font_family="Be Vietnam Pro",
)

# ─────────────────────────────────────────────────────────────────────────────
# UTILITY FUNCTIONS
# ─────────────────────────────────────────────────────────────────────────────
def module_header(title: str, subtitle: str):
    st.markdown(f"""
    <div class="module-header">
      <h2>🔷 {title}</h2>
      <p>{subtitle}</p>
    </div>""", unsafe_allow_html=True)

def section(title: str):
    st.markdown(f'<div class="section-title">{title}</div>', unsafe_allow_html=True)

def info(msg: str):
    st.markdown(f'<div class="info-box">ℹ️ {msg}</div>', unsafe_allow_html=True)

def warn(msg: str):
    st.markdown(f'<div class="warn-box">⚠️ {msg}</div>', unsafe_allow_html=True)

def make_fig(**kwargs):
    """Create a plotly figure with default dark theme."""
    fig = go.Figure(**kwargs)
    fig.update_layout(**PLOTLY_THEME, margin=dict(t=40, b=30, l=30, r=20))
    return fig

# ─────────────────────────────────────────────────────────────────────────────
# MODULE 1 — Cobb-Douglas TFP + Growth Accounting
# ─────────────────────────────────────────────────────────────────────────────
def module_m1():
    module_header(
        "M1 · Dự Báo Kinh Tế — Cobb-Douglas Mở Rộng",
        "Ước lượng TFP, phân rã tăng trưởng và dự báo GDP Việt Nam 2026-2030 (Bài 1)"
    )

    df = pd.DataFrame(MACRO_DATA)
    Y  = df["GDP_trillion_VND"].values
    K  = df["K"].values
    L  = df["L"].values
    D  = df["D_pct"].values
    AI = df["AI_k"].values
    H  = df["H_pct"].values

    # ── Controls ──────────────────────────────────────────────────────────────
    section("⚙️ Tham Số Hàm Sản Xuất")
    c1, c2, c3, c4, c5 = st.columns(5)
    alpha = c1.slider("α (Vốn vật chất K)", 0.1, 0.6, 0.33, 0.01)
    beta  = c2.slider("β (Lao động L)", 0.1, 0.6, 0.42, 0.01)
    gamma = c3.slider("γ (Số hóa D)", 0.01, 0.25, 0.10, 0.01)
    delta = c4.slider("δ (Năng lực AI)", 0.01, 0.20, 0.08, 0.01)
    theta = c5.slider("θ (Nhân lực số H)", 0.01, 0.20, 0.07, 0.01)

    if abs(alpha+beta+gamma+delta+theta - 1.0) > 0.05:
        warn(f"Tổng hệ số = {alpha+beta+gamma+delta+theta:.2f} ≠ 1 (vi phạm CRS). Điều chỉnh lại.")

    # ── Compute TFP ───────────────────────────────────────────────────────────
    M_prod = (K**alpha) * (L**beta) * (D**gamma) * (AI**delta) * (H**theta)
    A_t = Y / M_prod
    A_mean = A_t.mean()
    Y_hat = A_mean * M_prod
    MAPE = np.mean(np.abs((Y - Y_hat) / Y)) * 100

    # ── KPI row ───────────────────────────────────────────────────────────────
    section("📊 Kết Quả Chính")
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("TFP Trung bình (Ā)", f"{A_mean:.4f}")
    k2.metric("MAPE Dự báo", f"{MAPE:.2f}%")
    k3.metric("TFP 2025 / TFP 2020", f"{A_t[-1]/A_t[0]:.3f}x")
    k4.metric("GDP 2025 Thực tế (ngh.tỷ VND)", "12.847,6")

    # ── TFP Chart ─────────────────────────────────────────────────────────────
    col_l, col_r = st.columns(2)

    with col_l:
        section("Xu hướng TFP (A_t) 2020–2025")
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=YEARS, y=A_t,
            mode="lines+markers+text",
            name="TFP",
            line=dict(color="#c9a227", width=3),
            marker=dict(size=9, color="#c9a227"),
            text=[f"{v:.4f}" for v in A_t],
            textposition="top center",
            textfont=dict(size=9),
        ))
        fig.add_hline(y=A_mean, line_dash="dash", line_color="#58a6ff", annotation_text=f"Ā={A_mean:.4f}")
        fig.update_layout(**PLOTLY_THEME, title="Năng Suất Nhân Tố Tổng Hợp TFP (A_t)",
                          xaxis_title="Năm", yaxis_title="TFP")
        st.plotly_chart(fig, use_container_width=True)

    with col_r:
        section("GDP Thực tế vs Dự báo")
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(x=YEARS, y=Y, mode="lines+markers", name="GDP Thực tế",
                                  line=dict(color="#3fb950", width=3), marker=dict(size=8)))
        fig2.add_trace(go.Scatter(x=YEARS, y=Y_hat, mode="lines+markers", name="GDP Dự báo",
                                  line=dict(color="#f85149", width=2, dash="dash"), marker=dict(size=8)))
        fig2.update_layout(**PLOTLY_THEME, title=f"GDP Thực tế vs Dự báo (MAPE = {MAPE:.2f}%)",
                           xaxis_title="Năm", yaxis_title="Nghìn tỷ VND")
        st.plotly_chart(fig2, use_container_width=True)

    # ── Growth Accounting ────────────────────────────────────────────────────
    section("📈 Phân Rã Tăng Trưởng GDP (Growth Accounting) 2020–2025")
    log_Y = np.log(Y)
    dY = np.diff(log_Y)
    dK_contrib  = alpha  * np.diff(np.log(K))
    dL_contrib  = beta   * np.diff(np.log(L))
    dD_contrib  = gamma  * np.diff(np.log(D))
    dAI_contrib = delta  * np.diff(np.log(AI))
    dH_contrib  = theta  * np.diff(np.log(H))
    dA_contrib  = dY - dK_contrib - dL_contrib - dD_contrib - dAI_contrib - dH_contrib

    avg = lambda arr: np.mean(arr) * 100
    contrib_pct = {
        "Vốn vật chất K": avg(dK_contrib / dY),
        "Lao động L":      avg(dL_contrib / dY),
        "Số hóa D":        avg(dD_contrib / dY),
        "Năng lực AI":     avg(dAI_contrib / dY),
        "Nhân lực số H":   avg(dH_contrib / dY),
        "TFP (Đổi mới)":   avg(dA_contrib / dY),
    }

    ga_col, bar_col = st.columns([1, 1.4])
    with ga_col:
        ga_df = pd.DataFrame({"Yếu tố": list(contrib_pct.keys()),
                               "Đóng góp (%)": [round(v, 2) for v in contrib_pct.values()]})
        st.dataframe(ga_df.style.background_gradient(subset=["Đóng góp (%)"], cmap="YlOrRd"),
                     use_container_width=True, hide_index=True)
    with bar_col:
        colors = ["#58a6ff", "#3fb950", "#c9a227", "#e6c84a", "#79c0ff", "#f0883e"]
        fig3 = go.Figure(go.Bar(
            x=list(contrib_pct.keys()), y=list(contrib_pct.values()),
            marker_color=colors, text=[f"{v:.1f}%" for v in contrib_pct.values()],
            textposition="outside"
        ))
        fig3.update_layout(**PLOTLY_THEME, title="Đóng Góp Vào Tăng Trưởng GDP (%)",
                           yaxis_title="%", xaxis_tickangle=-20)
        st.plotly_chart(fig3, use_container_width=True)

    # ── 2030 Forecast ─────────────────────────────────────────────────────────
    section("🔭 Dự Báo GDP 2030 — Kịch Bản Người Dùng")
    f1, f2, f3, f4 = st.columns(4)
    D_2030  = f1.slider("Kinh tế số/GDP 2030 (%)", 20, 40, 30)
    AI_2030 = f2.slider("Doanh nghiệp số 2030 (nghìn)", 80, 150, 100)
    H_2030  = f3.slider("Lao động qua ĐT 2030 (%)", 29, 45, 35)
    KL_g    = f4.slider("Tăng trưởng K & L (%/năm)", 3, 10, 6)
    TFP_g   = st.slider("Tăng trưởng TFP (%/năm)", 0.5, 3.0, 1.2, 0.1)

    t_steps = 5  # 2025 → 2030
    K_2030 = K[-1] * (1 + KL_g / 100) ** t_steps
    L_2030 = L[-1] * (1 + KL_g / 100) ** t_steps
    A_2030 = A_mean * (1 + TFP_g / 100) ** t_steps
    Y_2030 = A_2030 * (K_2030**alpha) * (L_2030**beta) * \
             (D_2030**gamma) * (AI_2030**delta) * (H_2030**theta)

    st.markdown(f"""
    <div class="kpi-grid">
      <div class="kpi-card gold"><div class="kpi-value">{Y_2030:,.0f}</div><div class="kpi-label">GDP 2030 Dự báo (ngh.tỷ VND)</div></div>
      <div class="kpi-card green"><div class="kpi-value">{(Y_2030/Y[-1]-1)*100:.1f}%</div><div class="kpi-label">Tăng trưởng so với 2025</div></div>
      <div class="kpi-card"><div class="kpi-value">{Y_2030/102300:.2f}</div><div class="kpi-label">GDP/người 2030 (tr.VND)</div></div>
      <div class="kpi-card"><div class="kpi-value">{A_2030:.4f}</div><div class="kpi-label">TFP 2030 Dự báo</div></div>
    </div>""", unsafe_allow_html=True)

    with st.expander("📝 Bình luận chính sách — Bài 1"):
        st.markdown("""
**TFP Việt Nam 2020-2025** có xu hướng **tăng** — đây là dấu hiệu tích cực cho thấy chất lượng tăng
trưởng đang cải thiện, chuyển từ chiều rộng sang chiều sâu.

**Yếu tố đóng góp lớn nhất** trong 3 yếu tố mới là **Số hóa (D)** (~13-14%), phản ánh đúng bùng nổ
thanh toán điện tử và TMĐT hậu COVID-19. AI capacity đóng góp ít hơn do quy mô still nhỏ.

**Mục tiêu 30% kinh tế số/GDP vào 2030** khả thi về mặt toán học nhưng cần ràng buộc đồng bộ:
H ≥ 35%, AI ≥ 100k doanh nghiệp, và TFP tăng tối thiểu 1.2%/năm.
        """)


# ─────────────────────────────────────────────────────────────────────────────
# MODULE 2 — TOPSIS + Entropy Weighting
# ─────────────────────────────────────────────────────────────────────────────
def module_m2():
    module_header(
        "M2 · Đánh Giá Sẵn Sàng Số — TOPSIS & Entropy",
        "Xếp hạng 6 vùng kinh tế theo mức độ ưu tiên đầu tư AI (Bài 3 & 6)"
    )

    # ── TOPSIS logic ──────────────────────────────────────────────────────────
    def run_topsis(X, w, is_benefit):
        norm = X / np.sqrt((X**2).sum(axis=0))
        V    = norm * w
        A_p  = np.where(is_benefit, V.max(0), V.min(0))
        A_n  = np.where(is_benefit, V.min(0), V.max(0))
        S_p  = np.sqrt(((V - A_p)**2).sum(1))
        S_n  = np.sqrt(((V - A_n)**2).sum(1))
        return S_n / (S_p + S_n)

    def entropy_weights(X):
        P   = X / X.sum(0)
        k   = 1.0 / np.log(len(X))
        E   = -k * np.nansum(P * np.log(P + 1e-12), 0)
        d   = 1 - E
        return d / d.sum()

    df_r = pd.DataFrame(REGIONS_DATA)
    crit = ["grdp_per_capita", "fdi", "digital_index", "ai_readiness",
            "trained_labor", "rd_intensity", "internet", "gini"]
    crit_labels = ["GRDP/người", "FDI", "Chỉ số số", "AI Readiness",
                   "LĐ Đào tạo", "R&D/GRDP", "Internet", "Gini"]
    is_benefit = np.array([True]*7 + [False])
    X = df_r[crit].values.astype(float)

    # ── Weight sliders ────────────────────────────────────────────────────────
    section("⚙️ Trọng Số Tiêu Chí")
    st.caption("Kéo thanh trượt để thay đổi trọng số (tổng sẽ được chuẩn hóa tự động)")
    sc = st.columns(8)
    raw_w = []
    defaults = [0.10, 0.10, 0.15, 0.20, 0.15, 0.15, 0.05, 0.10]
    for i, (lbl, d) in enumerate(zip(crit_labels, defaults)):
        raw_w.append(sc[i].number_input(lbl, 0.0, 1.0, d, 0.05, key=f"w{i}"))
    w_expert = np.array(raw_w)
    w_expert = w_expert / w_expert.sum()

    # ── Entropy weights ───────────────────────────────────────────────────────
    w_entropy = entropy_weights(X)

    # ── Compute scores ────────────────────────────────────────────────────────
    scores_e = run_topsis(X, w_expert, is_benefit)
    scores_n = run_topsis(X, w_entropy, is_benefit)

    df_res = pd.DataFrame({
        "Vùng": REGIONS,
        "TOPSIS (Chuyên gia)": np.round(scores_e, 4),
        "TOPSIS (Entropy)": np.round(scores_n, 4),
        "Hạng (Chuyên gia)": scores_e.argsort()[::-1].argsort() + 1,
        "Hạng (Entropy)": scores_n.argsort()[::-1].argsort() + 1,
    }).sort_values("Hạng (Chuyên gia)")

    col_l, col_r = st.columns([1.1, 1])
    with col_l:
        section("Bảng Xếp Hạng")
        st.dataframe(df_res.style.background_gradient(subset=["TOPSIS (Chuyên gia)", "TOPSIS (Entropy)"],
                                                       cmap="RdYlGn"),
                     use_container_width=True, hide_index=True)

        section("Trọng Số Entropy vs Chuyên Gia")
        fig_w = go.Figure()
        fig_w.add_trace(go.Bar(name="Chuyên gia", x=crit_labels, y=w_expert,
                               marker_color="#c9a227"))
        fig_w.add_trace(go.Bar(name="Entropy", x=crit_labels, y=w_entropy,
                               marker_color="#58a6ff"))
        fig_w.update_layout(**PLOTLY_THEME, barmode="group", title="So Sánh Trọng Số",
                            xaxis_tickangle=-20, showlegend=True)
        st.plotly_chart(fig_w, use_container_width=True)

    with col_r:
        section("Radar Chart — Năng Lực Số Từng Vùng")
        # Normalize for radar
        X_n = (X - X.min(0)) / (X.max(0) - X.min(0) + 1e-9)
        colors_r = ["#c9a227", "#58a6ff", "#3fb950", "#f85149", "#e6c84a", "#79c0ff"]
        fig_radar = go.Figure()
        for i, (rn, clr) in enumerate(zip(REGIONS, colors_r)):
            vals = list(X_n[i]) + [X_n[i][0]]
            fig_radar.add_trace(go.Scatterpolar(
                r=vals, theta=crit_labels + [crit_labels[0]],
                fill="toself", name=rn[:20], line_color=clr, opacity=0.6))
        fig_radar.update_layout(**PLOTLY_THEME, polar=dict(
            bgcolor="#161b22",
            radialaxis=dict(visible=True, range=[0, 1], gridcolor="#30363d"),
            angularaxis=dict(gridcolor="#30363d")),
            title="Chuẩn hóa năng lực số 6 vùng", showlegend=True)
        st.plotly_chart(fig_radar, use_container_width=True)

    # ── Sector Priority ──────────────────────────────────────────────────────
    section("🏭 Bài 3 — Chỉ Số Ưu Tiên Ngành (Priority Index)")
    df_s = pd.DataFrame(SECTOR_DATA)
    cols_good = ["growth", "productivity", "spillover", "export", "labor", "ai_readiness"]
    def norm_good(x): return (x - x.min()) / (x.max() - x.min() + 1e-9)
    def norm_bad(x):  return (x.max() - x) / (x.max() - x.min() + 1e-9)

    df_n = df_s.copy()
    for c in cols_good:
        df_n[c] = norm_good(df_s[c])
    df_n["automation_risk"] = norm_bad(df_s["automation_risk"])

    w_def = np.array([0.15, 0.15, 0.20, 0.15, 0.10, 0.20])
    w_risk = 0.15
    df_n["Priority"] = df_n[cols_good].values @ w_def - w_risk * df_n["automation_risk"].values
    df_n["Sector"] = df_s["sector"]
    df_n = df_n.sort_values("Priority", ascending=False)

    fig_pri = px.bar(df_n, x="Priority", y="Sector", orientation="h",
                     color="Priority", color_continuous_scale=["#1a1f2e", "#c9a227"],
                     text=df_n["Priority"].round(3))
    fig_pri.update_layout(**PLOTLY_THEME, title="Priority Index — 10 Ngành Kinh Tế Việt Nam",
                          xaxis_title="Điểm Priority", yaxis_title="")
    st.plotly_chart(fig_pri, use_container_width=True)

    with st.expander("📝 Bình luận chính sách — Bài 3 & 6"):
        st.markdown("""
**TOPSIS với trọng số chuyên gia**: Đông Nam Bộ và Đồng bằng sông Hồng dẫn đầu nhờ ưu thế
tổng hợp về AI Readiness, FDI và lao động đào tạo → Đây là 2 vùng ưu tiên đặt trung tâm AI.

**Entropy weights** gán trọng số cao cho FDI (phân tán lớn) và giảm trọng số Gini (phân tán nhỏ),
đôi khi thay đổi thứ hạng các vùng trung bình.

**Priority Index ngành**: CNTT-Truyền thông, CN Chế biến-Chế tạo và Tài chính-NH dẫn đầu
— phù hợp định hướng Nghị quyết 57-NQ/TW về chọn ngành lan tỏa cao.
        """)


# ─────────────────────────────────────────────────────────────────────────────
# MODULE 3 — LP Budget Allocation + MIP Project Selection
# ─────────────────────────────────────────────────────────────────────────────
def module_m3():
    module_header(
        "M3 · Tối Ưu Phân Bổ Ngân Sách — LP & MIP",
        "Phân bổ ngân sách số theo vùng & lựa chọn dự án chiến lược (Bài 2, 4, 5)"
    )

    tab1, tab2, tab3 = st.tabs(["📊 Bài 2 — LP Đơn Giản", "🗺️ Bài 4 — LP Vùng × Hạng Mục", "📋 Bài 5 — MIP Dự Án"])

    # ── TAB 1: Bài 2 ──────────────────────────────────────────────────────────
    with tab1:
        section("⚙️ Tham Số")
        budget = st.slider("Ngân sách tổng (nghìn tỷ VND)", 80, 160, 100, 5, key="b2")
        min_x3 = st.slider("Sàn đầu tư Nhân lực số x₃ ≥", 15, 40, 20, key="x3")

        coeffs = [0.85, 1.20, 0.95, 1.35]
        mins   = [25, 15, min_x3, 10]
        labels = ["x₁: Hạ tầng số", "x₂: AI & Dữ liệu", "x₃: Nhân lực số", "x₄: R&D Công nghệ"]

        if PULP_OK:
            m = pulp.LpProblem("B2", pulp.LpMaximize)
            xs = [pulp.LpVariable(f"x{i+1}", lowBound=0) for i in range(4)]
            m += sum(c*x for c, x in zip(coeffs, xs))
            m += sum(xs) <= budget
            for i in range(4): m += xs[i] >= mins[i]
            m += 0.35*(xs[0]+xs[2]) <= 0.65*(xs[1]+xs[3])
            m.solve(pulp.PULP_CBC_CMD(msg=False))

            if pulp.LpStatus[m.status] == "Optimal":
                vals = [v.varValue for v in xs]
                Z    = pulp.value(m.objective)
                sp   = m.constraints["_C1"].pi if "_C1" in m.constraints else 1.35

                k1, k2, k3 = st.columns(3)
                k1.metric("Z* GDP Kỳ vọng", f"{Z:,.2f} ngh.tỷ")
                k2.metric("Shadow Price (Ngân sách)", f"{sp:.4f}")
                k3.metric("Hiệu suất (Z/Budget)", f"{Z/budget:.4f}")

                fig_b2 = go.Figure(go.Bar(
                    x=labels, y=vals,
                    text=[f"{v:.1f}" for v in vals],
                    textposition="outside",
                    marker_color=["#58a6ff", "#c9a227", "#3fb950", "#f0883e"]
                ))
                fig_b2.update_layout(**PLOTLY_THEME, title=f"Phân Bổ Tối Ưu (Budget={budget})",
                                     yaxis_title="Nghìn tỷ VND")
                st.plotly_chart(fig_b2, use_container_width=True)

                # Sensitivity
                section("📈 Phân Tích Độ Nhạy Z*(Budget)")
                b_range = range(100, 161, 10)
                z_vals = []
                for b in b_range:
                    m2 = pulp.LpProblem("s", pulp.LpMaximize)
                    xs2 = [pulp.LpVariable(f"x{i}", lowBound=0) for i in range(4)]
                    m2 += sum(c*x for c,x in zip(coeffs, xs2))
                    m2 += sum(xs2) <= b
                    for i in range(4): m2 += xs2[i] >= mins[i]
                    m2 += 0.35*(xs2[0]+xs2[2]) <= 0.65*(xs2[1]+xs2[3])
                    m2.solve(pulp.PULP_CBC_CMD(msg=False))
                    z_vals.append(pulp.value(m2.objective) if m2.status == 1 else None)
                fig_s = make_fig()
                fig_s.add_trace(go.Scatter(x=list(b_range), y=z_vals, mode="lines+markers",
                                           line=dict(color="#c9a227", width=2), marker_color="#c9a227"))
                fig_s.update_layout(title="Z*(Budget) — Đường cong tối ưu",
                                    xaxis_title="Ngân sách", yaxis_title="Z*")
                st.plotly_chart(fig_s, use_container_width=True)
            else:
                warn(f"Bài toán không khả thi với x₃ ≥ {min_x3}. Giảm ràng buộc sàn.")
        else:
            warn("PuLP chưa được cài đặt. Chạy: `pip install pulp`")

    # ── TAB 2: Bài 4 ──────────────────────────────────────────────────────────
    with tab2:
        beta_mat = np.array([
            [1.15, 0.85, 0.55, 1.30],
            [0.95, 1.25, 1.40, 1.05],
            [1.05, 0.95, 0.85, 1.15],
            [1.20, 0.75, 0.45, 1.35],
            [0.90, 1.30, 1.55, 1.00],
            [1.10, 0.85, 0.65, 1.25]
        ])
        D0_arr = np.array([38, 78, 55, 32, 82, 48])

        c1, c2, c3 = st.columns(3)
        total_b4 = c1.slider("Tổng ngân sách (tỷ VND)", 30000, 60000, 50000, 1000, key="b4")
        lam = c2.slider("λ — Ngưỡng công bằng số", 0.5, 0.9, 0.7, 0.05, key="lam4")
        fair_on = c3.checkbox("Bật ràng buộc công bằng (C5)", True, key="fair4")

        if PULP_OK:
            prob = pulp.LpProblem("B4", pulp.LpMaximize)
            xv = pulp.LpVariable.dicts("x", (REGIONS_SHORT, ITEMS_SHORT), lowBound=0)
            M  = pulp.LpVariable("M", lowBound=0)

            obj = pulp.lpSum(beta_mat[ri, ji] * xv[r][j]
                             for ri, r in enumerate(REGIONS_SHORT)
                             for ji, j in enumerate(ITEMS_SHORT))
            prob += obj
            prob += pulp.lpSum(xv[r][j] for r in REGIONS_SHORT for j in ITEMS_SHORT) <= total_b4
            for r in REGIONS_SHORT:
                prob += pulp.lpSum(xv[r][j] for j in ITEMS_SHORT) >= 5000
                prob += pulp.lpSum(xv[r][j] for j in ITEMS_SHORT) <= 12000
            prob += pulp.lpSum(xv[r]["H"] for r in REGIONS_SHORT) >= 12000

            if fair_on:
                gam = 0.002
                for ri, r in enumerate(REGIONS_SHORT):
                    prob += D0_arr[ri] + gam * xv[r]["D"] <= M
                    prob += D0_arr[ri] + gam * xv[r]["D"] >= lam * M

            prob.solve(pulp.PULP_CBC_CMD(msg=False))

            if prob.status == 1:
                Z4 = pulp.value(prob.objective)
                alloc = np.array([[xv[r][j].varValue for j in ITEMS_SHORT] for r in REGIONS_SHORT])

                st.metric("Z* GDP Gain (tỷ VND)", f"{Z4:,.0f}")
                df_alloc = pd.DataFrame(alloc, index=REGIONS, columns=ITEMS)
                section("Ma Trận Phân Bổ Ngân Sách Tối Ưu (tỷ VND)")
                st.dataframe(df_alloc.style.background_gradient(cmap="YlOrRd").format("{:,.0f}"),
                             use_container_width=True)

                fig_h = go.Figure(go.Heatmap(
                    z=alloc, x=ITEMS, y=REGIONS,
                    colorscale=[[0, "#161b22"], [0.5, "#c9a227"], [1, "#f85149"]],
                    text=[[f"{v:,.0f}" for v in row] for row in alloc],
                    texttemplate="%{text}"
                ))
                fig_h.update_layout(**PLOTLY_THEME, title="Heatmap Phân Bổ Ngân Sách")
                st.plotly_chart(fig_h, use_container_width=True)
            else:
                warn("Bài toán không khả thi. Kiểm tra tham số ràng buộc.")
        else:
            warn("PuLP chưa cài đặt.")

    # ── TAB 3: Bài 5 MIP ─────────────────────────────────────────────────────
    with tab3:
        C  = {1:12000,2:11500,3:18000,4:4500,5:3200,6:5800,7:6500,8:15000,
              9:2500,10:7200,11:4800,12:8500,13:20000,14:3800,15:1500}
        C1 = {1:8500,2:7500,3:12000,4:3500,5:2500,6:4000,7:4500,8:9000,
              9:1800,10:5000,11:3500,12:5500,13:13000,14:2800,15:1200}
        B  = {1:21500,2:20800,3:32500,4:9200,5:6800,6:11400,7:12200,8:28500,
              9:5800,10:13800,11:8500,12:16200,13:35000,14:7500,15:3800}
        P  = list(range(1, 16))

        c1b, c2b = st.columns(2)
        budget5 = c1b.slider("Ngân sách tổng 5 năm (tỷ)", 70000, 120000, 80000, 5000, key="b5")
        force_p12 = c2b.checkbox("Bắt buộc P1 + P2 (redundancy)", False, key="fp12")

        if PULP_OK:
            m5 = pulp.LpProblem("B5", pulp.LpMaximize)
            y5 = pulp.LpVariable.dicts("y", P, cat="Binary")
            m5 += pulp.lpSum(B[i]*y5[i] for i in P)
            m5 += pulp.lpSum(C[i]*y5[i] for i in P) <= budget5
            m5 += pulp.lpSum(C1[i]*y5[i] for i in P) <= 40000
            if force_p12:
                m5 += y5[1] >= 1; m5 += y5[2] >= 1
            else:
                m5 += y5[1] + y5[2] <= 1
            m5 += y5[8] <= y5[12]
            m5 += y5[13] <= y5[12]
            m5 += y5[4] + y5[5] >= 1
            m5 += y5[14] >= 1
            m5 += pulp.lpSum(y5[i] for i in P) >= 7
            m5 += pulp.lpSum(y5[i] for i in P) <= 11
            m5.solve(pulp.PULP_CBC_CMD(msg=False))

            if m5.status == 1:
                sel = [i for i in P if y5[i].varValue > 0.5]
                Z5  = pulp.value(m5.objective)
                tot_c = sum(C[i] for i in sel)

                k1c, k2c, k3c = st.columns(3)
                k1c.metric("Z* Tổng lợi ích (tỷ VND)", f"{Z5:,.0f}")
                k2c.metric("Số dự án được chọn", len(sel))
                k3c.metric("NPV biên (Z*/Chi phí)", f"{Z5/tot_c:.3f}")

                df_proj = pd.DataFrame({
                    "Dự Án": [PROJECTS[i] for i in sel],
                    "Chi phí (tỷ)": [C[i] for i in sel],
                    "Lợi ích NPV (tỷ)": [B[i] for i in sel],
                    "Tỷ suất (B/C)": [round(B[i]/C[i], 2) for i in sel],
                })
                st.dataframe(df_proj.style.background_gradient(subset=["Tỷ suất (B/C)"], cmap="RdYlGn"),
                             use_container_width=True, hide_index=True)

                # Waterfall benefit chart
                fig_wf = go.Figure(go.Bar(
                    x=[PROJECTS[i].split(":")[0] for i in sel],
                    y=[B[i] for i in sel],
                    text=[f"{B[i]:,}" for i in sel],
                    textposition="outside",
                    marker_color=["#c9a227" if B[i]/C[i] >= 2 else "#58a6ff" for i in sel]
                ))
                fig_wf.update_layout(**PLOTLY_THEME, title="Lợi Ích NPV Dự Án Được Chọn",
                                     yaxis_title="Tỷ VND", xaxis_tickangle=-30)
                st.plotly_chart(fig_wf, use_container_width=True)
            else:
                warn("Bài toán không khả thi. Có thể ràng buộc P1+P2 vượt giai đoạn 1.")
        else:
            warn("PuLP chưa cài đặt.")


# ─────────────────────────────────────────────────────────────────────────────
# MODULE 4 — Labour Market Simulation
# ─────────────────────────────────────────────────────────────────────────────
def module_m4():
    module_header(
        "M4 · Mô Phỏng Thị Trường Lao Động — NetJob Optimizer",
        "Tối ưu hóa tác động AI tới việc làm 8 ngành Việt Nam (Bài 9)"
    )

    risk = np.array([18, 42, 25, 38, 52, 35, 28, 22]) / 100
    a1   = np.array([8.5, 32.5, 12.8, 22.4, 45.8, 28.5, 62.5, 18.5])
    b1   = np.array([45, 28, 35, 32, 22, 30, 20, 55])
    c1   = np.array([5.2, 62.4, 18.5, 48.2, 72.5, 42.8, 32.5, 12.5])
    d1   = np.array([50, 32, 42, 38, 26, 36, 24, 62])
    labor = np.array([13.20, 11.50, 4.80, 7.80, 0.55, 1.95, 0.62, 2.15])  # triệu LĐ

    section("⚙️ Cài Đặt Bài Toán")
    bud_m4 = st.slider("Tổng ngân sách (tỷ VND)", 10000, 50000, 30000, 1000, key="bm4")
    add_5pct = st.checkbox("Thêm ràng buộc: Displaced ≤ 5% lao động ngành", False, key="c5pct")

    if CVXPY_OK:
        N = 8
        x_AI = cp.Variable(N, nonneg=True)
        x_H  = cp.Variable(N, nonneg=True)

        NewJob  = cp.multiply(a1, x_AI)
        Upgrade = cp.multiply(b1, x_H)
        Displ   = cp.multiply(cp.multiply(c1, risk), x_AI)
        Retrain = cp.multiply(d1, x_H)
        NetJob  = NewJob + Upgrade - Displ

        constrs = [
            cp.sum(x_AI + x_H) <= bud_m4,
            NetJob >= 0,
            Displ <= Retrain
        ]
        if add_5pct:
            constrs += [Displ <= 0.05 * labor * 1e6 / 1e3]  # scaled units

        prob_m4 = cp.Problem(cp.Maximize(cp.sum(NetJob)), constrs)
        prob_m4.solve(solver=cp.ECOS, warm_start=True)

        if prob_m4.status in ["optimal", "optimal_inaccurate"]:
            netjob_vals = NetJob.value
            xai_vals    = x_AI.value
            xh_vals     = x_H.value
            displ_vals  = Displ.value

            total_net = netjob_vals.sum()
            total_displ = displ_vals.sum()

            c1c, c2c, c3c = st.columns(3)
            c1c.metric("Tổng NetJob Ròng (nghìn việc)", f"{total_net:,.0f}")
            c2c.metric("Tổng việc làm bị dịch chuyển", f"{total_displ:,.0f}")
            c3c.metric("Trạng thái tối ưu", prob_m4.status.capitalize())

            df_labor = pd.DataFrame({
                "Ngành": SECTORS,
                "x_AI (tỷ)": np.round(xai_vals, 1),
                "x_H (tỷ)": np.round(xh_vals, 1),
                "Việc làm mới": np.round((a1 * xai_vals), 0).astype(int),
                "Nâng cấp kỹ năng": np.round((b1 * xh_vals), 0).astype(int),
                "Bị thay thế": np.round(displ_vals, 0).astype(int),
                "NetJob (nghìn)": np.round(netjob_vals, 1),
            })
            st.dataframe(
                df_labor.style.applymap(
                    lambda v: "color: #3fb950" if isinstance(v, (int, float)) and v > 0
                              else ("color: #f85149" if isinstance(v, (int, float)) and v < 0 else ""),
                    subset=["NetJob (nghìn)"]
                ).background_gradient(subset=["x_AI (tỷ)", "x_H (tỷ)"], cmap="Blues"),
                use_container_width=True, hide_index=True
            )

            # Stacked bar
            fig_lm = go.Figure()
            fig_lm.add_trace(go.Bar(name="Việc làm mới (AI)", x=SECTORS,
                                    y=a1 * xai_vals, marker_color="#3fb950"))
            fig_lm.add_trace(go.Bar(name="Nâng cấp kỹ năng (H)", x=SECTORS,
                                    y=b1 * xh_vals, marker_color="#58a6ff"))
            fig_lm.add_trace(go.Bar(name="Bị thay thế (–)", x=SECTORS,
                                    y=-displ_vals, marker_color="#f85149"))
            fig_lm.update_layout(**PLOTLY_THEME, barmode="relative",
                                 title="Cán Cân Việc Làm Theo Ngành (Nghìn Việc)",
                                 yaxis_title="Nghìn việc làm", xaxis_tickangle=-25)
            st.plotly_chart(fig_lm, use_container_width=True)

            # Risk heatmap
            fig_risk = go.Figure(go.Bar(
                x=SECTORS, y=risk * 100,
                marker_color=[f"rgb({int(200*r)},{int(60*(1-r))},50)" for r in risk],
                text=[f"{v:.0f}%" for v in risk * 100], textposition="outside"
            ))
            fig_risk.update_layout(**PLOTLY_THEME, title="Rủi Ro Tự Động Hóa Theo Ngành (%)")
            st.plotly_chart(fig_risk, use_container_width=True)
        else:
            warn(f"Trạng thái solver: {prob_m4.status}. Thử tắt ràng buộc 5%.")
    else:
        warn("CVXPY chưa cài đặt. Chạy: `pip install cvxpy`")


# ─────────────────────────────────────────────────────────────────────────────
# MODULE 5 — Multi-Objective & Stochastic LP
# ─────────────────────────────────────────────────────────────────────────────
def module_m5():
    module_header(
        "M5 · Đánh Giá Rủi Ro — Đa Mục Tiêu & Ngẫu Nhiên",
        "NSGA-II Pareto frontier + Two-Stage Stochastic LP (Bài 7 & 10)"
    )

    tab_nsga, tab_sp = st.tabs(["🎯 Bài 7 — NSGA-II Pareto", "🎲 Bài 10 — Stochastic LP"])

    # ── NSGA-II ──────────────────────────────────────────────────────────────
    with tab_nsga:
        section("⚙️ Tham Số NSGA-II")
        c1, c2 = st.columns(2)
        pop_size = c1.slider("Population Size", 50, 200, 100, 10, key="pop5")
        n_gen    = c2.slider("Số thế hệ (n_gen)", 50, 300, 150, 10, key="ngen5")

        w_nsga = st.columns(4)
        wg = w_nsga[0].slider("w₁ Tăng trưởng", 0.0, 1.0, 0.40, 0.05, key="w1n")
        wb = w_nsga[1].slider("w₂ Bao trùm",    0.0, 1.0, 0.25, 0.05, key="w2n")
        we = w_nsga[2].slider("w₃ Môi trường",  0.0, 1.0, 0.20, 0.05, key="w3n")
        ws = w_nsga[3].slider("w₄ An ninh số",  0.0, 1.0, 0.15, 0.05, key="w4n")
        w_topsis_nsga = np.array([wg, wb, we, ws])
        if w_topsis_nsga.sum() > 0:
            w_topsis_nsga /= w_topsis_nsga.sum()

        run_btn = st.button("▶ Chạy NSGA-II", key="run_nsga")

        if run_btn:
            if not PYMOO_OK:
                warn("pymoo chưa cài đặt. Chạy: `pip install pymoo`")
            else:
                beta_mat = np.array([
                    [1.15,0.85,0.55,1.30],[0.95,1.25,1.40,1.05],
                    [1.05,0.95,0.85,1.15],[1.20,0.75,0.45,1.35],
                    [0.90,1.30,1.55,1.00],[1.10,0.85,0.65,1.25]
                ])
                e   = np.array([0.42,0.55,0.48,0.32,0.62,0.38])
                rho = np.array([0.18,0.45,0.28,0.12,0.52,0.22])
                sig = np.array([0.32,0.28,0.30,0.35,0.25,0.30])

                class VP(ElementwiseProblem):
                    def __init__(self):
                        super().__init__(n_var=24,n_obj=4,n_ieq_constr=14,
                                         xl=np.zeros(24),xu=np.ones(24)*12000)
                    def _evaluate(self,x,out,*args,**kwargs):
                        X = x.reshape(6,4)
                        f1 = -(beta_mat*X).sum()
                        s  = X.sum(1); f2 = np.abs(s-s.mean()).mean()
                        f3 = (e*(X[:,0]+X[:,2])).sum()
                        f4 = (rho*X[:,2]).sum()-(sig*X[:,3]).sum()
                        g  = [X.sum()-50000]
                        g += [5000-X[r].sum() for r in range(6)]
                        g += [X[r].sum()-12000 for r in range(6)]
                        g += [12000-X[:,3].sum()]
                        out["F"]=[f1,f2,f3,f4]; out["G"]=g

                with st.spinner(f"Đang tối ưu hóa... ({n_gen} thế hệ)"):
                    res = pymoo_minimize(VP(), NSGA2(pop_size=pop_size),
                                        ("n_gen", n_gen), seed=42, verbose=False)

                F = res.F
                st.success(f"✅ Tìm được {len(F)} nghiệm Pareto!")

                # 3D scatter
                fig3d = go.Figure(go.Scatter3d(
                    x=-F[:,0], y=F[:,1], z=F[:,2],
                    mode="markers",
                    marker=dict(
                        size=4,
                        color=F[:,3],
                        colorscale=[[0,"#3fb950"],[0.5,"#c9a227"],[1,"#f85149"]],
                        colorbar=dict(title="An ninh rủi ro (f4)"),
                        opacity=0.8
                    )
                ))
                fig3d.update_layout(**PLOTLY_THEME,
                                    scene=dict(
                                        xaxis_title="f1: GDP Gain",
                                        yaxis_title="f2: Bất bình đẳng",
                                        zaxis_title="f3: Phát thải",
                                        bgcolor="#161b22"
                                    ),
                                    title="Đường Biên Pareto 3D (f1, f2, f3)")
                st.plotly_chart(fig3d, use_container_width=True)

                # TOPSIS on Pareto set
                F_std = F.copy()
                is_b_nsga = np.array([True, False, False, False])
                scores_pareto = []
                for sol in F_std:
                    # scale each obj 0-1
                    pass
                F_norm = (F - F.min(0)) / (F.max(0) - F.min(0) + 1e-9)
                A_p = np.where(is_b_nsga, F_norm.min(0), F_norm.max(0))
                A_n = np.where(is_b_nsga, F_norm.max(0), F_norm.min(0))
                sp  = np.sqrt(((F_norm - A_p)**2 * w_topsis_nsga).sum(1))
                sn  = np.sqrt(((F_norm - A_n)**2 * w_topsis_nsga).sum(1))
                c_star = sn / (sp + sn)
                best_idx = c_star.argmax()

                sec_c1, sec_c2 = st.columns(2)
                with sec_c1:
                    st.success(f"**Nghiệm thỏa hiệp TOPSIS** (idx {best_idx}):")
                    comp = ["GDP Gain", "Bất bình đẳng", "Phát thải", "Rủi ro an ninh"]
                    for lbl, val in zip(comp, F[best_idx]):
                        st.write(f"• {lbl}: `{val:,.2f}`")

                with sec_c2:
                    # Trade-off growth vs equity
                    fig_to = go.Figure()
                    fig_to.add_trace(go.Scatter(x=-F[:,0], y=F[:,1], mode="markers",
                                                marker=dict(size=4, color="#58a6ff", opacity=0.6),
                                                name="Pareto set"))
                    fig_to.add_trace(go.Scatter(x=[-F[best_idx,0]], y=[F[best_idx,1]],
                                                mode="markers",
                                                marker=dict(size=12, color="#c9a227", symbol="star"),
                                                name="Nghiệm thỏa hiệp"))
                    fig_to.update_layout(**PLOTLY_THEME, title="Đánh Đổi: Tăng Trưởng vs Bao Trùm",
                                         xaxis_title="GDP Gain (f1)", yaxis_title="Bất bình đẳng (f2)")
                    st.plotly_chart(fig_to, use_container_width=True)
        else:
            info("Nhấn **▶ Chạy NSGA-II** để tính toán. Thời gian khoảng 15-60 giây tùy tham số.")

    # ── Stochastic LP ─────────────────────────────────────────────────────────
    with tab_sp:
        section("🎲 Two-Stage Stochastic Programming (Pyomo)")
        J   = ["I", "D", "AI", "H"]
        S   = ["s1", "s2", "s3", "s4"]
        p_s = {"s1":0.30,"s2":0.45,"s3":0.20,"s4":0.05}
        b   = {"I":1.00,"D":1.10,"AI":1.25,"H":0.95}
        bs  = {("s1","I"):1.25,("s1","D"):1.35,("s1","AI"):1.55,("s1","H"):1.05,
               ("s2","I"):1.00,("s2","D"):1.10,("s2","AI"):1.25,("s2","H"):0.95,
               ("s3","I"):0.75,("s3","D"):0.85,("s3","AI"):0.90,("s3","H"):1.00,
               ("s4","I"):0.40,("s4","D"):0.50,("s4","AI"):0.55,("s4","H"):1.10}

        c1s, c2s = st.columns(2)
        b1_sp = c1s.slider("Ngân sách giai đoạn 1 (tỷ)", 50000, 75000, 65000, 1000, key="sp1")
        b2_sp = c2s.slider("Ngân sách dự phòng GĐ 2 (tỷ)", 5000, 25000, 15000, 1000, key="sp2")

        run_sp = st.button("▶ Giải Stochastic LP", key="run_sp")
        if run_sp:
            if not PYOMO_OK:
                warn("Pyomo chưa cài. Chạy: `pip install pyomo` và `conda install glpk`")
            else:
                try:
                    m = pyo.ConcreteModel()
                    m.J = pyo.Set(initialize=J)
                    m.S = pyo.Set(initialize=S)
                    m.x = pyo.Var(m.J, within=pyo.NonNegativeReals)
                    m.y = pyo.Var(m.S, m.J, within=pyo.NonNegativeReals)
                    m.b1c = pyo.Constraint(expr=sum(m.x[j] for j in J) <= b1_sp)
                    m.b2c = pyo.ConstraintList()
                    m.lnk = pyo.ConstraintList()
                    for s in S:
                        m.b2c.add(sum(m.y[s,j] for j in J) <= b2_sp)
                        m.lnk.add(m.y[s,"AI"] <= 0.5*m.x["H"])
                    m.obj = pyo.Objective(
                        expr=sum(b[j]*m.x[j] for j in J) +
                             sum(p_s[s]*sum(bs[s,j]*m.y[s,j] for j in J) for s in S),
                        sense=pyo.maximize
                    )
                    solver = pyo.SolverFactory("glpk")
                    res_sp = solver.solve(m, tee=False)

                    x_vals  = {j: pyo.value(m.x[j]) for j in J}
                    Z_sp    = pyo.value(m.obj)

                    # EV solution (weighted average scenario)
                    b_ev = {j: sum(p_s[s]*bs[s,j] for s in S) for j in J}
                    m_ev = pyo.ConcreteModel()
                    m_ev.J = pyo.Set(initialize=J)
                    m_ev.x = pyo.Var(m_ev.J, within=pyo.NonNegativeReals)
                    m_ev.y = pyo.Var(m_ev.J, within=pyo.NonNegativeReals)
                    m_ev.b1c = pyo.Constraint(expr=sum(m_ev.x[j] for j in J) <= b1_sp)
                    m_ev.b2c = pyo.Constraint(expr=sum(m_ev.y[j] for j in J) <= b2_sp)
                    m_ev.lnk = pyo.Constraint(expr=m_ev.y["AI"] <= 0.5*m_ev.x["H"])
                    m_ev.obj = pyo.Objective(
                        expr=sum((b[j]+b_ev[j])*m_ev.x[j] for j in J),
                        sense=pyo.maximize
                    )
                    solver.solve(m_ev, tee=False)
                    x_ev = {j: pyo.value(m_ev.x[j]) for j in J}
                    Z_ev = pyo.value(m_ev.obj)

                    VSS = Z_sp - Z_ev

                    c1r, c2r, c3r = st.columns(3)
                    c1r.metric("Z* Stochastic (SP)", f"{Z_sp:,.0f}")
                    c2r.metric("Z* Deterministic (EV)", f"{Z_ev:,.0f}")
                    c3r.metric("VSS (Giá trị tư duy xác suất)", f"{VSS:,.0f}")

                    df_sp = pd.DataFrame({
                        "Hạng mục": J,
                        "SP: Quyết định GĐ 1 (tỷ)": [round(x_vals[j], 0) for j in J],
                        "EV: Quyết định GĐ 1 (tỷ)": [round(x_ev[j], 0) for j in J],
                        "β cơ bản": [b[j] for j in J],
                    })
                    st.dataframe(df_sp, use_container_width=True, hide_index=True)

                    fig_sp = go.Figure()
                    fig_sp.add_trace(go.Bar(name="SP (Ngẫu nhiên)", x=J,
                                           y=[x_vals[j] for j in J], marker_color="#c9a227"))
                    fig_sp.add_trace(go.Bar(name="EV (Xác định)", x=J,
                                           y=[x_ev[j] for j in J], marker_color="#58a6ff"))
                    fig_sp.update_layout(**PLOTLY_THEME, barmode="group",
                                        title=f"SP vs EV — Quyết Định Giai Đoạn 1 (VSS={VSS:,.0f})",
                                        yaxis_title="Tỷ VND")
                    st.plotly_chart(fig_sp, use_container_width=True)

                    if VSS > 0:
                        info(f"VSS = {VSS:,.0f} tỷ VND > 0 → Tư duy xác suất tạo thêm giá trị. "
                             "Chính phủ nên lập kế hoạch ngân sách theo kịch bản thay vì chốt cứng một con số.")
                    else:
                        info("VSS ≈ 0: Trong bài toán này, hai cách tiếp cận cho kết quả tương đương.")

                except Exception as ex:
                    warn(f"Lỗi Pyomo: {ex}. Kiểm tra GLPK đã cài chưa: `conda install -c conda-forge glpk`")
        else:
            # Show scenario tree visualization
            section("📊 Cây Kịch Bản Tương Lai")
            scen_df = pd.DataFrame({
                "Kịch bản": ["s1 Lạc quan", "s2 Cơ sở", "s3 Bi quan", "s4 Khủng hoảng"],
                "Xác suất": [0.30, 0.45, 0.20, 0.05],
                "Tăng trưởng TG (%)": [3.5, 2.8, 1.5, 0.2],
                "FDI VN (tỷ USD)": [32, 27, 20, 12],
                "Xuất khẩu tăng (%)": [12, 8, 3, -5],
            })
            st.dataframe(scen_df.style.background_gradient(subset=["Xác suất"], cmap="Blues"),
                         use_container_width=True, hide_index=True)
            info("Nhấn **▶ Giải Stochastic LP** để so sánh SP vs EV và tính VSS.")


# ─────────────────────────────────────────────────────────────────────────────
# MODULE 6 — 5 Policy Scenarios Dashboard
# ─────────────────────────────────────────────────────────────────────────────
def module_m6():
    module_header(
        "M6 · Dashboard Tổng Hợp — 5 Kịch Bản Chính Sách 2026-2030",
        "So sánh định lượng 5 kịch bản AIDEOM-VN theo 8 KPI kinh tế - xã hội (Bài 12)"
    )

    # ── Scenario definitions ──────────────────────────────────────────────────
    SCENARIOS = {
        "S1: Truyền thống":  dict(K=0.70, D=0.10, AI=0.10, H=0.10),
        "S2: Số hóa nhanh":  dict(K=0.25, D=0.45, AI=0.15, H=0.15),
        "S3: AI dẫn dắt":    dict(K=0.20, D=0.20, AI=0.45, H=0.15),
        "S4: Bao trùm số":   dict(K=0.30, D=0.20, AI=0.10, H=0.40),
        "S5: Tối ưu cân bằng": dict(K=0.30, D=0.28, AI=0.22, H=0.20),
    }
    SCN_COLORS = ["#8b949e", "#58a6ff", "#c9a227", "#3fb950", "#e6c84a"]

    section("⚙️ Tham Số Mô Phỏng")
    c1m6, c2m6, c3m6 = st.columns(3)
    budget_m6 = c1m6.slider("Tổng ngân sách 2026-2030 (ngh.tỷ/năm)", 1000, 3000, 1500, 100, key="bm6")
    years_m6  = c2m6.slider("Số năm mô phỏng", 5, 15, 10, key="ym6")
    tfp_rate  = c3m6.slider("Tăng trưởng TFP baseline (%/năm)", 0.5, 3.0, 1.2, 0.1, key="tm6")

    # ── Simulation ────────────────────────────────────────────────────────────
    def simulate_scenario(alloc, budget, T, tfp_g):
        K, D, AI, H = 27500, 20.3, 86, 30.0
        A = 1.6  # from M1 calibration
        L = 54.0
        alpha, beta, gamma, delta, theta = 0.33, 0.42, 0.10, 0.08, 0.07

        gdp_hist, k_hist, d_hist, ai_hist, h_hist = [], [], [], [], []
        for t in range(T):
            Y = A * (K**alpha) * (L**beta) * (D**gamma) * (AI**delta) * (H**theta)
            gdp_hist.append(Y); k_hist.append(K); d_hist.append(D)
            ai_hist.append(AI); h_hist.append(H)
            bud = budget
            K  += alloc["K"] * bud * 0.95 - 0.05 * K
            D  += alloc["D"] * bud / 100   - 0.12 * D
            AI += alloc["AI"] * bud / 20   - 0.15 * AI
            H  += alloc["H"] * bud / 200   - 0.02 * H
            A  *= (1 + tfp_g / 100)
        return gdp_hist, k_hist, d_hist, ai_hist, h_hist

    results = {}
    for sname, alloc in SCENARIOS.items():
        g, k, d, ai, h = simulate_scenario(alloc, budget_m6, years_m6, tfp_rate)
        results[sname] = {"gdp": g, "K": k, "D": d, "AI": ai, "H": h}

    sim_years = list(range(2026, 2026 + years_m6))

    # ── GDP Comparison Chart ──────────────────────────────────────────────────
    section("📊 Quỹ Đạo GDP Dự Báo Theo Kịch Bản")
    fig_gdp = go.Figure()
    for (sname, data), clr in zip(results.items(), SCN_COLORS):
        fig_gdp.add_trace(go.Scatter(
            x=sim_years, y=data["gdp"],
            name=sname, mode="lines+markers",
            line=dict(color=clr, width=2.5),
            marker=dict(size=5)
        ))
    fig_gdp.update_layout(**PLOTLY_THEME, title="Dự Báo GDP 5 Kịch Bản (Nghìn Tỷ VND)",
                          xaxis_title="Năm", yaxis_title="Nghìn tỷ VND", showlegend=True)
    st.plotly_chart(fig_gdp, use_container_width=True)

    # ── KPI Comparison Table ──────────────────────────────────────────────────
    section("📋 Bảng So Sánh KPI Cuối Kỳ")
    final_year = -1
    kpi_rows = []
    for sname, data in results.items():
        gdp_end  = data["gdp"][final_year]
        gdp_grow = (data["gdp"][final_year] / data["gdp"][0] - 1) * 100
        d_end    = data["D"][final_year]
        ai_end   = data["AI"][final_year]
        h_end    = data["H"][final_year]
        kpi_rows.append({
            "Kịch Bản": sname,
            f"GDP {sim_years[-1]} (ngh.tỷ)": round(gdp_end, 0),
            "Tăng trưởng (%)": round(gdp_grow, 1),
            "Kinh tế số (%)": round(d_end, 1),
            "DN công nghệ (nghìn)": round(ai_end, 0),
            "LĐ qua đào tạo (%)": round(h_end, 1),
        })
    kpi_df = pd.DataFrame(kpi_rows)
    best_gdp = kpi_df[f"GDP {sim_years[-1]} (ngh.tỷ)"].idxmax()

    def highlight_best(row):
        return ["background-color: #1c2f1c; color: #3fb950" if row.name == best_gdp else "" for _ in row]

    st.dataframe(kpi_df.style.apply(highlight_best, axis=1)
                               .background_gradient(subset=[f"GDP {sim_years[-1]} (ngh.tỷ)", "Tăng trưởng (%)"],
                                                     cmap="RdYlGn"),
                 use_container_width=True, hide_index=True)

    info(f"✅ Kịch bản GDP cao nhất: **{kpi_df.loc[best_gdp, 'Kịch Bản']}**")

    # ── Allocation Donut Charts ───────────────────────────────────────────────
    section("🍩 Cơ Cấu Phân Bổ Từng Kịch Bản")
    donut_cols = st.columns(5)
    for col, (sname, alloc) in zip(donut_cols, SCENARIOS.items()):
        vals  = [alloc["K"], alloc["D"], alloc["AI"], alloc["H"]]
        lbls  = ["K", "D", "AI", "H"]
        fig_d = go.Figure(go.Pie(
            labels=lbls, values=vals, hole=0.55,
            marker_colors=["#58a6ff", "#c9a227", "#3fb950", "#e6c84a"],
            textinfo="label+percent", textfont_size=10
        ))
        fig_d.update_layout(
            **PLOTLY_THEME,
            title=dict(text=sname.split(":")[0], font_size=12),
            showlegend=False,
            margin=dict(t=40, b=5, l=5, r=5),
            height=220
        )
        col.plotly_chart(fig_d, use_container_width=True)

    # ── Radar Comparison ─────────────────────────────────────────────────────
    section("🕸️ So Sánh Đa Chiều 5 Kịch Bản — Cuối Kỳ")
    radar_metrics = ["GDP", "Kinh tế số", "DN công nghệ", "LĐ đào tạo", "Tăng trưởng"]
    radar_fig = go.Figure()
    for (sname, kpi_row), clr in zip(zip(kpi_df["Kịch Bản"], kpi_df.itertuples()), SCN_COLORS):
        raw = [kpi_row[2], kpi_row[4], kpi_row[5], kpi_row[6], kpi_row[3]]
        mn = [min(kpi_df.iloc[:,i+1]) for i in range(5)]
        mx = [max(kpi_df.iloc[:,i+1]) for i in range(5)]
        norm = [(v-lo)/(hi-lo+1e-9) for v, lo, hi in zip(raw, mn, mx)]
        vals = norm + [norm[0]]
        lbls = radar_metrics + [radar_metrics[0]]
        radar_fig.add_trace(go.Scatterpolar(
            r=vals, theta=lbls, fill="toself",
            name=sname, line_color=clr, opacity=0.65
        ))
    radar_fig.update_layout(**PLOTLY_THEME,
                            polar=dict(bgcolor="#161b22",
                                       radialaxis=dict(range=[0, 1], gridcolor="#30363d"),
                                       angularaxis=dict(gridcolor="#30363d")),
                            title="Radar So Sánh 5 Kịch Bản (Chuẩn hóa)",
                            showlegend=True)
    st.plotly_chart(radar_fig, use_container_width=True)

    # ── Risk Alerts ──────────────────────────────────────────────────────────
    section("🚨 Bảng Cảnh Báo Rủi Ro")
    alerts = []
    for sname, alloc in SCENARIOS.items():
        if alloc["H"] < 0.15:
            alerts.append(f"⚠️ **{sname}**: Đầu tư nhân lực thấp ({alloc['H']*100:.0f}%) → Rủi ro thiếu hụt kỹ sư AI")
        if alloc["AI"] > 0.40:
            alerts.append(f"⚠️ **{sname}**: AI chiếm {alloc['AI']*100:.0f}% → Rủi ro phụ thuộc công nghệ ngoại")
        if alloc["D"] > 0.40:
            alerts.append(f"ℹ️ **{sname}**: Số hóa ưu tiên cao ({alloc['D']*100:.0f}%) → Cần đảm bảo an ninh mạng")
    if not alerts:
        alerts = ["✅ Tất cả kịch bản trong ngưỡng rủi ro chấp nhận được."]
    for a in alerts:
        st.markdown(a)

    with st.expander("📝 Khuyến nghị chính sách — Bài 12"):
        st.markdown("""
**S5 (Tối ưu cân bằng)** thường cho kết quả tổng hợp tốt nhất vì phân bổ hợp lý
giữa 4 hạng mục, tránh rủi ro tập trung.

**S3 (AI dẫn dắt)** cho GDP cao nhất nhưng kèm rủi ro:
- Thiếu nhân lực vận hành AI (H thấp)
- Phụ thuộc vào chuỗi cung ứng bán dẫn quốc tế

**Khuyến nghị**: Giai đoạn 2026-2028 nên theo S4 (đầu tư H để tạo nền tảng),
giai đoạn 2028-2030 chuyển sang S3 khi lực lượng kỹ sư AI đủ quy mô.
Đây là chiến lược "two-phase" tương tự kết luận từ Bài 8 (Dynamic Optimization).
        """)


# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────────────────────
def render_sidebar():
    with st.sidebar:
        st.markdown("""
        <div style="text-align:center; padding: 1rem 0;">
          <div style="font-size:2.5rem;">🇻🇳</div>
          <div style="font-size:1.1rem; font-weight:700; color:#c9a227; letter-spacing:0.05em;">AIDEOM-VN</div>
          <div style="font-size:0.72rem; color:#8b949e;">AI-Driven Economic Optimization Model</div>
          <div style="font-size:0.68rem; color:#484f58; margin-top:0.3rem;">Viện QTKD • ĐH Kinh Tế</div>
        </div>
        <hr style="border-color:#21262d; margin: 0.5rem 0 1rem 0;">
        """, unsafe_allow_html=True)

        module = st.radio(
            "📦 Chọn Module",
            options=[
                "🏠  Tổng quan AIDEOM-VN",
                "M1  Dự báo kinh tế",
                "M2  Đánh giá sẵn sàng số",
                "M3  Tối ưu phân bổ ngân sách",
                "M4  Mô phỏng lao động",
                "M5  Đa mục tiêu & ngẫu nhiên",
                "M6  Dashboard 5 kịch bản",
            ],
            key="module_select"
        )

        st.markdown("""
        <hr style="border-color:#21262d; margin: 1rem 0 0.8rem 0;">
        <div style="font-size:0.78rem; color:#8b949e; padding: 0 0.3rem;">
          <b style="color:#c9d1d9;">Phụ thuộc Python:</b><br>
          <code>streamlit plotly pulp cvxpy pymoo pyomo numpy pandas scipy</code>
        </div>
        <div style="margin-top:0.8rem; font-size:0.72rem; color:#484f58; padding: 0 0.3rem;">
          Dữ liệu: GSO/NSO Việt Nam 2020-2025<br>
          Mô hình: AIDEOM-VN (Bài báo gốc)<br>
          Môn: Các Mô Hình Ra Quyết Định
        </div>
        """, unsafe_allow_html=True)

    return module


# ─────────────────────────────────────────────────────────────────────────────
# HOME PAGE
# ─────────────────────────────────────────────────────────────────────────────
def page_home():
    st.markdown("""
    <div style="text-align:center; padding: 2rem 0 1rem 0;">
      <div style="font-size:3.5rem;">🇻🇳</div>
      <h1 style="color:#c9a227; font-size:2rem; margin:0.5rem 0; letter-spacing:0.05em;">
        AIDEOM-VN Dashboard
      </h1>
      <p style="color:#8b949e; font-size:1rem; max-width:600px; margin:0 auto;">
        Hệ thống Mô hình Ra Quyết Định Phát Triển Kinh Tế Việt Nam trong Kỷ Nguyên AI<br>
        <em>Môn: Các Mô Hình Ra Quyết Định — Viện Quản Trị Kinh Doanh, ĐH Kinh Tế</em>
      </p>
    </div>
    """, unsafe_allow_html=True)

    # Architecture diagram
    col1, col2, col3 = st.columns(3)
    cards = [
        ("M1 + M2", "Phân tích & Đánh giá", "Cobb-Douglas TFP | TOPSIS | Entropy Weights | Priority Index", "#58a6ff"),
        ("M3 + M4", "Tối ưu hóa", "LP Vùng-Hạng mục | MIP Dự án | NetJob Optimizer | Shadow Price", "#c9a227"),
        ("M5 + M6", "Rủi ro & Kịch bản", "NSGA-II Pareto | Stochastic LP | 5 Kịch bản chính sách | Dashboard", "#3fb950"),
    ]
    for col, (mod, title, desc, clr) in zip([col1, col2, col3], cards):
        col.markdown(f"""
        <div style="background:#161b22; border:1px solid #30363d; border-top:3px solid {clr};
                    border-radius:8px; padding:1.2rem; height:180px;">
          <div style="color:{clr}; font-weight:700; font-size:1rem; margin-bottom:0.4rem;">{mod}</div>
          <div style="color:#c9d1d9; font-weight:600; font-size:0.9rem; margin-bottom:0.6rem;">{title}</div>
          <div style="color:#8b949e; font-size:0.78rem; line-height:1.5;">{desc}</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    section("📊 Số Liệu Tham Chiếu Nhanh Việt Nam 2024-2025")
    quick_data = pd.DataFrame({
        "Chỉ tiêu": ["GDP (tỷ USD)", "Tăng trưởng GDP (%)", "FDI giải ngân (tỷ USD)",
                     "Kinh tế số / GDP (%)", "GII xếp hạng /139", "DN công nghệ số (nghìn)"],
        "2024": [476.3, 7.09, 25.35, 18.3, 44, 73.8],
        "2025": [514.0, 8.02, 27.60, "≈19.5", 44, 80.1],
        "Nguồn": ["NSO 2026", "NSO 2026", "MPI", "MIC/MoST", "WIPO 2025", "MoST 2026"],
    })
    st.dataframe(quick_data, use_container_width=True, hide_index=True)

    section("🗺️ Luồng Dữ Liệu AIDEOM-VN")
    info("""
    **Input**: vietnam_macro_2020_2025.csv | vietnam_sectors_2024.csv | vietnam_regions_2024.csv
    → **M1** Cobb-Douglas TFP → **M2** TOPSIS Readiness
    → **M3** LP/MIP Budget Allocation → **M4** NetJob Simulation
    → **M5** NSGA-II + Stochastic LP → **M6** 5-Scenario Dashboard
    → **Output**: Khuyến nghị phân bổ nguồn lực tối ưu cho Việt Nam 2026-2030
    """)

    section("⚠️ Kiểm Tra Môi Trường")
    env_cols = st.columns(5)
    deps = [("PuLP", PULP_OK), ("CVXPY", CVXPY_OK), ("pymoo", PYMOO_OK), ("Pyomo", PYOMO_OK),
            ("Plotly", True)]
    for col, (lib, ok) in zip(env_cols, deps):
        icon = "✅" if ok else "❌"
        clr  = "#3fb950" if ok else "#f85149"
        col.markdown(f"<div style='text-align:center; color:{clr}; font-size:0.85rem;'>{icon}<br>{lib}</div>",
                     unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# MAIN ROUTER
# ─────────────────────────────────────────────────────────────────────────────
def main():
    module = render_sidebar()

    if   "Tổng quan"   in module: page_home()
    elif "M1"          in module: module_m1()
    elif "M2"          in module: module_m2()
    elif "M3"          in module: module_m3()
    elif "M4"          in module: module_m4()
    elif "M5"          in module: module_m5()
    elif "M6"          in module: module_m6()

    # Footer
    st.markdown("""
    <div style="text-align:center; color:#484f58; font-size:0.72rem; margin-top:3rem; padding:1rem;
                border-top: 1px solid #21262d;">
      AIDEOM-VN • Dữ liệu: GSO/NSO, World Bank, MoST Việt Nam 2020-2025 •
      Môn Các Mô Hình Ra Quyết Định — Viện QTKD, ĐH Kinh Tế
    </div>""", unsafe_allow_html=True)


if __name__ == "__main__":
    main()
