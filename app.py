# =====================================================================
# AIDEOM-VN — Mô hình Ra quyết định Phát triển Kinh tế Việt Nam
#             trong Kỷ nguyên AI
# Giao diện kiểm tra bài tập (Bài 1 -> Bài 12) bằng Streamlit
# Sinh viên: Trần Hoàng Bách — Viện Quản trị Kinh doanh
#            Đại học Kinh tế, ĐHQGHN
# Chạy:  streamlit run app.py
# =====================================================================
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

# ---------------------------------------------------------------------
# CẤU HÌNH TRANG & STYLE TỐI GIẢN
# ---------------------------------------------------------------------
st.set_page_config(page_title="AIDEOM-VN", page_icon="🇻🇳", layout="wide")

st.markdown(
    """
    <style>
      .block-container {padding-top: 2rem; padding-bottom: 3rem;}
      h1, h2, h3 {letter-spacing: -0.01em;}
      .stMetric {background: rgba(180,180,180,0.06); border-radius: 12px;
                 padding: 12px 16px;}
      div[data-testid="stSidebar"] {min-width: 270px;}
    </style>
    """,
    unsafe_allow_html=True,
)

# =====================================================================
# HÀM NẠP DỮ LIỆU CHUNG (đọc 3 tệp CSV gốc cùng thư mục với app.py)
# =====================================================================
@st.cache_data
def load_macro():
    df = pd.read_csv("vietnam_macro_2020_2025.csv")
    return df.sort_values("year").reset_index(drop=True)


@st.cache_data
def load_sectors():
    return pd.read_csv("vietnam_sectors_2024.csv")


@st.cache_data
def load_regions():
    return pd.read_csv("vietnam_regions_2024.csv")


# Tên tiếng Việt bổ sung (file gốc chỉ có tên tiếng Anh)
SECTOR_VI = {
    "Agriculture-Forestry-Fishery": "Nông-Lâm-Thủy sản",
    "Manufacturing": "CN chế biến chế tạo",
    "Construction": "Xây dựng",
    "Mining": "Khai khoáng",
    "Wholesale-Retail": "Bán buôn - Bán lẻ",
    "Finance-Banking-Insurance": "Tài chính - Ngân hàng",
    "Logistics-Transport-Warehousing": "Logistics - Vận tải",
    "Information-Communication-IT": "CNTT - Truyền thông",
    "Education-Training": "Giáo dục - Đào tạo",
    "Healthcare": "Y tế",
}
REGION_VI = {
    "Northern Midlands and Mountains": "Trung du & Miền núi Bắc",
    "Red River Delta": "Đồng bằng sông Hồng",
    "North Central and South Central Coast": "Bắc TB & DH Trung Bộ",
    "Central Highlands": "Tây Nguyên",
    "Southeast": "Đông Nam Bộ",
    "Mekong Delta": "Đồng bằng sông Cửu Long",
}

# Bảng chuỗi đầu vào theo Bảng 1.3 của đề (dùng cho Bài 1 & các bài sau)
K_SERIES = np.array([16500, 17800, 19600, 21300, 23500, 25900], dtype=float)
L_SERIES = np.array([53.6, 50.5, 51.7, 52.4, 52.9, 53.4], dtype=float)
D_SERIES = np.array([12.0, 12.7, 14.3, 16.5, 18.3, 19.5], dtype=float)
AI_SERIES = np.array([55.6, 60.2, 65.4, 67.0, 73.8, 80.1], dtype=float)
H_SERIES = np.array([24.1, 26.1, 26.2, 27.0, 28.4, 29.2], dtype=float)
ALPHA, BETA, GAMMA, DELTA, THETA = 0.33, 0.42, 0.10, 0.08, 0.07


def cobb_core(K, L, D, AI, H):
    return (K ** ALPHA) * (L ** BETA) * (D ** GAMMA) * (AI ** DELTA) * (H ** THETA)


# Bảng màu Plotly nhất quán toàn app
SEQ = px.colors.qualitative.Set2


# =====================================================================
# SIDEBAR — ĐIỀU HƯỚNG
# =====================================================================
st.sidebar.title("🇻🇳 AIDEOM-VN")
st.sidebar.caption("Mô hình ra quyết định phát triển kinh tế VN trong kỉ nguyên AI")

PAGES = [
    "🏠 Trang chủ Tổng quan Hệ thống",
    "🔹 Bài 1 — Cobb-Douglas + AI",
    "🔹 Bài 2 — LP ngân sách số",
    "🔹 Bài 3 — Priority 10 ngành",
    "🔹 Bài 4 — LP ngành-vùng",
    "🔹 Bài 5 — MIP 15 dự án",
    "🔹 Bài 6 — TOPSIS 6 vùng",
    "🔹 Bài 7 — NSGA-II Pareto",
    "🔹 Bài 8 — Tối ưu động 2026-2035",
    "🔹 Bài 9 — Lao động & AI",
    "🔹 Bài 10 — Stochastic 2 giai đoạn",
    "🔹 Bài 11 — Q-learning RL",
    "🚀 Bài 12 — Đồ án AIDEOM tích hợp",
]
choice = st.sidebar.radio("Điều hướng", PAGES, label_visibility="collapsed")

st.sidebar.markdown("---")
st.sidebar.markdown(
    "**Sinh viên thực hiện**  \n"
    "Trần Hoàng Bách  \n"
    "Viện Quản trị Kinh doanh  \n"
    "Đại học Kinh tế, ĐHQGHN"
)
st.sidebar.caption("Dữ liệu: NSO, MoST, MIC, MPI, WB, GII 2025")

# Nạp dữ liệu một lần dùng chung
macro = load_macro()
sectors = load_sectors()
regions = load_regions()
sectors = sectors.copy()
sectors["sector_name_vi"] = sectors["sector_name_en"].map(SECTOR_VI)
regions = regions.copy()
regions["region_name_vi"] = regions["region_name_en"].map(REGION_VI)


def header(title, goal):
    st.title(title)
    st.markdown(f"> **Mục tiêu kinh tế:** {goal}")
    st.markdown("---")


# =====================================================================
# 🏠 TRANG CHỦ TỔNG QUAN HỆ THỐNG
# =====================================================================
if choice == PAGES[0]:
    st.title("🇻🇳 AIDEOM-VN")
    st.subheader("AI-Driven Decision Optimization Model for Vietnam")
    st.markdown(
        "Web app giải **12 bài toán mô hình ra quyết định** phát triển kinh tế "
        "Việt Nam trong kỉ nguyên AI — dữ liệu thực 2020–2025. Hệ thống AIDEOM-VN "
        "chuyển bài toán chính sách kinh tế thành mô hình toán có ràng buộc và giải "
        "bằng các kỹ thuật tối ưu LP, MIP, đa mục tiêu, động, ngẫu nhiên và học tăng cường."
    )

    # --- Số liệu vĩ mô tham chiếu nhanh 2025 ---
    last = macro.iloc[-1]
    prev = macro.iloc[-2]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("GDP 2025", f"{last['GDP_billion_USD']:.1f} tỷ USD",
              f"{last['GDP_growth_pct']:.2f}%")
    c2.metric("Kinh tế số / GDP", f"≈{last['digital_economy_share_GDP_pct']:.1f}%",
              f"+{last['digital_economy_share_GDP_pct']-prev['digital_economy_share_GDP_pct']:.1f} đpt")
    c3.metric("FDI giải ngân 2025", f"{last['FDI_disbursed_billion_USD']:.1f} tỷ USD",
              f"+{(last['FDI_disbursed_billion_USD']/prev['FDI_disbursed_billion_USD']-1)*100:.1f}%")
    c4.metric("GDP/người 2025", f"{int(last['GDP_per_capita_USD']):,} USD",
              f"+{(last['GDP_per_capita_USD']/prev['GDP_per_capita_USD']-1)*100:.1f}%")

    st.markdown("---")
    st.subheader("📦 Xác nhận nạp dữ liệu (3 tệp CSV gốc)")
    st.caption("Kiểm tra Shape / số dòng / số cột & xem trước 5 dòng đầu của mỗi tệp.")

    for name, df in [
        ("vietnam_macro_2020_2025.csv", macro),
        ("vietnam_sectors_2024.csv", sectors),
        ("vietnam_regions_2024.csv", regions),
    ]:
        st.markdown(f"**`{name}`** — Shape `{df.shape}` "
                    f"({df.shape[0]} dòng × {df.shape[1]} cột)")
        st.dataframe(df.head(), use_container_width=True)

    st.markdown("---")
    st.subheader("🗺️ 12 bài toán theo 4 cấp độ")
    levels = pd.DataFrame({
        "Cấp độ": ["DỄ", "DỄ", "DỄ", "TRUNG BÌNH", "TRUNG BÌNH", "TRUNG BÌNH",
                    "KHÁ KHÓ", "KHÁ KHÓ", "KHÁ KHÓ", "KHÓ", "KHÓ", "KHÓ"],
        "Bài": [f"Bài {i}" for i in range(1, 13)],
        "Nội dung": [
            "Hàm sản xuất Cobb-Douglas mở rộng + AI — growth accounting, dự báo GDP 2030",
            "LP phân bổ ngân sách 4 hạng mục — scipy/pulp, shadow price",
            "Chỉ số ưu tiên 10 ngành — min-max norm, weighted scoring, sensitivity",
            "LP phân bổ ngân sách số ngành-vùng (PuLP + CVXPY), công bằng vùng",
            "MIP lựa chọn 15 dự án chuyển đổi số — knapsack + ràng buộc logic",
            "TOPSIS xếp hạng 6 vùng — entropy weight, AHP",
            "NSGA-II tối ưu đa mục tiêu — Pareto 4 mục tiêu xung đột",
            "Tối ưu động liên thời gian 2026-2035 — CVXPY, welfare chiết khấu",
            "Tác động AI tới lao động — NetJob ròng theo ngành",
            "Quy hoạch ngẫu nhiên 2 giai đoạn — Pyomo, VSS & EVPI",
            "Q-learning chính sách kinh tế thích nghi — MDP 81 trạng thái",
            "Đồ án tích hợp AIDEOM-VN — dashboard 5 kịch bản chính sách",
        ],
    })
    st.dataframe(levels, use_container_width=True, hide_index=True)


# =====================================================================
# --- BÀI 1 --- HÀM SẢN XUẤT COBB-DOUGLAS MỞ RỘNG VỚI AI & SỐ HÓA
# =====================================================================
elif choice == PAGES[1]:
    header("🔹 Bài 1 — Hàm sản xuất Cobb-Douglas mở rộng + AI",
           "Ước lượng TFP, phân rã tăng trưởng (growth accounting) và dự báo GDP 2030 "
           "cho nền kinh tế Việt Nam với các yếu tố mới: số hóa D, năng lực AI, nhân lực số H.")

    year = macro["year"].values
    K, L, D, AI, H = K_SERIES, L_SERIES, D_SERIES, AI_SERIES, H_SERIES
    Y = macro["GDP_trillion_VND"].values.astype(float)

    # --- Câu 1.4.1: TFP A_t (giải ngược) ---
    core = cobb_core(K, L, D, AI, H)
    A = Y / core
    tfp_cagr = (A[-1] / A[0]) ** (1 / (len(A) - 1)) - 1

    st.subheader("Câu 1.4.1 — Ước lượng năng suất nhân tố tổng hợp (TFP) $A_t$")
    c1, c2 = st.columns([2, 1])
    with c1:
        fig = px.line(x=year, y=A, markers=True,
                      labels={"x": "Năm", "y": "TFP $A_t$"},
                      title="TFP $A_t$ giải ngược từ hàm sản xuất, 2020–2025")
        fig.update_traces(line_color="#c0392b")
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        st.metric("TFP CAGR 2020-2025", f"{tfp_cagr*100:.2f}%/năm")
        st.metric("A trung bình", f"{A.mean():.4f}")
        st.caption("TFP đi lên ⇒ tăng trưởng dựa nhiều hơn vào hiệu quả/công nghệ "
                   "(tăng trưởng theo chiều sâu).")
    t141 = pd.DataFrame({"Năm": year, "Y (GDP)": Y.round(1),
                         "Core CD": core.round(2), "TFP A_t": A.round(4)})
    st.dataframe(t141, use_container_width=True, hide_index=True)

    # --- Câu 1.4.2: Dự báo & MAPE ---
    st.subheader("Câu 1.4.2 — Dự báo $\\hat{Y}_t$ với A = trung bình & MAPE")
    A_bar = A.mean()
    Y_hat = A_bar * core
    ape = np.abs((Y - Y_hat) / Y) * 100
    mape = ape.mean()
    st.metric("MAPE (Mean Absolute Percentage Error)", f"{mape:.2f}%")
    dfp = pd.DataFrame({"Năm": year, "Y thực tế": Y, "Ŷ dự báo": Y_hat.round(1),
                        "Sai số %": ape.round(2)})
    fig = go.Figure()
    fig.add_scatter(x=year, y=Y, mode="lines+markers", name="Y thực tế",
                    line_color="#2c3e50")
    fig.add_scatter(x=year, y=Y_hat, mode="lines+markers", name="Ŷ dự báo (A=const)",
                    line=dict(color="#e67e22", dash="dash"))
    fig.update_layout(title=f"Y thực tế vs Ŷ dự báo (MAPE = {mape:.2f}%)",
                      xaxis_title="Năm", yaxis_title="GDP (nghìn tỷ VND)")
    st.plotly_chart(fig, use_container_width=True)
    st.dataframe(dfp, use_container_width=True, hide_index=True)

    # --- Câu 1.4.3: Phân rã tăng trưởng ---
    st.subheader("Câu 1.4.3 — Phân rã tăng trưởng GDP bình quân năm (growth accounting)")
    n = len(year) - 1
    avg_dln = lambda x: (np.log(x[-1]) - np.log(x[0])) / n
    dlnY = avg_dln(Y)
    contrib = {
        "K (vốn vật chất)": ALPHA * avg_dln(K),
        "L (lao động)": BETA * avg_dln(L),
        "D (số hóa)": GAMMA * avg_dln(D),
        "AI (DN số)": DELTA * avg_dln(AI),
        "H (nhân lực số)": THETA * avg_dln(H),
    }
    contrib["TFP (phần dư)"] = dlnY - sum(contrib.values())
    tot = sum(contrib.values())
    t143 = pd.DataFrame({
        "Yếu tố": list(contrib.keys()),
        "Đóng góp (điểm %/năm)": [v * 100 for v in contrib.values()],
        "Tỷ trọng (%)": [v / tot * 100 for v in contrib.values()],
    })
    fig = px.bar(t143, x="Yếu tố", y="Đóng góp (điểm %/năm)",
                 color="Yếu tố", color_discrete_sequence=SEQ,
                 title="Đóng góp từng yếu tố vào tăng trưởng GDP (điểm %/năm)")
    fig.update_layout(showlegend=False)
    st.plotly_chart(fig, use_container_width=True)
    st.dataframe(t143.round(3), use_container_width=True, hide_index=True)
    new_factors = {k: v for k, v in contrib.items()
                   if k in ["D (số hóa)", "AI (DN số)", "H (nhân lực số)"]}
    st.info(f"Trong các yếu tố MỚI, đóng góp lớn nhất: **{max(new_factors, key=new_factors.get)}** "
            f"— tăng trưởng GDP log bình quân ≈ {dlnY*100:.2f}%/năm.")

    # --- Câu 1.4.4: Dự báo 2030 (tương tác) ---
    st.subheader("Câu 1.4.4 — Mô phỏng & dự báo GDP 2030")
    cc = st.columns(3)
    D_2030 = cc[0].slider("D (KTS/GDP %) 2030", 19.5, 40.0, 30.0, 0.5)
    AI_2030 = cc[1].slider("AI (nghìn DN số) 2030", 80.0, 150.0, 100.0, 1.0)
    H_2030 = cc[2].slider("H (LĐ qua ĐT %) 2030", 29.0, 50.0, 35.0, 0.5)
    cc2 = st.columns(3)
    g_KL = cc2[0].slider("Tăng trưởng K, L (%/năm)", 3.0, 10.0, 6.0, 0.5) / 100
    g_TFP = cc2[1].slider("Tăng trưởng TFP (%/năm)", 0.0, 3.0, 1.2, 0.1) / 100
    n_fwd = 2030 - 2025
    K30 = K[-1] * (1 + g_KL) ** n_fwd
    L30 = L[-1] * (1 + g_KL) ** n_fwd
    A30 = A[-1] * (1 + g_TFP) ** n_fwd
    Y30 = A30 * cobb_core(K30, L30, D_2030, AI_2030, H_2030)
    growth30 = (Y30 / Y[-1]) ** (1 / n_fwd) - 1
    m1, m2 = st.columns(2)
    m1.metric("GDP dự báo 2030", f"{Y30:,.0f} nghìn tỷ VND")
    m2.metric("Tăng trưởng BQ 2025→2030", f"{growth30*100:.2f}%/năm")
    fig = go.Figure()
    fig.add_scatter(x=year, y=Y, mode="lines+markers", name="GDP thực tế 2020-2025",
                    line_color="#2c3e50")
    fig.add_scatter(x=[2025, 2030], y=[Y[-1], Y30], mode="lines+markers",
                    name="Dự báo tới 2030", line=dict(color="#27ae60", dash="dash"))
    fig.update_layout(title="GDP thực tế 2020-2025 và dự báo 2030",
                      xaxis_title="Năm", yaxis_title="GDP (nghìn tỷ VND)")
    st.plotly_chart(fig, use_container_width=True)


# =====================================================================
# --- BÀI 2 --- LP PHÂN BỔ NGÂN SÁCH ĐẦU TƯ SỐ (scipy + pulp, shadow price)
# =====================================================================
elif choice == PAGES[2]:
    header("🔹 Bài 2 — LP phân bổ ngân sách đầu tư số",
           "Phân bổ 100 nghìn tỷ VND cho 4 hạng mục (hạ tầng số, R&D/AI, nhân lực số, "
           "doanh nghiệp AI) tối đa hóa tăng GDP kỳ vọng; phân tích giá đối ngẫu.")
    from scipy.optimize import linprog
    import pulp

    profit = np.array([0.85, 1.20, 0.95, 1.35])
    names = ["x1: Hạ tầng số", "x2: R&D công nghệ", "x3: Nhân lực số", "x4: Doanh nghiệp AI"]
    BUDGET = 100

    def build_constraints(B, x3_min=20):
        A_ub = [[1, 1, 1, 1], [-1, 0, 0, 0], [0, -1, 0, 0],
                [0, 0, -1, 0], [0, 0, 0, -1], [0.35, -0.65, 0.35, -0.65]]
        b_ub = [B, -25, -15, -x3_min, -10, 0]
        return A_ub, b_ub

    # --- Câu 2.4.1: scipy.linprog ---
    st.subheader("Câu 2.4.1 — Giải bằng `scipy.optimize.linprog` (HiGHS)")
    c = (-profit).tolist()
    A_ub, b_ub = build_constraints(BUDGET)
    res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=[(0, None)] * 4, method="highs")
    Z_opt, x_opt = -res.fun, res.x
    m1, m2 = st.columns(2)
    m1.metric("Z* tối ưu (lợi ích kỳ vọng)", f"{Z_opt:.3f} nghìn tỷ")
    m2.metric("Vốn sử dụng", f"{x_opt.sum():.1f} / {BUDGET}")
    t241 = pd.DataFrame({"Hạng mục": names, "Phân bổ x* (ngh.tỷ)": x_opt.round(3),
                         "Tỷ trọng %": (x_opt / x_opt.sum() * 100).round(1),
                         "Hệ số lợi ích": profit})
    fig = px.pie(t241, names="Hạng mục", values="Phân bổ x* (ngh.tỷ)",
                 color_discrete_sequence=SEQ, title="Cơ cấu phân bổ tối ưu")
    st.plotly_chart(fig, use_container_width=True)
    st.dataframe(t241, use_container_width=True, hide_index=True)

    # --- Câu 2.4.2: PuLP + dual ---
    st.subheader("Câu 2.4.2 — Giải bằng PuLP & giá đối ngẫu (shadow price)")
    prob = pulp.LpProblem("Budget_Allocation", pulp.LpMaximize)
    x = [pulp.LpVariable(f"x{i+1}", lowBound=0) for i in range(4)]
    prob += pulp.lpSum(profit[i] * x[i] for i in range(4))
    prob += (pulp.lpSum(x) <= BUDGET, "NganSachTong")
    prob += (x[0] >= 25, "San_x1")
    prob += (x[1] >= 15, "San_x2")
    prob += (x[2] >= 20, "San_x3")
    prob += (x[3] >= 10, "San_x4")
    prob += (0.35 * (x[0] + x[2]) <= 0.65 * (x[1] + x[3]), "CanDoiCoCau")
    prob.solve(pulp.PULP_CBC_CMD(msg=0))
    duals = pd.DataFrame([
        {"Ràng buộc": cn, "Shadow price": round(co.pi, 4), "Slack": round(co.slack, 4)}
        for cn, co in prob.constraints.items()])
    sp_budget = prob.constraints["NganSachTong"].pi
    st.dataframe(duals, use_container_width=True, hide_index=True)
    st.success(f"**Shadow price ngân sách tổng = {sp_budget:.4f}** — tăng 1 nghìn tỷ "
               f"ngân sách ⇒ Z* tăng thêm ≈ {sp_budget:.2f} nghìn tỷ. Ràng buộc ngân "
               f"sách đang *chặt* (binding): ngân sách là nguồn lực khan hiếm quyết định Z*.")

    # --- Câu 2.4.3: Độ nhạy Z*(B) ---
    st.subheader("Câu 2.4.3 — Độ nhạy: đường cong $Z^*(B)$")
    B_grid = np.linspace(90, 150, 61)
    Z_grid = []
    for B in B_grid:
        Ab, bb = build_constraints(B)
        r = linprog(c, A_ub=Ab, b_ub=bb, bounds=[(0, None)] * 4, method="highs")
        Z_grid.append(-r.fun)
    fig = px.line(x=B_grid, y=Z_grid, labels={"x": "Ngân sách tổng B (ngh.tỷ)", "y": "Z*"},
                  title="Đường cong giá trị tối ưu Z*(B)")
    fig.update_traces(line_color="#2980b9")
    for B in [100, 120, 140]:
        Ab, bb = build_constraints(B)
        zz = -linprog(c, A_ub=Ab, b_ub=bb, bounds=[(0, None)] * 4, method="highs").fun
        fig.add_scatter(x=[B], y=[zz], mode="markers+text", text=[f"{zz:.1f}"],
                        textposition="top center", marker=dict(color="#e74c3c", size=11),
                        showlegend=False)
    st.plotly_chart(fig, use_container_width=True)

    # --- Câu 2.4.4: x3>=30 ---
    st.subheader("Câu 2.4.4 — Ưu tiên nhân lực số: ràng buộc $x_3 \\geq 30$")
    x3_min = st.slider("Sàn nhân lực số x3", 20, 45, 30, 1)
    An, bn = build_constraints(BUDGET, x3_min=x3_min)
    res2 = linprog(c, A_ub=An, b_ub=bn, bounds=[(0, None)] * 4, method="highs")
    if res2.status == 0:
        Z_new = -res2.fun
        dZ = Z_opt - Z_new
        m1, m2 = st.columns(2)
        m1.metric("Z* mới", f"{Z_new:.3f}", f"{-dZ:.3f}")
        m2.metric("Chi phí cơ hội", f"{dZ:.2f} nghìn tỷ ({dZ/Z_opt*100:.2f}%)")
        cmp = pd.DataFrame({"Hạng mục": names, "x* gốc (x3≥20)": x_opt.round(2),
                            f"x* mới (x3≥{x3_min})": res2.x.round(2)})
        figb = go.Figure()
        figb.add_bar(x=names, y=x_opt, name="Gốc (x3≥20)", marker_color="#3498db")
        figb.add_bar(x=names, y=res2.x, name=f"x3≥{x3_min}", marker_color="#e67e22")
        figb.update_layout(barmode="group", yaxis_title="Ngân sách (ngh.tỷ)",
                           title="Phân bổ: trước vs sau khi ưu tiên nhân lực số")
        st.plotly_chart(figb, use_container_width=True)
        st.dataframe(cmp, use_container_width=True, hide_index=True)
        st.info(f"Bài toán **vẫn khả thi** (tổng sàn = 25+15+{x3_min}+10 = {80+x3_min-30} ≤ 100). "
                f"Buộc rót thêm vào x3 (lợi ích biên 0.95) thay vì x4 (1.35) làm Z* giảm — "
                f"đây là chi phí cơ hội của mục tiêu chính sách.")
    else:
        st.error(f"Bài toán KHÔNG khả thi với x3 ≥ {x3_min}.")


# =====================================================================
# --- BÀI 3 --- CHỈ SỐ ƯU TIÊN NGÀNH (min-max norm, weighted scoring)
# =====================================================================
elif choice == PAGES[3]:
    header("🔹 Bài 3 — Chỉ số ưu tiên ngành (Priority) cho 10 ngành",
           "Xây dựng chỉ số ưu tiên định lượng để xác định ngành nào nên đẩy mạnh "
           "chuyển đổi số & AI trước, tạo hiệu ứng lan tỏa tối đa.")
    df = sectors.copy()
    cols_good = ["growth_rate_2024_pct", "gdp_share_2024_pct", "spillover_coef_0_1",
                 "export_billion_USD", "labor_million", "ai_readiness_0_100"]
    col_bad = "automation_risk_pct"
    crit_labels = ["Tăng trưởng", "GDP-share", "Lan tỏa", "Xuất khẩu",
                   "Việc làm", "AI readiness", "Rủi ro (đảo)"]

    norm_good = lambda x: (x - x.min()) / (x.max() - x.min())
    norm_bad = lambda x: (x.max() - x) / (x.max() - x.min())
    Xg = df[cols_good].apply(norm_good)
    Xb = norm_bad(df[col_bad])
    Xnorm = Xg.copy()
    Xnorm["risk_inv"] = Xb.values
    Xnorm.columns = crit_labels
    X7 = Xnorm.values

    # --- Câu 3.4.1 ---
    st.subheader("Câu 3.4.1 — Ma trận chuẩn hóa min-max 7 tiêu chí (Risk đảo dấu)")
    nv = Xnorm.copy()
    nv.insert(0, "Ngành", df["sector_name_vi"].values)
    st.dataframe(nv.round(3), use_container_width=True, hide_index=True)

    # --- Câu 3.4.2 ---
    st.subheader("Câu 3.4.2 — Priority (trọng số mặc định) & xếp hạng")
    w_raw = np.array([0.15, 0.15, 0.20, 0.15, 0.10, 0.20, 0.15])
    w_default = w_raw / w_raw.sum()
    df["Priority"] = X7 @ w_default
    rank = df[["sector_name_vi", "Priority"]].sort_values("Priority", ascending=False)
    rank = rank.reset_index(drop=True)
    rank.index += 1
    fig = px.bar(rank.sort_values("Priority"), x="Priority", y="sector_name_vi",
                 orientation="h", color="Priority", color_continuous_scale="RdYlGn",
                 title="Điểm Priority theo trọng số mặc định", labels={"sector_name_vi": ""})
    st.plotly_chart(fig, use_container_width=True)
    st.dataframe(rank.rename(columns={"sector_name_vi": "Ngành"}), use_container_width=True)
    st.success(f"**TOP-3 ưu tiên:** {', '.join(rank['sector_name_vi'].head(3).tolist())}")

    # --- Câu 3.4.3: độ nhạy a6 ---
    st.subheader("Câu 3.4.3 — Độ nhạy theo trọng số $a_6$ (AI readiness)")
    a6_values = np.arange(0.05, 0.40 + 1e-9, 0.05)
    idx_a6 = 5
    base_other = np.delete(w_raw, idx_a6)
    base_other = base_other / base_other.sum()
    sectors_vi = df["sector_name_vi"].values
    rank_matrix = np.zeros((len(a6_values), len(sectors_vi)), dtype=int)
    top3_track = []
    for r, a6 in enumerate(a6_values):
        w = np.zeros(7)
        w[idx_a6] = a6
        w_rest = base_other * (1 - a6)
        j = 0
        for k in range(7):
            if k != idx_a6:
                w[k] = w_rest[j]; j += 1
        scores = X7 @ w
        rank_matrix[r] = (-scores).argsort().argsort() + 1
        top3_track.append([sectors_vi[i] for i in np.argsort(-scores)[:3]])
    changed = len({tuple(t) for t in top3_track}) > 1
    fig = px.imshow(rank_matrix.T, aspect="auto", color_continuous_scale="RdYlGn_r",
                    labels=dict(x="Trọng số a6", y="", color="Hạng"),
                    x=[f"{v:.2f}" for v in a6_values], y=list(sectors_vi),
                    text_auto=True, title="Heatmap thứ hạng khi thay đổi a6")
    st.plotly_chart(fig, use_container_width=True)
    st.info(f"TOP-3 **{'CÓ' if changed else 'KHÔNG'} thay đổi** khi a6 biến thiên 0.05→0.40.")

    # --- Câu 3.4.4 ---
    st.subheader("Câu 3.4.4 — So sánh 'Định hướng tăng trưởng' vs 'Bao trùm'")
    w_growth = np.array([0.25, 0.20, 0.05, 0.20, 0.05, 0.20, 0.05])
    w_incl = np.array([0.10, 0.05, 0.25, 0.05, 0.25, 0.05, 0.25])
    w_growth /= w_growth.sum(); w_incl /= w_incl.sum()
    df["P_growth"] = X7 @ w_growth
    df["P_incl"] = X7 @ w_incl
    rg = df.sort_values("P_growth", ascending=False)["sector_name_vi"].head(3).tolist()
    ri = df.sort_values("P_incl", ascending=False)["sector_name_vi"].head(3).tolist()
    comp = pd.DataFrame({"Hạng": [1, 2, 3], "Tăng trưởng": rg, "Bao trùm": ri})
    cf = df[["sector_name_vi", "P_growth", "P_incl"]].sort_values("P_growth")
    fig = go.Figure()
    fig.add_bar(y=cf["sector_name_vi"], x=cf["P_growth"], orientation="h",
                name="Định hướng tăng trưởng", marker_color="#e67e22")
    fig.add_bar(y=cf["sector_name_vi"], x=cf["P_incl"], orientation="h",
                name="Định hướng bao trùm", marker_color="#16a085")
    fig.update_layout(barmode="group", title="Priority theo 2 định hướng chính sách",
                      xaxis_title="Priority")
    st.plotly_chart(fig, use_container_width=True)
    st.dataframe(comp, use_container_width=True, hide_index=True)
    st.caption(f"Chung cả 2 top-3: {sorted(set(rg) & set(ri)) or 'không có'}")


# =====================================================================
# --- BÀI 4 --- LP PHÂN BỔ NGÂN SÁCH SỐ NGÀNH-VÙNG (PuLP + CVXPY)
# =====================================================================
elif choice == PAGES[4]:
    header("🔹 Bài 4 — LP phân bổ ngân sách số theo ngành-vùng",
           "Phân bổ 50 nghìn tỷ VND cho 6 vùng × 4 hạng mục (I, D, AI, H) tối đa hóa "
           "GDP gain với ràng buộc công bằng vùng miền.")
    import pulp

    reg_codes = ["NMM", "RRD", "NCC", "CH", "SE", "MD"]
    items = ["I", "D", "AI", "H"]
    region_vi = {"NMM": "Trung du-MN Bắc", "RRD": "ĐB sông Hồng", "NCC": "BTB-DH Trung",
                 "CH": "Tây Nguyên", "SE": "Đông Nam Bộ", "MD": "ĐB sông Cửu Long"}
    item_vi = {"I": "Hạ tầng (I)", "D": "Số hóa (D)", "AI": "Trí tuệ NT (AI)", "H": "Nhân lực (H)"}
    beta = {("NMM", "I"): 1.15, ("NMM", "D"): 0.85, ("NMM", "AI"): 0.55, ("NMM", "H"): 1.30,
            ("RRD", "I"): 0.95, ("RRD", "D"): 1.25, ("RRD", "AI"): 1.40, ("RRD", "H"): 1.05,
            ("NCC", "I"): 1.05, ("NCC", "D"): 0.95, ("NCC", "AI"): 0.85, ("NCC", "H"): 1.15,
            ("CH", "I"): 1.20, ("CH", "D"): 0.75, ("CH", "AI"): 0.45, ("CH", "H"): 1.35,
            ("SE", "I"): 0.90, ("SE", "D"): 1.30, ("SE", "AI"): 1.55, ("SE", "H"): 1.00,
            ("MD", "I"): 1.10, ("MD", "D"): 0.85, ("MD", "AI"): 0.65, ("MD", "H"): 1.25}
    D0 = {"NMM": 38, "RRD": 78, "NCC": 55, "CH": 32, "SE": 82, "MD": 48}
    gamma_c, BUDGET, R_MIN, R_MAX, H_MIN = 0.002, 50000, 5000, 12000, 12000

    # Chẩn đoán λ khả thi
    D_floor_max = max(D0.values())
    worst = min(D0, key=D0.get)
    D_ceil_worst = D0[worst] + gamma_c * R_MAX
    lam_max = D_ceil_worst / D_floor_max

    st.subheader("⚠️ Chẩn đoán ràng buộc công bằng C5")
    st.warning(f"Với λ=0.70 như đề, mô hình **BẤT KHẢ THI**: cần Dmax ≥ {D_floor_max} "
               f"(vùng SE) nhưng vùng yếu nhất ({worst}) dồn hết {R_MAX} chỉ đạt "
               f"D = {D_ceil_worst:.1f}. **λ khả thi lớn nhất ≈ {lam_max:.4f}**. "
               f"App dùng λ = 0.68 cho mô hình 'có công bằng'.")
    LAM = st.slider("Chọn λ (hệ số công bằng vùng)", 0.30, float(round(lam_max, 2)), 0.68, 0.01)

    @st.cache_data
    def solve_pulp(lam, fair=True):
        m = pulp.LpProblem("VN_Digital_Budget", pulp.LpMaximize)
        x = pulp.LpVariable.dicts("x", (reg_codes, items), lowBound=0)
        m += pulp.lpSum(beta[(r, j)] * x[r][j] for r in reg_codes for j in items)
        m += pulp.lpSum(x[r][j] for r in reg_codes for j in items) <= BUDGET
        for r in reg_codes:
            m += pulp.lpSum(x[r][j] for j in items) >= R_MIN
            m += pulp.lpSum(x[r][j] for j in items) <= R_MAX
        m += pulp.lpSum(x[r]["H"] for r in reg_codes) >= H_MIN
        if fair:
            Mv = pulp.LpVariable("Dmax", lowBound=0)
            for r in reg_codes:
                m += D0[r] + gamma_c * x[r]["D"] <= Mv
                m += D0[r] + gamma_c * x[r]["D"] >= lam * Mv
        m.solve(pulp.PULP_CBC_CMD(msg=0))
        status = pulp.LpStatus[m.status]
        if status != "Optimal":
            return status, None, None
        X = np.array([[x[r][j].value() for j in items] for r in reg_codes])
        return status, pulp.value(m.objective), X

    st.subheader(f"Câu 4.4.1 — Giải bằng PuLP/CBC (có C5, λ={LAM})")
    status_p, Z_pulp, X_pulp = solve_pulp(LAM, True)
    if status_p != "Optimal":
        st.error(f"Trạng thái: {status_p} — thử giảm λ.")
    else:
        st.metric("Z* (PuLP) — GDP gain", f"{Z_pulp:,.1f} tỷ VND")
        X_df = pd.DataFrame(X_pulp, index=[region_vi[r] for r in reg_codes],
                            columns=[item_vi[j] for j in items])
        X_df["TỔNG vùng"] = X_df.sum(axis=1)
        st.dataframe(X_df.round(1), use_container_width=True)

        # --- Câu 4.4.2: CVXPY ---
        st.subheader(f"Câu 4.4.2 — Giải bằng CVXPY & so sánh PuLP")
        try:
            import cvxpy as cp
            B_mat = np.array([[beta[(r, j)] for j in items] for r in reg_codes])
            X = cp.Variable((6, 4), nonneg=True)
            cons = [cp.sum(X) <= BUDGET, cp.sum(X[:, items.index("H")]) >= H_MIN]
            for i in range(6):
                cons += [cp.sum(X[i, :]) >= R_MIN, cp.sum(X[i, :]) <= R_MAX]
            Mv = cp.Variable(nonneg=True)
            d0 = np.array([D0[r] for r in reg_codes])
            for i in range(6):
                cons += [d0[i] + gamma_c * X[i, items.index("D")] <= Mv,
                         d0[i] + gamma_c * X[i, items.index("D")] >= LAM * Mv]
            probc = cp.Problem(cp.Maximize(cp.sum(cp.multiply(B_mat, X))), cons)
            probc.solve(solver=cp.HIGHS)
            same = abs(probc.value - Z_pulp) < 1e-2
            c1, c2 = st.columns(2)
            c1.metric("Z* (CVXPY)", f"{probc.value:,.1f}")
            c2.metric("Z* (PuLP)", f"{Z_pulp:,.1f}")
            st.success(f"Hai phương pháp cho kết quả **{'GIỐNG NHAU' if same else 'KHÁC NHAU'}** "
                       "(cùng LP lồi ⇒ optimum toàn cục duy nhất về Z*).")
        except Exception as e:
            st.info(f"CVXPY không khả dụng ({e}); kết quả PuLP vẫn hợp lệ.")

        # --- Câu 4.4.3: heatmap ---
        st.subheader("Câu 4.4.3 — Heatmap phân bổ tối ưu")
        fig = px.imshow(X_pulp, color_continuous_scale="YlOrRd", aspect="auto",
                        text_auto=".0f",
                        labels=dict(x="Hạng mục", y="Vùng", color="Tỷ VND"),
                        x=[item_vi[j] for j in items], y=[region_vi[r] for r in reg_codes],
                        title=f"Phân bổ tối ưu x[vùng, hạng mục] (λ={LAM})")
        st.plotly_chart(fig, use_container_width=True)
        reg_tot = X_pulp.sum(axis=1)
        top_r = reg_codes[int(reg_tot.argmax())]
        st.info(f"Vùng nhận nhiều ngân sách nhất: **{region_vi[top_r]}** ({reg_tot.max():,.0f} tỷ).")

        # --- Câu 4.4.4: chi phí công bằng ---
        st.subheader("Câu 4.4.4 — Chi phí kinh tế của công bằng vùng miền (bỏ C5)")
        st_nf, Z_nf, X_nf = solve_pulp(LAM, False)
        cost_fair = Z_nf - Z_pulp
        c1, c2, c3 = st.columns(3)
        c1.metric("Z* bỏ C5", f"{Z_nf:,.0f}")
        c2.metric("Z* có C5", f"{Z_pulp:,.0f}")
        c3.metric("Chi phí công bằng", f"{cost_fair:,.0f} tỷ", f"{cost_fair/Z_nf*100:.2f}%")
        cmp = pd.DataFrame({"Vùng": [region_vi[r] for r in reg_codes],
                            "Tổng (có C5)": X_pulp.sum(axis=1).round(0),
                            "Tổng (bỏ C5)": X_nf.sum(axis=1).round(0)})
        fig = go.Figure()
        fig.add_bar(y=cmp["Vùng"], x=cmp["Tổng (có C5)"], orientation="h",
                    name=f"Có công bằng (λ={LAM})", marker_color="#16a085")
        fig.add_bar(y=cmp["Vùng"], x=cmp["Tổng (bỏ C5)"], orientation="h",
                    name="Bỏ công bằng (max hiệu quả)", marker_color="#e74c3c")
        fig.update_layout(barmode="group", title="Phân bổ vùng: có vs bỏ ràng buộc công bằng",
                          xaxis_title="Tổng ngân sách vùng (tỷ VND)")
        st.plotly_chart(fig, use_container_width=True)
        st.dataframe(cmp, use_container_width=True, hide_index=True)


# =====================================================================
# --- BÀI 5 --- MIP LỰA CHỌN 15 DỰ ÁN CHUYỂN ĐỔI SỐ (knapsack + logic)
# =====================================================================
elif choice == PAGES[5]:
    header("🔹 Bài 5 — MIP lựa chọn danh mục dự án chuyển đổi số",
           "Chọn tập dự án tối ưu trong 15 dự án ứng cử (ngân sách 80 nghìn tỷ) với "
           "ràng buộc loại trừ, tiên quyết, bắt buộc và đếm số lượng.")
    from pulp import (LpProblem, LpMaximize, LpVariable, lpSum,
                      PULP_CBC_CMD, LpStatus, value)

    P = list(range(1, 16))
    proj_names = {1: "TT dữ liệu Hòa Lạc", 2: "TT dữ liệu phía Nam", 3: "5G toàn quốc",
                  4: "VNeID 2.0", 5: "Cổng DVC v3", 6: "Y tế số", 7: "Giáo dục số K-12",
                  8: "TT AI + supercomputing", 9: "Sandbox fintech", 10: "Logistics thông minh",
                  11: "Nông nghiệp số ĐBSCL", 12: "Đào tạo 50k kỹ sư AI", 13: "Bán dẫn Bắc Ninh",
                  14: "An ninh mạng SOC", 15: "Open Data quốc gia"}
    C = {1: 12000, 2: 11500, 3: 18000, 4: 4500, 5: 3200, 6: 5800, 7: 6500, 8: 15000,
         9: 2500, 10: 7200, 11: 4800, 12: 8500, 13: 20000, 14: 3800, 15: 1500}
    C1 = {1: 8500, 2: 7500, 3: 12000, 4: 3500, 5: 2500, 6: 4000, 7: 4500, 8: 9000,
          9: 1800, 10: 5000, 11: 3500, 12: 5500, 13: 13000, 14: 2800, 15: 1200}
    B = {1: 21500, 2: 20800, 3: 32500, 4: 9200, 5: 6800, 6: 11400, 7: 12200, 8: 28500,
         9: 5800, 10: 13800, 11: 8500, 12: 16200, 13: 35000, 14: 7500, 15: 3800}
    N_MIN, N_MAX = 7, 11

    def build_solve(cap_budget=80000, op_budget=40000, obj_coeff=None, p1p2="exclusive"):
        coeff = obj_coeff if obj_coeff is not None else B
        m = LpProblem("VN_Project", LpMaximize)
        y = LpVariable.dicts("y", P, cat="Binary")
        m += lpSum(coeff[i] * y[i] for i in P)
        m += lpSum(C[i] * y[i] for i in P) <= cap_budget
        m += lpSum(C1[i] * y[i] for i in P) <= op_budget
        if p1p2 == "exclusive":
            m += y[1] + y[2] <= 1
        elif p1p2 == "both":
            m += y[1] == 1; m += y[2] == 1
        m += y[8] <= y[12]
        m += y[13] <= y[12]
        m += y[4] + y[5] >= 1
        m += y[14] >= 1
        m += lpSum(y[i] for i in P) >= N_MIN
        m += lpSum(y[i] for i in P) <= N_MAX
        m.solve(PULP_CBC_CMD(msg=False))
        status = LpStatus[m.status]
        if status != "Optimal":
            return status, None, []
        sel = [i for i in P if y[i].value() > 0.5]
        return status, value(m.objective), sel

    # --- Câu 5.4.1 ---
    st.subheader("Câu 5.4.1 — Giải bằng PuLP/CBC")
    cap_b = st.slider("Ngân sách đầu tư (nghìn tỷ)", 80, 120, 80, 5) * 1000
    st1, Z1, sel1 = build_solve(cap_budget=cap_b)
    tot_c = sum(C[i] for i in sel1)
    tot_c1 = sum(C1[i] for i in sel1)
    c1, c2, c3 = st.columns(3)
    c1.metric("Z* tổng lợi ích", f"{Z1:,.0f} tỷ")
    c2.metric("Vốn đầu tư dùng", f"{tot_c:,.0f} / {cap_b:,}")
    c3.metric("Vận hành dùng", f"{tot_c1:,.0f} / 40,000")
    det = pd.DataFrame({"Dự án": [f"P{i} — {proj_names[i]}" for i in sel1],
                        "Vốn ĐT": [C[i] for i in sel1], "Vận hành": [C1[i] for i in sel1],
                        "Lợi ích B": [B[i] for i in sel1],
                        "B/C": [round(B[i] / C[i], 2) for i in sel1]})
    fig = px.bar(det.sort_values("Lợi ích B"), x="Lợi ích B", y="Dự án", orientation="h",
                 color="B/C", color_continuous_scale="Viridis",
                 title="Danh mục dự án được chọn")
    st.plotly_chart(fig, use_container_width=True)
    st.dataframe(det, use_container_width=True, hide_index=True)
    npv_cap = Z1 / tot_c
    st.info(f"NPV biên (Z*/vốn đầu tư) = **{npv_cap:.2f}** — mỗi 1 tỷ vốn đầu tư tạo "
            f"{npv_cap:.2f} tỷ lợi ích. Nút thắt thực sự là **ngân sách vận hành** "
            f"(dùng {tot_c1:,.0f}/40,000), không phải vốn đầu tư.")

    # --- Câu 5.4.2: nới ngân sách vận hành ---
    st.subheader("Câu 5.4.2 — Z* theo ngân sách vận hành (nút thắt thực sự)")
    op_grid = list(range(40000, 70001, 5000))
    Z_byop = [build_solve(cap_budget=100000, op_budget=ob)[1] for ob in op_grid]
    fig = px.line(x=op_grid, y=Z_byop, markers=True,
                  labels={"x": "Ngân sách vận hành C1 (tỷ VND)", "y": "Z* (tỷ VND)"},
                  title="Z* tăng khi nới ngân sách vận hành")
    fig.update_traces(line_color="#8e44ad")
    st.plotly_chart(fig, use_container_width=True)
    st.caption("Nới ngân sách đầu tư 80k→100k không đổi Z* (non-binding); "
               "phải nới ngân sách vận hành mới mở thêm dư địa.")

    # --- Câu 5.4.3: redundancy P1 & P2 ---
    st.subheader("Câu 5.4.3 — Quốc hội yêu cầu cả P1 và P2 (redundancy)")
    st3, Z3, sel3 = build_solve(p1p2="both")
    if st3 == "Optimal":
        st.metric("Z* (buộc P1 & P2)", f"{Z3:,.0f} tỷ", f"{Z3-Z1:,.0f}")
        st.warning(f"Nếu GIỮ ràng buộc loại trừ gốc (y1+y2≤1) thì **bất khả thi**. "
                   f"Hiểu redundancy là THAY ràng buộc loại trừ ⇒ khả thi, nhưng Z* giảm "
                   f"{Z1-Z3:,.0f} tỷ ({(Z1-Z3)/Z1*100:.2f}%) — chi phí của yêu cầu dự phòng.")
        st.caption("Dự án chọn: " + ", ".join(f"P{i}" for i in sel3))

    # --- Câu 5.4.4: rủi ro tiến độ ---
    st.subheader("Câu 5.4.4 — Tối đa hóa lợi ích KỲ VỌNG (rủi ro tiến độ)")
    infra, ai_semi, digi_gov = [1, 2, 3], [8, 12, 13], [6, 7, 10, 11]
    p = {i: 0.80 for i in P}
    for i in infra: p[i] = 0.85
    for i in ai_semi: p[i] = 0.65
    for i in digi_gov: p[i] = 0.75
    coeff_exp = {i: p[i] * B[i] for i in P}
    st4, EZ, sel4 = build_solve(obj_coeff=coeff_exp)
    st.metric("E[Z] tối ưu (kỳ vọng)", f"{EZ:,.0f} tỷ")
    added = sorted(set(sel4) - set(sel1))
    removed = sorted(set(sel1) - set(sel4))
    dfp = pd.DataFrame({"Dự án": [f"P{i}" for i in P],
                        "Lợi ích B": [B[i] for i in P],
                        "Kỳ vọng p·B": [round(p[i] * B[i]) for i in P],
                        "Xác suất p": [p[i] for i in P]})
    fig = go.Figure()
    fig.add_bar(x=dfp["Dự án"], y=dfp["Lợi ích B"], name="Lợi ích B", marker_color="#bdc3c7")
    fig.add_bar(x=dfp["Dự án"], y=dfp["Kỳ vọng p·B"], name="Kỳ vọng p·B", marker_color="#2980b9")
    fig.update_layout(barmode="group", title="Lợi ích danh nghĩa vs kỳ vọng (có rủi ro)",
                      yaxis_title="Tỷ VND")
    st.plotly_chart(fig, use_container_width=True)
    st.info(f"So với 5.4.1: thêm {['P'+str(i) for i in added] or 'không'}, "
            f"bớt {['P'+str(i) for i in removed] or 'không'}. Các dự án AI/bán dẫn "
            f"(p=0.65) bị 'phạt' mạnh ⇒ danh mục dịch chuyển sang dự án chắc chắn hơn.")


# =====================================================================
# --- BÀI 6 --- TOPSIS XẾP HẠNG 6 VÙNG (entropy + AHP)
# =====================================================================
elif choice == PAGES[6]:
    header("🔹 Bài 6 — TOPSIS xếp hạng 6 vùng theo mức độ sẵn sàng AI",
           "Áp dụng MCDM-TOPSIS để chọn vùng triển khai trung tâm AI & sandbox dữ liệu "
           "trước; so sánh trọng số chuyên gia, Entropy và AHP.")
    df = regions.copy()
    criteria = ["grdp_per_capita_million_VND", "fdi_registered_billion_USD",
                "digital_index_0_100", "ai_readiness_0_100", "trained_labor_pct",
                "rd_intensity_pct", "internet_penetration_pct", "gini_coef"]
    crit_short = ["GRDP/người", "FDI", "Chỉ số số", "AI readiness", "LĐ đào tạo",
                  "R&D", "Internet", "Gini"]
    is_benefit = np.array([True, True, True, True, True, True, True, False])
    X = df[criteria].values.astype(float)
    names_vi = df["region_name_vi"].values
    n_alt, n_crit = X.shape

    def topsis(X, w, is_benefit):
        R = X / np.sqrt((X ** 2).sum(axis=0))
        V = R * w
        A_star = np.where(is_benefit, V.max(axis=0), V.min(axis=0))
        A_neg = np.where(is_benefit, V.min(axis=0), V.max(axis=0))
        S_star = np.sqrt(((V - A_star) ** 2).sum(axis=1))
        S_neg = np.sqrt(((V - A_neg) ** 2).sum(axis=1))
        return S_neg / (S_star + S_neg)

    # --- Câu 6.4.1 ---
    st.subheader("Câu 6.4.1 — TOPSIS với trọng số chuyên gia")
    w_expert = np.array([0.10, 0.10, 0.15, 0.20, 0.15, 0.15, 0.05, 0.10])
    C_exp = topsis(X, w_expert, is_benefit)
    t = pd.DataFrame({"Vùng": names_vi, "C* (chuyên gia)": C_exp}).sort_values(
        "C* (chuyên gia)", ascending=False).reset_index(drop=True)
    t.index += 1
    fig = px.bar(t.sort_values("C* (chuyên gia)"), x="C* (chuyên gia)", y="Vùng",
                 orientation="h", color="C* (chuyên gia)", color_continuous_scale="Tealgrn",
                 title="Điểm TOPSIS C* (trọng số chuyên gia)")
    st.plotly_chart(fig, use_container_width=True)
    st.dataframe(t.round(4), use_container_width=True)
    st.success(f"**TOP-3:** {', '.join(t['Vùng'].head(3).tolist())}")

    # --- Câu 6.4.2: Entropy ---
    st.subheader("Câu 6.4.2 — Trọng số khách quan bằng Entropy")
    def entropy_weights(X, is_benefit):
        Xc = X.copy().astype(float)
        for j in range(Xc.shape[1]):
            if not is_benefit[j]:
                Xc[:, j] = Xc[:, j].max() + Xc[:, j].min() - Xc[:, j]
        Pm = Xc / Xc.sum(axis=0)
        k = 1.0 / np.log(len(Xc))
        E = -k * np.nansum(Pm * np.log(Pm + 1e-12), axis=0)
        d = 1 - E
        return d / d.sum()
    w_ent = entropy_weights(X, is_benefit)
    C_ent = topsis(X, w_ent, is_benefit)
    wdf = pd.DataFrame({"Tiêu chí": crit_short, "Chuyên gia": w_expert.round(4),
                        "Entropy": w_ent.round(4)})
    fig = go.Figure()
    fig.add_bar(x=crit_short, y=w_expert, name="Chuyên gia", marker_color="#2980b9")
    fig.add_bar(x=crit_short, y=w_ent, name="Entropy", marker_color="#e67e22")
    fig.update_layout(barmode="group", title="So sánh trọng số chuyên gia vs Entropy")
    st.plotly_chart(fig, use_container_width=True)
    cmp = pd.DataFrame({"Vùng": names_vi,
                        "Hạng chuyên gia": pd.Series(C_exp).rank(ascending=False).astype(int).values,
                        "Hạng entropy": pd.Series(C_ent).rank(ascending=False).astype(int).values})
    d2 = (cmp["Hạng chuyên gia"] - cmp["Hạng entropy"]) ** 2
    rho = 1 - 6 * d2.sum() / (n_alt * (n_alt ** 2 - 1))
    st.dataframe(cmp, use_container_width=True, hide_index=True)
    st.info(f"Spearman ρ = **{rho:.4f}** — Entropy dồn trọng số vào tiêu chí phân hóa "
            f"mạnh (FDI, R&D); nhóm dẫn đầu nhìn chung ổn định.")

    # --- Câu 6.4.3: độ nhạy w_AI ---
    st.subheader("Câu 6.4.3 — Độ nhạy theo trọng số $w_{AI}$")
    idx_ai = criteria.index("ai_readiness_0_100")
    base_other = np.delete(w_expert, idx_ai)
    base_other = base_other / base_other.sum()
    w_ai_grid = np.arange(0.10, 0.40 + 1e-9, 0.05)
    track = np.zeros((len(w_ai_grid), n_alt))
    top3 = []
    for r, wai in enumerate(w_ai_grid):
        w = np.zeros(n_crit)
        w[idx_ai] = wai
        j = 0
        for k in range(n_crit):
            if k != idx_ai:
                w[k] = base_other[j] * (1 - wai); j += 1
        Cc = topsis(X, w, is_benefit)
        track[r] = Cc
        top3.append([names_vi[i] for i in np.argsort(-Cc)[:3]])
    stable = len({tuple(x) for x in top3}) == 1
    fig = go.Figure()
    for i in range(n_alt):
        fig.add_scatter(x=w_ai_grid, y=track[:, i], mode="lines+markers", name=names_vi[i])
    fig.update_layout(title="Điểm TOPSIS C* theo trọng số w_AI",
                      xaxis_title="w_AI", yaxis_title="C*")
    st.plotly_chart(fig, use_container_width=True)
    st.info(f"TOP-3 **{'ỔN ĐỊNH' if stable else 'CÓ thay đổi'}** khi w_AI biến thiên "
            f"0.10→0.40. Đông Nam Bộ & ĐB sông Hồng vượt trội ở hầu hết tiêu chí.")

    # --- Câu 6.4.4: AHP ---
    st.subheader("Câu 6.4.4 — AHP đơn giản (eigenvector + CR) so với TOPSIS")
    A_pair = np.outer(w_expert, 1.0 / w_expert)
    eigvals, eigvecs = np.linalg.eig(A_pair)
    imax = np.argmax(eigvals.real)
    lam_max = eigvals.real[imax]
    pv = np.abs(eigvecs[:, imax].real)
    w_ahp = pv / pv.sum()
    CI = (lam_max - n_crit) / (n_crit - 1)
    RI = {6: 1.24, 7: 1.32, 8: 1.41}[n_crit]
    CR = CI / RI
    C_ahp = topsis(X, w_ahp, is_benefit)
    final = pd.DataFrame({"Vùng": names_vi, "C* chuyên gia": C_exp.round(4),
                          "C* entropy": C_ent.round(4), "C* AHP": C_ahp.round(4)})
    c1, c2, c3 = st.columns(3)
    c1.metric("λ_max", f"{lam_max:.3f}")
    c2.metric("CI", f"{CI:.4f}")
    c3.metric("CR", f"{CR:.4f}", "Nhất quán" if CR < 0.1 else "Không nhất quán")
    fig = go.Figure()
    order = final.sort_values("C* chuyên gia", ascending=False)["Vùng"]
    for col, color in [("C* chuyên gia", "#2980b9"), ("C* entropy", "#e67e22"),
                       ("C* AHP", "#16a085")]:
        fig.add_bar(x=order, y=[final.set_index("Vùng").loc[v, col] for v in order],
                    name=col, marker_color=color)
    fig.update_layout(barmode="group", title="So sánh điểm TOPSIS theo 3 bộ trọng số")
    st.plotly_chart(fig, use_container_width=True)
    st.dataframe(final, use_container_width=True, hide_index=True)


# =====================================================================
# --- BÀI 7 --- NSGA-II TỐI ƯU ĐA MỤC TIÊU PARETO (pymoo)
# =====================================================================
elif choice == PAGES[7]:
    header("🔹 Bài 7 — NSGA-II tối ưu đa mục tiêu Pareto",
           "24 biến (6 vùng × 4 hạng mục), 4 mục tiêu xung đột: tăng trưởng (max), "
           "bao trùm/bất bình đẳng (min), phát thải (min), rủi ro ròng (min).")

    reg_codes = ["NMM", "RRD", "NCC", "CH", "SE", "MD"]
    items = ["I", "D", "AI", "H"]
    region_vi = {"NMM": "Trung du-MN Bắc", "RRD": "ĐB sông Hồng", "NCC": "BTB-DH Trung",
                 "CH": "Tây Nguyên", "SE": "Đông Nam Bộ", "MD": "ĐB sông Cửu Long"}
    BETA = np.array([[1.15, 0.85, 0.55, 1.30], [0.95, 1.25, 1.40, 1.05],
                     [1.05, 0.95, 0.85, 1.15], [1.20, 0.75, 0.45, 1.35],
                     [0.90, 1.30, 1.55, 1.00], [1.10, 0.85, 0.65, 1.25]])
    E_FACTOR = np.array([0.42, 0.55, 0.48, 0.32, 0.62, 0.38])
    RHO_RISK = np.array([0.18, 0.45, 0.28, 0.12, 0.52, 0.22])
    SIG_MITIG = np.array([0.32, 0.28, 0.30, 0.35, 0.25, 0.30])
    BUDGET, R_MIN, R_MAX, H_MIN = 50000, 5000, 12000, 12000

    @st.cache_data(show_spinner="Đang chạy NSGA-II (pop=100, n_gen=200)...")
    def run_nsga():
        from pymoo.core.problem import ElementwiseProblem
        from pymoo.algorithms.moo.nsga2 import NSGA2
        from pymoo.optimize import minimize

        class VNProblem(ElementwiseProblem):
            def __init__(self):
                super().__init__(n_var=24, n_obj=4, n_ieq_constr=14,
                                 xl=np.zeros(24), xu=np.ones(24) * R_MAX)

            def _evaluate(self, x, out, *a, **k):
                X = x.reshape(6, 4)
                f1 = -(BETA * X).sum()
                sums = X.sum(axis=1)
                f2 = np.abs(sums - sums.mean()).mean()
                f3 = (E_FACTOR * (X[:, 0] + X[:, 2])).sum()
                f4 = (RHO_RISK * X[:, 2]).sum() - (SIG_MITIG * X[:, 3]).sum()
                out["F"] = [f1, f2, f3, f4]
                g = [X.sum() - BUDGET]
                for r in range(6):
                    g.append(R_MIN - X[r].sum())
                    g.append(X[r].sum() - R_MAX)
                g.append(H_MIN - X[:, 3].sum())
                out["G"] = g

        res = minimize(VNProblem(), NSGA2(pop_size=100), ("n_gen", 200),
                       seed=42, verbose=False)
        return res.F, res.X, float(res.CV.max())

    try:
        F, Xsol, cv = run_nsga()
        GDP, INEQ, EMIS, RISK = -F[:, 0], F[:, 1], F[:, 2], F[:, 3]

        st.subheader("Câu 7.4.1 — Chạy NSGA-II")
        c1, c2 = st.columns(2)
        c1.metric("Số nghiệm Pareto", f"{len(F)}")
        c2.metric("Vi phạm ràng buộc lớn nhất (CV)", f"{cv:.4f}",
                  "Toàn bộ khả thi" if cv < 1e-6 else "")

        # --- Câu 7.4.2: scatter 3D + parallel ---
        st.subheader("Câu 7.4.2 — Trực quan hóa mặt Pareto")
        fig = px.scatter_3d(x=GDP, y=INEQ, z=EMIS, color=RISK,
                            color_continuous_scale="Viridis",
                            labels={"x": "GDP gain (f1)", "y": "Bất BĐ (f2)",
                                    "z": "Phát thải (f3)", "color": "Rủi ro (f4)"},
                            title="Mặt Pareto 3D (màu = rủi ro ròng f4)")
        st.plotly_chart(fig, use_container_width=True)
        Fp = np.column_stack([GDP, INEQ, EMIS, RISK])
        figp = px.parallel_coordinates(
            pd.DataFrame(Fp, columns=["GDP gain", "Bất bình đẳng", "Phát thải", "Rủi ro"]),
            color=GDP, color_continuous_scale="Tealrose",
            title="Parallel coordinates — 4 mục tiêu trên tập Pareto")
        st.plotly_chart(figp, use_container_width=True)

        # --- Câu 7.4.3: TOPSIS chọn nghiệm thỏa hiệp ---
        st.subheader("Câu 7.4.3 — TOPSIS chọn nghiệm thỏa hiệp")
        cc = st.columns(4)
        w1 = cc[0].slider("Tăng trưởng", 0.0, 1.0, 0.40, 0.05)
        w2 = cc[1].slider("Bao trùm", 0.0, 1.0, 0.25, 0.05)
        w3 = cc[2].slider("Môi trường", 0.0, 1.0, 0.20, 0.05)
        w4 = cc[3].slider("An ninh", 0.0, 1.0, 0.15, 0.05)
        w_policy = np.array([w1, w2, w3, w4])
        w_policy = w_policy / w_policy.sum()

        def topsis_pareto(F, w):
            rng = F.max(0) - F.min(0)
            rng[rng == 0] = 1e-12
            norm = (F.max(0) - F) / rng
            V = norm * w
            ideal, anti = V.max(0), V.min(0)
            Sp = np.sqrt(((V - ideal) ** 2).sum(1))
            Sn = np.sqrt(((V - anti) ** 2).sum(1))
            return Sn / (Sp + Sn)

        Cc = topsis_pareto(F, w_policy)
        idx_comp = int(Cc.argmax())
        m = st.columns(4)
        m[0].metric("GDP gain", f"{GDP[idx_comp]:,.0f}")
        m[1].metric("Bất BĐ", f"{INEQ[idx_comp]:.1f}")
        m[2].metric("Phát thải", f"{EMIS[idx_comp]:,.0f}")
        m[3].metric("Rủi ro", f"{RISK[idx_comp]:,.0f}")
        X_comp = Xsol[idx_comp].reshape(6, 4)
        Xc_df = pd.DataFrame(X_comp, index=[region_vi[r] for r in reg_codes], columns=items)
        Xc_df["Tổng"] = Xc_df.sum(axis=1)
        st.dataframe(Xc_df.round(0), use_container_width=True)

        # --- Câu 7.4.4: chi phí cơ hội + radar ---
        st.subheader("Câu 7.4.4 — Chi phí cơ hội: tăng trưởng-max vs thỏa hiệp")
        idx_g = int(GDP.argmax())
        def pct(new, base):
            return (new - base) / abs(base) * 100 if base != 0 else 0
        rows = pd.DataFrame({
            "Mục tiêu": ["GDP gain", "Bất bình đẳng", "Phát thải", "Rủi ro ròng"],
            "Tăng trưởng-max": [GDP[idx_g], INEQ[idx_g], EMIS[idx_g], RISK[idx_g]],
            "Thỏa hiệp": [GDP[idx_comp], INEQ[idx_comp], EMIS[idx_comp], RISK[idx_comp]],
        })
        rows["Δ% so thỏa hiệp"] = [pct(rows.iloc[i, 1], rows.iloc[i, 2]) for i in range(4)]

        def to_good(F):
            rng = F.max(0) - F.min(0); rng[rng == 0] = 1e-12
            return (F.max(0) - F) / rng
        G = to_good(F)
        labels_r = ["Tăng trưởng", "Bao trùm", "Môi trường", "An ninh"]
        figr = go.Figure()
        for idx, lab, col in [(idx_comp, "Thỏa hiệp", "#16a085"),
                              (idx_g, "Tăng trưởng-max", "#e74c3c")]:
            figr.add_scatterpolar(r=G[idx].tolist() + [G[idx][0]],
                                  theta=labels_r + [labels_r[0]], fill="toself",
                                  name=lab, line_color=col)
        figr.update_layout(title="Radar: Thỏa hiệp vs Tăng trưởng-max (xa tâm = tốt hơn)",
                           polar=dict(radialaxis=dict(range=[0, 1])))
        st.plotly_chart(figr, use_container_width=True)
        st.dataframe(rows.round(1), use_container_width=True, hide_index=True)
        st.info("Không có nghiệm thống trị toàn bộ; tăng trưởng tối đa luôn kéo theo cái "
                "giá về bao trùm/môi trường. Nghiệm thỏa hiệp TOPSIS cân bằng 4 mục tiêu.")
    except Exception as e:
        st.error(f"Lỗi khi chạy NSGA-II: {e}")


# =====================================================================
# --- BÀI 8 --- TỐI ƯU ĐỘNG LIÊN THỜI GIAN 2026-2035 (CVXPY)
# =====================================================================
elif choice == PAGES[8]:
    header("🔹 Bài 8 — Tối ưu động phân bổ liên thời gian 2026-2035",
           "Tối đa hóa welfare chiết khấu Σ ρᵗ·ln(Cₜ) với hàm sản xuất Cobb-Douglas và "
           "động học tích lũy vốn K, D, AI, H.")
    try:
        import cvxpy as cp
    except Exception as e:
        st.error(f"CVXPY không khả dụng: {e}")
        st.stop()

    T = 10
    years = np.arange(2026, 2026 + T)
    A_TFP, L_LAB = 1.0, 54.0
    EXPO = np.array([ALPHA, GAMMA, DELTA, THETA, BETA])  # K, D, AI, H, L
    dep_K, dep_D, dep_AI, dep_H = 0.05, 0.12, 0.15, 0.02
    eff_H = 0.8
    K0, D0, AI0, H0 = 27500.0, 20.3, 86.0, 30.0

    @st.cache_data
    def build_solve(rho=0.97, shock_idx=None, shock_pct=0.08, fixed_invest=None):
        K = cp.Variable(T + 1, nonneg=True); D = cp.Variable(T + 1, nonneg=True)
        AI = cp.Variable(T + 1, nonneg=True); H = cp.Variable(T + 1, nonneg=True)
        Y = cp.Variable(T, nonneg=True)
        IK = cp.Variable(T, nonneg=True); ID = cp.Variable(T, nonneg=True)
        IAI = cp.Variable(T, nonneg=True); IH = cp.Variable(T, nonneg=True)
        C = cp.Variable(T, nonneg=True)
        cons = [K[0] == K0, D[0] == D0, AI[0] == AI0, H[0] == H0]
        for t in range(T):
            stack = cp.vstack([K[t], D[t], AI[t], H[t], cp.Constant(L_LAB)])
            prod = A_TFP * cp.geo_mean(stack, EXPO)
            if shock_idx is not None and t == shock_idx:
                cons += [Y[t] <= (1 - shock_pct) * prod]
            else:
                cons += [Y[t] <= prod]
            cons += [C[t] + IK[t] + ID[t] + IAI[t] + IH[t] <= Y[t]]
            if fixed_invest is not None:
                cons += [IK[t] + ID[t] + IAI[t] + IH[t] == fixed_invest[t]]
            cons += [K[t + 1] == (1 - dep_K) * K[t] + IK[t],
                     D[t + 1] == (1 - dep_D) * D[t] + ID[t],
                     AI[t + 1] == (1 - dep_AI) * AI[t] + IAI[t],
                     H[t + 1] == (1 - dep_H) * H[t] + eff_H * IH[t]]
        obj = cp.Maximize(cp.sum([rho ** t * cp.log(C[t]) for t in range(T)]))
        prob = cp.Problem(obj, cons)
        prob.solve(solver=cp.CLARABEL)
        return dict(status=prob.status, welfare=prob.value, K=K.value, D=D.value,
                    AI=AI.value, H=H.value, Y=Y.value, C=C.value,
                    I_total=(IK.value + ID.value + IAI.value + IH.value))

    rho = st.slider("Hệ số chiết khấu ρ", 0.85, 0.99, 0.97, 0.01)
    base = build_solve(rho=rho)

    st.subheader("Câu 8.3.1 — Giải bài toán tối ưu động (CVXPY/CLARABEL)")
    st.metric("Welfare tối ưu W* = Σ ρᵗ·ln(Cₜ)", f"{base['welfare']:.4f}")
    traj = pd.DataFrame({"Năm": years, "Y": base["Y"].round(1), "C": base["C"].round(1),
                         "Đầu tư": base["I_total"].round(1), "K": base["K"][:T].round(0),
                         "D": base["D"][:T].round(1), "AI": base["AI"][:T].round(1),
                         "H": base["H"][:T].round(1)})
    st.dataframe(traj, use_container_width=True, hide_index=True)

    st.subheader("Câu 8.3.2 — Quỹ đạo tối ưu K, D, AI, H, Y, C")
    from plotly.subplots import make_subplots
    series = [("K (vốn vật chất)", base["K"][:T]), ("D (số hóa)", base["D"][:T]),
              ("AI (DN số)", base["AI"][:T]), ("H (nhân lực số)", base["H"][:T]),
              ("Y (sản lượng)", base["Y"]), ("C (tiêu dùng)", base["C"])]
    fig = make_subplots(rows=2, cols=3, subplot_titles=[s[0] for s in series])
    for i, (name, val) in enumerate(series):
        fig.add_scatter(x=years, y=val, mode="lines+markers", name=name,
                        row=i // 3 + 1, col=i % 3 + 1, line_color=SEQ[i % len(SEQ)])
    fig.update_layout(height=560, showlegend=False, title_text="Quỹ đạo tối ưu 2026-2035")
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Câu 8.3.3 — Cú sốc 2028: Y giảm 8% (như bão Yagi)")
    shock_pct = st.slider("Mức sốc giảm Y năm 2028 (%)", 0, 20, 8, 1) / 100
    shock_idx = int(np.where(years == 2028)[0][0])
    shock = build_solve(rho=rho, shock_idx=shock_idx, shock_pct=shock_pct)
    c1, c2 = st.columns(2)
    c1.metric("Welfare sau sốc", f"{shock['welfare']:.4f}",
              f"{shock['welfare']-base['welfare']:.4f}")
    fig = make_subplots(rows=1, cols=2, subplot_titles=["Sản lượng Y", "Tổng đầu tư"])
    fig.add_scatter(x=years, y=base["Y"], mode="lines+markers", name="Kế hoạch gốc",
                    line_color="#27ae60", row=1, col=1)
    fig.add_scatter(x=years, y=shock["Y"], mode="lines+markers", name="Sau sốc 2028",
                    line=dict(color="#c0392b", dash="dash"), row=1, col=1)
    fig.add_scatter(x=years, y=base["I_total"], mode="lines+markers", name="Đầu tư gốc",
                    line_color="#27ae60", row=1, col=2, showlegend=False)
    fig.add_scatter(x=years, y=shock["I_total"], mode="lines+markers", name="Đầu tư sau sốc",
                    line=dict(color="#c0392b", dash="dash"), row=1, col=2, showlegend=False)
    fig.add_vline(x=2028, line_dash="dot", line_color="gray")
    fig.update_layout(height=400, title_text="Phản ứng của mô hình trước cú sốc 2028")
    st.plotly_chart(fig, use_container_width=True)
    st.caption("Sau sốc, mô hình tái phân bổ để bù vốn hao và hội tụ về quỹ đạo tối ưu "
               "(consumption smoothing dàn trải tổn thất).")

    st.subheader("Câu 8.3.4 — Trải đều vs Front-load (cùng tổng ngân sách)")
    TOTAL = base["I_total"].sum()
    even = np.full(T, TOTAL / T)
    fw = np.array([3, 3, 3, 1, 1, 1, 0.5, 0.5, 0.5, 0.5])
    front = TOTAL * fw / fw.sum()
    sol_e = build_solve(rho=rho, fixed_invest=even)
    sol_f = build_solve(rho=rho, fixed_invest=front)
    winner = "FRONT-LOAD" if sol_f["welfare"] > sol_e["welfare"] else "TRẢI ĐỀU"
    c1, c2, c3 = st.columns(3)
    c1.metric("Welfare trải đều", f"{sol_e['welfare']:.4f}")
    c2.metric("Welfare front-load", f"{sol_f['welfare']:.4f}")
    c3.metric("Thắng", winner, f"{abs(sol_f['welfare']-sol_e['welfare']):.4f}")
    fig = make_subplots(rows=1, cols=2, subplot_titles=["Lịch đầu tư", "Quỹ đạo tiêu dùng C"])
    fig.add_bar(x=years, y=even, name="Trải đều", marker_color="#3498db", row=1, col=1)
    fig.add_bar(x=years, y=front, name="Front-load", marker_color="#e67e22", row=1, col=1)
    fig.add_scatter(x=years, y=sol_e["C"], mode="lines+markers", name="C trải đều",
                    line_color="#3498db", row=1, col=2)
    fig.add_scatter(x=years, y=sol_f["C"], mode="lines+markers", name="C front-load",
                    line_color="#e67e22", row=1, col=2)
    fig.update_layout(height=400, title_text=f"{winner} cho welfare cao hơn")
    st.plotly_chart(fig, use_container_width=True)
    st.info("Đầu tư SỚM tạo vốn tích lũy sinh lợi trong nhiều năm (lãi kép); miễn lợi suất "
            "vốn > tốc độ chiết khấu, front-load thường tối ưu hơn — phù hợp định hướng "
            "'đầu tư đi trước một bước' cho hạ tầng số/AI nền tảng.")


# =====================================================================
# --- BÀI 9 --- TÁC ĐỘNG AI TỚI THỊ TRƯỜNG LAO ĐỘNG (LP, NetJob)
# =====================================================================
elif choice == PAGES[9]:
    header("🔹 Bài 9 — Tác động AI tới thị trường lao động Việt Nam",
           "Phân bổ 30 nghìn tỷ cho 8 ngành (x_AI, x_H) tối đa hóa tổng NetJob ròng, "
           "bảo đảm không ngành nào mất việc làm ròng.")
    try:
        import cvxpy as cp
    except Exception as e:
        st.error(f"CVXPY không khả dụng: {e}")
        st.stop()

    N = 8
    sector_vi = ["Nông-Lâm-Thủy sản", "CN chế biến chế tạo", "Xây dựng", "Bán buôn-bán lẻ",
                 "Tài chính-Ngân hàng", "Logistics-Vận tải", "CNTT-Truyền thông",
                 "Giáo dục-Đào tạo"]
    L = np.array([13.20, 11.50, 4.80, 7.80, 0.55, 1.95, 0.62, 2.15])
    risk = np.array([18, 42, 25, 38, 52, 35, 28, 22]) / 100
    a1 = np.array([8.5, 32.5, 12.8, 22.4, 45.8, 28.5, 62.5, 18.5])
    b1 = np.array([45, 28, 35, 32, 22, 30, 20, 55])
    c1c = np.array([5.2, 62.4, 18.5, 48.2, 72.5, 42.8, 32.5, 12.5])
    d1 = np.array([50, 32, 42, 38, 26, 36, 24, 62])
    BUDGET = 30000
    disp_rate = c1c * risk
    net_ai = a1 - disp_rate
    PER_CAP = BUDGET / N * 1.5

    def solve_model(per_cap=None, cap5=False):
        xA = cp.Variable(N, nonneg=True)
        xH = cp.Variable(N, nonneg=True)
        Displaced = cp.multiply(disp_rate, xA)
        RetrainCap = cp.multiply(d1, xH)
        NetJob = cp.multiply(a1, xA) + cp.multiply(b1, xH) - Displaced
        cons = [cp.sum(xA + xH) <= BUDGET, NetJob >= 0, Displaced <= RetrainCap]
        if per_cap is not None:
            cons += [xA <= per_cap, xH <= per_cap]
        if cap5:
            cons += [Displaced <= 0.05 * (L * 1000)]
        prob = cp.Problem(cp.Maximize(cp.sum(NetJob)), cons)
        prob.solve()
        return dict(status=prob.status, Z=prob.value, xA=xA.value, xH=xH.value,
                    net=NetJob.value, disp=Displaced.value)

    # --- Câu 9.4.1 ---
    st.subheader("Câu 9.4.1 — Phân bổ tối ưu (x_AI, x_H) & NetJob ròng")
    sol = solve_model(per_cap=PER_CAP)
    st.metric("Tổng NetJob Z*", f"{sol['Z']:,.0f} job-units")
    rows = []
    for i in range(N):
        rows.append({"Ngành": sector_vi[i], "x_AI": round(sol["xA"][i]),
                     "x_H": round(sol["xH"][i]),
                     "NewJob": round(a1[i] * sol["xA"][i]),
                     "Upgrade": round(b1[i] * sol["xH"][i]),
                     "Displaced": round(disp_rate[i] * sol["xA"][i]),
                     "NetJob": round(sol["net"][i])})
    rdf = pd.DataFrame(rows)
    fig = px.bar(rdf, x="Ngành", y="NetJob", color="NetJob",
                 color_continuous_scale="Greens", title="NetJob ròng theo ngành")
    fig.update_layout(xaxis_tickangle=-25)
    st.plotly_chart(fig, use_container_width=True)
    st.dataframe(rdf, use_container_width=True, hide_index=True)
    st.caption("Mô hình có thêm trần per-ngành để trải đều (LP thuần dồn vào hệ số "
               "sinh việc cao nhất — đúng toán nhưng kém ý nghĩa chính sách).")

    # --- Câu 9.4.2 ---
    st.subheader("Câu 9.4.2 — Ngưỡng đào tạo tối thiểu x_H ngành 2 (CN chế biến chế tạo)")
    i = 1
    st.latex(rf"NetJob_2 = ({a1[i]} - {c1c[i]}\times{risk[i]:.2f})\cdot x_{{AI}} + "
             rf"{b1[i]}\cdot x_H = ({net_ai[i]:+.3f})\cdot x_{{AI}} + {b1[i]}\cdot x_H")
    xAI_max = PER_CAP
    if net_ai[i] >= 0:
        ratio = disp_rate[i] / d1[i]
        st.success(f"Hệ số ròng AI = {net_ai[i]:+.3f} > 0 ⇒ NetJob₂ ≥ 0 với mọi x_AI "
                   f"(kể cả x_H=0). Nhưng ràng buộc năng lực đào tạo lại buộc "
                   f"x_H ≥ {ratio:.4f}·x_AI; khi x_AI tối đa = {xAI_max:,.0f} thì "
                   f"x_H tối thiểu = {ratio*xAI_max:,.1f}.")
        xH_grid = np.linspace(0, PER_CAP, 60)
        netjob2 = net_ai[i] * xAI_max + b1[i] * xH_grid
    else:
        xH_th = -net_ai[i] * xAI_max / b1[i]
        st.warning(f"Hệ số ròng AI = {net_ai[i]:.3f} < 0 ⇒ cần x_H ≥ {xH_th:,.1f}.")
        xH_grid = np.linspace(0, PER_CAP, 60)
        netjob2 = net_ai[i] * xAI_max + b1[i] * xH_grid
    fig = px.area(x=xH_grid, y=netjob2, labels={"x": "x_H ngành 2", "y": "NetJob₂"},
                  title=f"NetJob₂ theo x_H (x_AI = {xAI_max:,.0f})")
    fig.add_hline(y=0, line_dash="dash", line_color="red")
    st.plotly_chart(fig, use_container_width=True)

    # --- Câu 9.4.3: Sankey nhóm dễ tổn thương ---
    st.subheader("Câu 9.4.3 — Luồng lao động nhóm dễ tổn thương (ngành 1, 3, 4)")
    vuln = [0, 2, 3]
    src_labels = [sector_vi[i] for i in vuln]
    dst_labels = ["Việc làm mới (AI)", "Nâng cấp kỹ năng", "Bị thay thế (AI)"]
    labels = src_labels + dst_labels
    ns = len(vuln)
    s_idx, t_idx, vals, lcolors = [], [], [], []
    palette = {"new": "rgba(41,128,185,0.55)", "up": "rgba(22,160,133,0.6)",
               "disp": "rgba(192,57,43,0.6)"}
    for k, i in enumerate(vuln):
        for off, val, c in [(0, a1[i] * sol["xA"][i], palette["new"]),
                            (1, b1[i] * sol["xH"][i], palette["up"]),
                            (2, disp_rate[i] * sol["xA"][i], palette["disp"])]:
            if val > 1e-9:
                s_idx.append(k); t_idx.append(ns + off); vals.append(float(val)); lcolors.append(c)
    fig = go.Figure(go.Sankey(
        node=dict(label=labels, pad=24, thickness=20,
                  color=["#2980b9"] * ns + ["#3498db", "#16a085", "#c0392b"]),
        link=dict(source=s_idx, target=t_idx, value=vals, color=lcolors)))
    fig.update_layout(title_text="Luồng dịch chuyển lao động (job-units)", height=420)
    st.plotly_chart(fig, use_container_width=True)
    st.caption("Nghiệm tối ưu hướng 3 ngành này sang đào tạo (x_AI≈0) ⇒ luồng 'nâng cấp "
               "kỹ năng' chiếm ưu thế, bảo vệ lao động phổ thông dễ tổn thương.")

    # --- Câu 9.4.4 ---
    st.subheader("Câu 9.4.4 — Ràng buộc 'không ngành nào mất quá 5% lao động'")
    sol5 = solve_model(per_cap=PER_CAP, cap5=True)
    if sol5["status"].startswith("optimal"):
        c1, c2 = st.columns(2)
        c1.metric("Z* có C4 (5%)", f"{sol5['Z']:,.0f}", f"{sol5['Z']-sol['Z']:,.0f}")
        c2.metric("Chi phí bảo vệ XH", f"{sol['Z']-sol5['Z']:,.0f} job-units")
        cmp = pd.DataFrame({"Ngành": sector_vi, "Không C4": sol["net"].round(0),
                            "Có C4 (5%)": sol5["net"].round(0)})
        fig = go.Figure()
        fig.add_bar(x=sector_vi, y=sol["net"], name="Không C4", marker_color="#3498db")
        fig.add_bar(x=sector_vi, y=sol5["net"], name="Có C4 (5%)", marker_color="#e67e22")
        fig.update_layout(barmode="group", xaxis_tickangle=-25,
                          title="NetJob theo ngành: tác động ràng buộc bảo vệ 5%")
        st.plotly_chart(fig, use_container_width=True)
        st.dataframe(cmp, use_container_width=True, hide_index=True)
        st.info("Bài toán vẫn khả thi nhưng Z* giảm — minh họa đánh đổi HIỆU QUẢ ↔ "
                "BẢO VỆ XÃ HỘI. Ràng buộc chặt nhất ở ngành c1 cao & lao động nền nhỏ "
                "(Tài chính-Ngân hàng: c1=72.5, L=0.55 triệu).")
    else:
        st.error("Bài toán bất khả thi với ràng buộc 5%.")


# =====================================================================
# --- BÀI 10 --- QUY HOẠCH NGẪU NHIÊN HAI GIAI ĐOẠN (Pyomo)
# =====================================================================
elif choice == PAGES[10]:
    header("🔹 Bài 10 — Quy hoạch ngẫu nhiên hai giai đoạn dưới bất định",
           "Quyết định ngân sách first-stage (here-and-now) trước khi biết kịch bản, "
           "second-stage recourse sau khi kịch bản hiện thực hóa; tính VSS & EVPI.")
    import pyomo.environ as pyo

    JS = ["I", "D", "AI", "H"]
    SS = ["s1", "s2", "s3", "s4"]
    SCN_NAME = {"s1": "Lạc quan", "s2": "Cơ sở", "s3": "Bi quan", "s4": "Khủng hoảng"}
    PROB = {"s1": 0.30, "s2": 0.45, "s3": 0.20, "s4": 0.05}
    BETA_B = {"I": 1.00, "D": 1.10, "AI": 1.25, "H": 0.95}
    BETA_S = {("s1", "I"): 1.25, ("s1", "D"): 1.35, ("s1", "AI"): 1.55, ("s1", "H"): 1.05,
              ("s2", "I"): 1.00, ("s2", "D"): 1.10, ("s2", "AI"): 1.25, ("s2", "H"): 0.95,
              ("s3", "I"): 0.75, ("s3", "D"): 0.85, ("s3", "AI"): 0.90, ("s3", "H"): 1.00,
              ("s4", "I"): 0.40, ("s4", "D"): 0.50, ("s4", "AI"): 0.55, ("s4", "H"): 1.10}
    BUDGET1, RESERVE, COUPLE, SOFT_CAP, PEN1, PEN2 = 65000.0, 15000.0, 0.5, 9000.0, 0.05, 0.85

    @st.cache_resource
    def get_solver():
        for name in ["glpk", "cbc", "appsi_highs"]:
            try:
                s = pyo.SolverFactory(name)
                if s.available(exception_flag=False):
                    return s, name
            except Exception:
                continue
        return None, None
    SOLVER, sname = get_solver()
    if SOLVER is None:
        st.error("Không tìm thấy solver (GLPK/CBC/HiGHS) cho Pyomo.")
        st.stop()
    st.caption(f"Solver Pyomo đang dùng: `{sname}`")

    def build_sp(fix_x=None):
        m = pyo.ConcreteModel()
        m.J = pyo.Set(initialize=JS); m.S = pyo.Set(initialize=SS)
        m.p = pyo.Param(m.S, initialize=PROB)
        m.beta = pyo.Param(m.J, initialize=BETA_B)
        m.beta_s = pyo.Param(m.S, m.J, initialize=BETA_S)
        m.x = pyo.Var(m.J, within=pyo.NonNegativeReals)
        m.y = pyo.Var(m.S, m.J, within=pyo.NonNegativeReals)
        m.u_lo = pyo.Var(m.S, within=pyo.NonNegativeReals)
        m.u_hi = pyo.Var(m.S, within=pyo.NonNegativeReals)
        m.budget1 = pyo.Constraint(expr=sum(m.x[j] for j in m.J) <= BUDGET1)
        if fix_x is not None:
            m.fixx = pyo.ConstraintList()
            for j in JS:
                m.fixx.add(m.x[j] == float(fix_x[j]))
        m.budget2 = pyo.Constraint(m.S, rule=lambda m, s: sum(m.y[s, j] for j in m.J) <= RESERVE)
        m.couple = pyo.Constraint(m.S, rule=lambda m, s: m.y[s, "AI"] <= COUPLE * m.x["H"])
        m.split = pyo.Constraint(m.S, rule=lambda m, s: sum(m.y[s, j] for j in m.J) == m.u_lo[s] + m.u_hi[s])
        m.lo_cap = pyo.Constraint(m.S, rule=lambda m, s: m.u_lo[s] <= SOFT_CAP)
        m.obj = pyo.Objective(rule=lambda m: (
            sum(m.beta[j] * m.x[j] for j in m.J)
            + sum(m.p[s] * (sum(m.beta_s[s, j] * m.y[s, j] for j in m.J)
                            - PEN1 * m.u_lo[s] - PEN2 * m.u_hi[s]) for s in m.S)),
            sense=pyo.maximize)
        return m

    def build_det(beta_scn, fix_x=None):
        m = pyo.ConcreteModel()
        m.J = pyo.Set(initialize=JS)
        m.beta = pyo.Param(m.J, initialize=BETA_B)
        m.beta_s = pyo.Param(m.J, initialize=beta_scn)
        m.x = pyo.Var(m.J, within=pyo.NonNegativeReals)
        m.y = pyo.Var(m.J, within=pyo.NonNegativeReals)
        m.u_lo = pyo.Var(within=pyo.NonNegativeReals)
        m.u_hi = pyo.Var(within=pyo.NonNegativeReals)
        m.budget1 = pyo.Constraint(expr=sum(m.x[j] for j in m.J) <= BUDGET1)
        if fix_x is not None:
            m.fixx = pyo.ConstraintList()
            for j in JS:
                m.fixx.add(m.x[j] == float(fix_x[j]))
        m.budget2 = pyo.Constraint(expr=sum(m.y[j] for j in m.J) <= RESERVE)
        m.couple = pyo.Constraint(expr=m.y["AI"] <= COUPLE * m.x["H"])
        m.split = pyo.Constraint(expr=sum(m.y[j] for j in m.J) == m.u_lo + m.u_hi)
        m.lo_cap = pyo.Constraint(expr=m.u_lo <= SOFT_CAP)
        m.obj = pyo.Objective(rule=lambda m: (
            sum(m.beta[j] * m.x[j] for j in m.J)
            + sum(m.beta_s[j] * m.y[j] for j in m.J) - PEN1 * m.u_lo - PEN2 * m.u_hi),
            sense=pyo.maximize)
        return m

    def solve(m):
        SOLVER.solve(m)
        return pyo.value(m.obj)

    def get_x(m):
        return {j: round(pyo.value(m.x[j]), 2) for j in JS}

    @st.cache_data
    def run_all():
        m_sp = build_sp()
        Z_SP = solve(m_sp)
        x_SP = get_x(m_sp)
        y_tab = pd.DataFrame({s: {j: round(pyo.value(m_sp.y[s, j]), 1) for j in JS}
                              for s in SS}).T
        det_x, det_Z = {}, {}
        for s in SS:
            md = build_det({j: BETA_S[(s, j)] for j in JS})
            det_Z[s] = solve(md); det_x[s] = get_x(md)
        beta_bar = {j: sum(PROB[s] * BETA_S[(s, j)] for s in SS) for j in JS}
        m_ev = build_det(beta_bar)
        Z_EV = solve(m_ev); x_EV = get_x(m_ev)
        EEV = solve(build_sp(fix_x=x_EV))
        WS = sum(PROB[s] * det_Z[s] for s in SS)
        return dict(Z_SP=Z_SP, x_SP=x_SP, y_tab=y_tab, det_x=det_x, det_Z=det_Z,
                    x_EV=x_EV, EEV=EEV, WS=WS, RP=Z_SP, VSS=Z_SP - EEV, EVPI=WS - Z_SP)

    R = run_all()

    st.subheader("Câu 10.5.1 — Lời giải Stochastic Programming (SP)")
    st.metric("Giá trị kỳ vọng RP", f"{R['RP']:,.1f} tỷ VND")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Quyết định first-stage x\\* (tỷ VND):**")
        st.dataframe(pd.Series(R["x_SP"], name="x*_SP").to_frame(), use_container_width=True)
    with c2:
        st.markdown("**Second-stage yˢ theo kịch bản:**")
        yt = R["y_tab"].copy()
        yt.index = [SCN_NAME[s] for s in yt.index]
        st.dataframe(yt, use_container_width=True)

    st.subheader("Câu 10.5.2 — Deterministic từng kịch bản & so sánh EV vs SP")
    cmp = pd.DataFrame({"x*_SP": R["x_SP"], "x_EV": R["x_EV"]})
    cmp["Chênh lệch"] = cmp["x*_SP"] - cmp["x_EV"]
    st.dataframe(cmp, use_container_width=True)
    figd = px.bar(pd.DataFrame(R["det_x"]).T.assign(
        scn=[SCN_NAME[s] for s in SS]).set_index("scn"),
        barmode="group", title="First-stage tối ưu theo từng kịch bản (giải riêng)")
    st.plotly_chart(figd, use_container_width=True)

    st.subheader("Câu 10.5.3 — VSS & EVPI")
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("EEV", f"{R['EEV']:,.0f}")
    c2.metric("RP (SP)", f"{R['RP']:,.0f}")
    c3.metric("WS", f"{R['WS']:,.0f}")
    c4.metric("VSS = RP-EEV", f"{R['VSS']:,.1f}")
    c5.metric("EVPI = WS-RP", f"{R['EVPI']:,.1f}")
    fig = px.bar(x=["EEV", "RP (SP)", "WS"], y=[R["EEV"], R["RP"], R["WS"]],
                 color=["EEV", "RP (SP)", "WS"], color_discrete_sequence=SEQ,
                 labels={"x": "", "y": "Lợi ích kỳ vọng (tỷ VND)"},
                 title=f"EEV ≤ RP ≤ WS — VSS={R['VSS']:,.0f}, EVPI={R['EVPI']:,.0f}")
    fig.update_layout(showlegend=False)
    st.plotly_chart(fig, use_container_width=True)
    st.info(f"**VSS = {R['VSS']:,.1f}** (>0): tư duy ngẫu nhiên (SP) tốt hơn dùng kịch bản "
            f"trung bình rồi áp cứng. **EVPI = {R['EVPI']:,.1f}**: cận trên cho giá trị của "
            f"thông tin/dự báo hoàn hảo về kịch bản tương lai.")


# =====================================================================
# --- BÀI 11 --- Q-LEARNING CHÍNH SÁCH KINH TẾ THÍCH NGHI (MDP)
# =====================================================================
elif choice == PAGES[11]:
    header("🔹 Bài 11 — Q-learning cho chính sách kinh tế thích nghi",
           "Mô hình hóa nền kinh tế VN như MDP 81 trạng thái × 5 hành động; huấn luyện "
           "π* qua Q-learning tabular và so sánh với chính sách rule-based.")
    st.caption("Lưu ý: AI hỗ trợ ra quyết định, KHÔNG thay thế trách nhiệm chính trị-xã hội.")

    ALLOC = {
        0: np.array([0.70, 0.10, 0.10, 0.10]),  # Truyền thống
        1: np.array([0.40, 0.25, 0.15, 0.20]),  # Cân bằng
        2: np.array([0.25, 0.45, 0.15, 0.15]),  # Số hóa nhanh
        3: np.array([0.20, 0.20, 0.45, 0.15]),  # AI dẫn dắt
        4: np.array([0.30, 0.20, 0.10, 0.40]),  # Bao trùm
    }
    W = np.array([0.40, 0.25, 0.20, 0.15])
    BINS = {"growth": [0.04, 0.07], "D": [22.0, 28.0], "AI": [110.0, 160.0], "U": [0.30, 0.45]}
    action_names = {0: "a0 Truyền thống (70%K)", 1: "a1 Cân bằng", 2: "a2 Số hóa nhanh (45%D)",
                    3: "a3 AI dẫn dắt (45%AI)", 4: "a4 Bao trùm (40%H)"}
    level = {0: "low", 1: "medium", 2: "high"}

    def _bin(v, edges):
        return 0 if v < edges[0] else (1 if v < edges[1] else 2)

    class Env:
        T = 10
        def reset(self, init_state=None):
            self.K, self.D, self.AI, self.H = 27500.0, 20.3, 86.0, 30.0
            self.U, self.last_growth = 0.38, 0.06
            self.last_Y = self._prod()
            self.t = 0
            self.state = (np.array(init_state, dtype=int) if init_state is not None
                          else self._disc())
            return self.state
        def _prod(self):
            return (self.K ** 0.33) * (54.0 ** 0.42) * (self.D ** 0.10) * \
                   (self.AI ** 0.08) * (self.H ** 0.07)
        def _disc(self):
            return np.array([_bin(self.last_growth, BINS["growth"]), _bin(self.D, BINS["D"]),
                             _bin(self.AI, BINS["AI"]), _bin(self.U, BINS["U"])], dtype=int)
        def step(self, action):
            a = ALLOC[action]; budget = 1000.0
            self.K += a[0] * budget
            self.D = (1 - 0.12) * self.D + a[1] * budget / 100.0
            self.AI = (1 - 0.15) * self.AI + a[2] * budget / 20.0
            self.H += a[3] * budget / 200.0
            Y = self._prod()
            growth = (Y - self.last_Y) / self.last_Y
            dU = 0.06 * a[2] - 0.10 * a[3]
            self.U = float(np.clip(self.U + dU, 0.05, 0.95))
            reward = W[0] * growth - W[1] * dU - W[2] * (a[2] - 0.5 * a[3]) - W[3] * a[0]
            self.last_growth, self.last_Y = growth, Y
            self.t += 1
            self.state = self._disc()
            return self.state, float(reward), self.t >= self.T

    @st.cache_data(show_spinner="Đang huấn luyện Q-learning (10,000 episodes)...")
    def train_q(alpha=0.1, gamma=0.95, n_eps=10000):
        rng = np.random.default_rng(42)
        env = Env()
        Q = np.zeros((3, 3, 3, 3, 5))
        hist = []
        for ep in range(n_eps):
            s = env.reset()
            eps = max(0.05, 1.0 - ep / 5000.0)
            tot = 0.0
            while True:
                a = int(rng.integers(5)) if rng.random() < eps else int(np.argmax(Q[tuple(s)]))
                s2, r, done = env.step(a)
                Q[tuple(s) + (a,)] += alpha * (r + gamma * Q[tuple(s2)].max() - Q[tuple(s) + (a,)])
                s = s2; tot += r
                if done:
                    break
            hist.append(tot)
        return Q, np.array(hist)

    def run_policy(Q, policy, n=2000):
        rng = np.random.default_rng(0)
        env = Env()
        out = []
        for _ in range(n):
            s = env.reset(); tot = 0.0
            while True:
                if policy == "pi":
                    a = int(np.argmax(Q[tuple(s)]))
                elif policy == "random":
                    a = int(rng.integers(5))
                else:
                    a = policy
                s, r, done = env.step(a)
                tot += r
                if done:
                    break
            out.append(tot)
        return np.array(out)

    Q, hist = train_q()

    st.subheader("Câu 11.3.1-2 — Môi trường & huấn luyện Q-learning")
    c1, c2, c3 = st.columns(3)
    c1.metric("Không gian trạng thái", "3⁴ = 81")
    c2.metric("Reward 100 ep đầu", f"{hist[:100].mean():.4f}")
    c3.metric("Reward 100 ep cuối", f"{hist[-100:].mean():.4f}")
    window = 100
    ma = np.convolve(hist, np.ones(window) / window, mode="valid")
    fig = px.line(x=np.arange(len(ma)), y=ma, labels={"x": "Episode", "y": "Tổng reward"},
                  title="Learning curve Q-learning (moving avg 100 ep)")
    fig.update_traces(line_color="#1f77b4")
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Câu 11.3.3 — Chính sách tối ưu π*(s) tại các trạng thái")
    test_states = {
        "VN 2026 thực tế (med, med, low, med)": (1, 1, 0, 1),
        "Suy thoái (low, low, low, high U)": (0, 0, 0, 2),
        "Bứt phá (high, high, high, low U)": (2, 2, 2, 0),
        "Số hóa cao nhưng thất nghiệp": (1, 2, 1, 2),
        "Tăng trưởng nóng thiếu AI": (2, 0, 0, 1),
    }
    rows = []
    for name, s in test_states.items():
        qv = Q[s]
        best = int(np.argmax(qv))
        rows.append({"Trạng thái": name,
                     "[g, D, AI, U]": f"[{level[s[0]]}, {level[s[1]]}, {level[s[2]]}, {level[s[3]]}]",
                     "π*(s)": action_names[best],
                     "Q max": round(float(qv[best]), 3)})
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    st.subheader("Câu 11.3.4 — So sánh π* với chính sách rule-based")
    pi_star = run_policy(Q, "pi")
    a1r = run_policy(Q, 1)
    a3r = run_policy(Q, 3)
    randr = run_policy(Q, "random")
    cmp = pd.DataFrame({
        "Chính sách": ["π* (Q-learning)", "a1 Cân bằng", "a3 AI dẫn dắt", "Random"],
        "Reward TB": [pi_star.mean(), a1r.mean(), a3r.mean(), randr.mean()],
        "Độ lệch chuẩn": [pi_star.std(), a1r.std(), a3r.std(), randr.std()],
    })
    fig = go.Figure()
    for name, data, c in [("π* (Q-learn)", pi_star, "#2ca02c"), ("a1 Cân bằng", a1r, "#ff7f0e"),
                          ("a3 AI dẫn dắt", a3r, "#d62728"), ("Random", randr, "#7f7f7f")]:
        fig.add_box(y=data, name=name, marker_color=c)
    fig.update_layout(title="Phân phối phần thưởng tích lũy (2000 episodes)",
                      yaxis_title="Tổng reward / episode", showlegend=False)
    st.plotly_chart(fig, use_container_width=True)
    st.dataframe(cmp.round(4), use_container_width=True, hide_index=True)
    st.info("π* học được vượt trội các chính sách cố định và random. Với MDP nhỏ (81 trạng "
            "thái), Q-learning tabular hội tụ tốt; DQN (11.3.5) chỉ phát huy lợi thế khi "
            "không gian trạng thái lớn/liên tục.")


# =====================================================================
# --- BÀI 12 --- ĐỒ ÁN TÍCH HỢP AIDEOM-VN (DASHBOARD 5 KỊCH BẢN)
# =====================================================================
elif choice == PAGES[12]:
    st.title("🚀 Bài 12 — Đồ án tích hợp mô hình tổng thể AIDEOM-VN")
    st.markdown(
        "> **Mục tiêu:** tích hợp các kỹ thuật Bài 1–11 thành hệ thống 6 module (M1–M6) "
        "với dashboard thử nghiệm 5 kịch bản chính sách kinh tế (S1–S5)."
    )
    st.markdown("---")

    # 5 kịch bản: tỷ lệ phân bổ [K, D, AI, H]
    SCENARIOS = {
        "S1. Truyền thống": np.array([0.70, 0.10, 0.10, 0.10]),
        "S2. Số hóa nhanh": np.array([0.25, 0.45, 0.15, 0.15]),
        "S3. AI dẫn dắt": np.array([0.20, 0.20, 0.45, 0.15]),
        "S4. Bao trùm số": np.array([0.30, 0.20, 0.10, 0.40]),
        "S5. Tối ưu cân bằng": np.array([0.40, 0.25, 0.20, 0.15]),
    }
    SCN_DESC = {
        "S1. Truyền thống": "Tập trung vốn vật chất, FDI, hạ tầng truyền thống, xuất khẩu.",
        "S2. Số hóa nhanh": "Tăng đầu tư chính phủ số, doanh nghiệp số, thanh toán số.",
        "S3. AI dẫn dắt": "Ưu tiên AI, dữ liệu lớn, bán dẫn, trung tâm dữ liệu.",
        "S4. Bao trùm số": "Ưu tiên vùng yếu, SME, giáo dục số, nông nghiệp số.",
        "S5. Tối ưu cân bằng": "Cân bằng 4 trụ cột theo trọng số chính sách AIDEOM-VN.",
    }

    scn = st.selectbox("🎛️ Chọn kịch bản chính sách để giả lập", list(SCENARIOS.keys()),
                       index=4)
    alloc = SCENARIOS[scn]
    st.caption(SCN_DESC[scn])

    tab1, tab2, tab3, tab4 = st.tabs(
        ["📈 Tổng quan (M1-M2)", "💰 Phân bổ (M3)", "🎯 So sánh 5 kịch bản (M6)",
         "⚠️ Cảnh báo rủi ro (M4-M5)"])

    # ---------------- TAB 1: M1 Cobb-Douglas + M2 TOPSIS ----------------
    with tab1:
        st.subheader("M1 — Dự báo kinh tế (Cobb-Douglas, kế thừa Bài 1)")
        year = macro["year"].values
        Y = macro["GDP_trillion_VND"].values.astype(float)
        core = cobb_core(K_SERIES, L_SERIES, D_SERIES, AI_SERIES, H_SERIES)
        A = Y / core
        A_bar = A.mean()
        Y_hat = A_bar * core
        mape = (np.abs((Y - Y_hat) / Y) * 100).mean()
        # Dự báo 2030 theo cơ cấu phân bổ kịch bản (đẩy mạnh trụ cột được ưu tiên)
        n_fwd = 5
        K30 = K_SERIES[-1] * (1 + 0.04 + 0.05 * alloc[0]) ** n_fwd
        D30 = D_SERIES[-1] * (1 + 0.06 + 0.30 * alloc[1]) ** n_fwd
        AI30 = AI_SERIES[-1] * (1 + 0.05 + 0.25 * alloc[2]) ** n_fwd
        H30 = H_SERIES[-1] * (1 + 0.02 + 0.20 * alloc[3]) ** n_fwd
        A30 = A[-1] * 1.012 ** n_fwd
        Y30 = A30 * cobb_core(K30, L_SERIES[-1] * 1.04 ** n_fwd, D30, AI30, H30)
        c1, c2, c3 = st.columns(3)
        c1.metric("MAPE (Cobb-Douglas)", f"{mape:.2f}%")
        c2.metric("Ā (TFP trung bình)", f"{A_bar:.4f}")
        c3.metric(f"Y 2030 dự báo ({scn[:2]})", f"{Y30:,.0f} ngh.tỷ")
        n = len(year) - 1
        avg_dln = lambda x: (np.log(x[-1]) - np.log(x[0])) / n
        contrib = {"TFP (A)": avg_dln(Y) - (ALPHA * avg_dln(K_SERIES) + BETA * avg_dln(L_SERIES)
                   + GAMMA * avg_dln(D_SERIES) + DELTA * avg_dln(AI_SERIES) + THETA * avg_dln(H_SERIES)),
                   "Vốn (K)": ALPHA * avg_dln(K_SERIES), "Lao động (L)": BETA * avg_dln(L_SERIES),
                   "Số hóa (D)": GAMMA * avg_dln(D_SERIES), "AI": DELTA * avg_dln(AI_SERIES),
                   "Nhân lực số (H)": THETA * avg_dln(H_SERIES)}
        cdf = pd.DataFrame({"Yếu tố": list(contrib.keys()),
                            "Đóng góp %/năm": [v * 100 for v in contrib.values()]})
        fig = px.bar(cdf, x="Yếu tố", y="Đóng góp %/năm", color="Yếu tố",
                     color_discrete_sequence=SEQ, title="Phân rã đóng góp tăng trưởng 2020-2025")
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("M2 — Đánh giá sẵn sàng số (TOPSIS, kế thừa Bài 6)")
        rdf = regions.copy()
        crit = ["grdp_per_capita_million_VND", "fdi_registered_billion_USD",
                "digital_index_0_100", "ai_readiness_0_100", "trained_labor_pct",
                "rd_intensity_pct", "internet_penetration_pct", "gini_coef"]
        isb = np.array([True] * 7 + [False])
        w = np.array([0.10, 0.10, 0.15, 0.20, 0.15, 0.15, 0.05, 0.10])
        Xr = rdf[crit].values.astype(float)
        Rn = Xr / np.sqrt((Xr ** 2).sum(0))
        V = Rn * w
        As = np.where(isb, V.max(0), V.min(0))
        An = np.where(isb, V.min(0), V.max(0))
        Cstar = np.sqrt(((V - An) ** 2).sum(1)) / (
            np.sqrt(((V - As) ** 2).sum(1)) + np.sqrt(((V - An) ** 2).sum(1)))
        rdf["TOPSIS"] = Cstar
        fig = px.bar(rdf.sort_values("TOPSIS"), x="TOPSIS", y="region_name_vi",
                     orientation="h", color="TOPSIS", color_continuous_scale="Tealgrn",
                     title="Xếp hạng sẵn sàng số 6 vùng", labels={"region_name_vi": ""})
        st.plotly_chart(fig, use_container_width=True)

    # ---------------- TAB 2: M3 phân bổ ngân sách ----------------
    with tab2:
        st.subheader("M3 — Tối ưu phân bổ ngân sách")
        budget = st.number_input("Tổng ngân sách giả lập (nghìn tỷ VND)", 50, 500, 100, 10)
        labels = ["Vốn vật chất (K)", "Hạ tầng số (D)", "Công nghệ AI", "Nhân lực số (H)"]
        amounts = alloc * budget
        c1, c2 = st.columns([1, 1])
        with c1:
            fig = px.pie(names=labels, values=amounts, color_discrete_sequence=SEQ,
                         title=f"Cơ cấu vốn — {scn}", hole=0.4)
            st.plotly_chart(fig, use_container_width=True)
        with c2:
            adf = pd.DataFrame({"Trụ cột": labels, "Tỷ lệ %": (alloc * 100).round(1),
                                "Ngân sách (ngh.tỷ)": amounts.round(1)})
            st.dataframe(adf, use_container_width=True, hide_index=True)
            st.metric("Tổng ngân sách phân bổ", f"{amounts.sum():,.0f} ngh.tỷ")

    # ---------------- TAB 3: so sánh 5 kịch bản (radar + bar) ----------------
    with tab3:
        st.subheader("M6 — So sánh đa mục tiêu 5 kịch bản")
        # Tính các KPI cho từng kịch bản từ dynamics Bài 11 (chạy 10 năm)
        def kpi_scn(alloc):
            K, D, AI, H, U = 27500.0, 20.3, 86.0, 30.0, 0.38
            lastY = (K ** 0.33) * (54.0 ** 0.42) * (D ** 0.10) * (AI ** 0.08) * (H ** 0.07)
            tot_growth = 0.0; tot_emis = 0.0; tot_cyber = 0.0
            for _ in range(10):
                a = alloc; b = 1000.0
                K += a[0] * b
                D = 0.88 * D + a[1] * b / 100.0
                AI = 0.85 * AI + a[2] * b / 20.0
                H += a[3] * b / 200.0
                Y = (K ** 0.33) * (54.0 ** 0.42) * (D ** 0.10) * (AI ** 0.08) * (H ** 0.07)
                tot_growth += (Y - lastY) / lastY
                U = float(np.clip(U + 0.06 * a[2] - 0.10 * a[3], 0.05, 0.95))
                tot_cyber += a[2] - 0.5 * a[3]
                tot_emis += a[0]
                lastY = Y
            # mục tiêu: tăng trưởng (cao tốt), việc làm-bao trùm ~ H (cao tốt),
            #           môi trường ~ 1-emission (cao tốt), an ninh ~ 1-cyber (cao tốt)
            return dict(growth=tot_growth, inclusion=alloc[3] * 10 + (1 - U),
                        green=10 - tot_emis, security=5 - tot_cyber, U_final=U)
        kpis = {name: kpi_scn(a) for name, a in SCENARIOS.items()}

        # Radar đa mục tiêu (chuẩn hóa 0-1)
        dims = ["growth", "inclusion", "green", "security"]
        dim_lbl = ["Tăng trưởng", "Bao trùm", "Xanh", "An ninh"]
        mat = np.array([[kpis[n][d] for d in dims] for n in SCENARIOS])
        matn = (mat - mat.min(0)) / (mat.max(0) - mat.min(0) + 1e-9)
        figr = go.Figure()
        for i, name in enumerate(SCENARIOS):
            highlight = name == scn
            figr.add_scatterpolar(r=matn[i].tolist() + [matn[i][0]],
                                  theta=dim_lbl + [dim_lbl[0]], fill="toself", name=name,
                                  opacity=1.0 if highlight else 0.3,
                                  line=dict(width=4 if highlight else 1.5))
        figr.update_layout(title="Radar đánh đổi đa mục tiêu (kịch bản chọn được tô đậm)",
                           polar=dict(radialaxis=dict(range=[0, 1])))
        st.plotly_chart(figr, use_container_width=True)

        # Bảng KPI tổng hợp
        kdf = pd.DataFrame({
            "Kịch bản": list(SCENARIOS.keys()),
            "Σ tăng trưởng 10 năm": [kpis[n]["growth"] for n in SCENARIOS],
            "Thất nghiệp cuối kỳ": [kpis[n]["U_final"] for n in SCENARIOS],
        })
        figb = px.bar(kdf, x="Kịch bản", y="Σ tăng trưởng 10 năm", color="Kịch bản",
                      color_discrete_sequence=SEQ, title="Tổng tăng trưởng tích lũy 10 năm")
        figb.update_layout(showlegend=False)
        st.plotly_chart(figb, use_container_width=True)
        st.dataframe(kdf.round(3), use_container_width=True, hide_index=True)

    # ---------------- TAB 4: M4 NetJob + M5 cảnh báo rủi ro ----------------
    with tab4:
        st.subheader("M4 — Mô phỏng việc làm ròng NetJob (kế thừa Bài 9)")
        sector_vi = ["Nông-Lâm-TS", "CN chế biến", "Xây dựng", "Bán buôn-lẻ",
                     "Tài chính-NH", "Logistics", "CNTT-TT", "Giáo dục"]
        risk = np.array([18, 42, 25, 38, 52, 35, 28, 22]) / 100
        a1 = np.array([8.5, 32.5, 12.8, 22.4, 45.8, 28.5, 62.5, 18.5])
        b1 = np.array([45, 28, 35, 32, 22, 30, 20, 55])
        c1c = np.array([5.2, 62.4, 18.5, 48.2, 72.5, 42.8, 32.5, 12.5])
        # Phân bổ x_AI, x_H theo tỷ lệ AI và H của kịch bản, ngân sách lao động 30k
        budget_job = 30000
        x_AI_share = alloc[2] / (alloc[2] + alloc[3])
        x_AI = np.full(8, budget_job * x_AI_share / 8)
        x_H = np.full(8, budget_job * (1 - x_AI_share) / 8)
        netjob = a1 * x_AI + b1 * x_H - c1c * risk * x_AI
        ndf = pd.DataFrame({"Ngành": sector_vi, "NetJob": netjob})
        fig = px.bar(ndf, x="Ngành", y="NetJob", color="NetJob",
                     color_continuous_scale="RdYlGn",
                     title=f"NetJob ròng theo ngành — {scn}")
        fig.update_layout(xaxis_tickangle=-25)
        st.plotly_chart(fig, use_container_width=True)
        neg = ndf[ndf["NetJob"] < 0]
        if len(neg):
            st.error(f"⚠️ Cảnh báo: {len(neg)} ngành có NetJob ÂM — "
                     f"{', '.join(neg['Ngành'])}. Cần tăng đầu tư đào tạo lại (H).")
        else:
            st.success("✅ Tất cả ngành đều có NetJob ròng dương dưới kịch bản này.")

        st.subheader("M5 — Đánh giá rủi ro (cyber / môi trường / phụ thuộc)")
        cyber = alloc[2] * 100
        emission = alloc[0] * 100
        depend = (1 - alloc[3]) * 100
        c1, c2, c3 = st.columns(3)
        c1.metric("Rủi ro an ninh dữ liệu", f"{cyber:.0f}/100",
                  "Cao" if cyber > 35 else "Vừa")
        c2.metric("Cường độ phát thải", f"{emission:.0f}/100",
                  "Cao" if emission > 50 else "Thấp")
        c3.metric("Phụ thuộc (thiếu nhân lực)", f"{depend:.0f}/100",
                  "Cao" if depend > 70 else "Vừa")
        risk_df = pd.DataFrame({"Loại rủi ro": ["An ninh dữ liệu (AI cao)",
                                "Phát thải (vốn vật chất cao)", "Phụ thuộc (nhân lực thấp)"],
                                "Điểm": [cyber, emission, depend]})
        fig = px.bar(risk_df, x="Điểm", y="Loại rủi ro", orientation="h", color="Điểm",
                     color_continuous_scale="Reds", title="Bản đồ nhiệt rủi ro kịch bản")
        st.plotly_chart(fig, use_container_width=True)
