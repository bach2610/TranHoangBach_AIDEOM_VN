# =====================================================================
# AIDEOM-VN — Mô hình Ra quyết định Phát triển Kinh tế Việt Nam
#             trong Kỷ nguyên AI  |  Bộ định tuyến (router) Streamlit
# Sinh viên: Trần Hoàng Bách — Viện Quản trị Kinh doanh, ĐH Kinh tế ĐHQGHN
# Kiến trúc mô-đun hóa: mỗi bài là một file .py độc lập trong gói `modules/`,
# mỗi mô-đun cung cấp hàm render(). app.py chỉ điều hướng (theo mẫu giảng viên).
# Chạy:  streamlit run app.py
# =====================================================================
import streamlit as st

from modules import home
from modules import bai1_cobb_douglas
from modules import bai2_lp_budget
from modules import bai3_priority_sectors
from modules import bai4_lp_region_budget
from modules import bai5_mip_project_selection
from modules import bai6_topsis_ai_regions
from modules import bai7_nsga2_pareto
from modules import bai8_dynamic_optimization
from modules import bai9_ai_labor_market
from modules import bai10_stochastic_programming
from modules import bai11_qlearning_policy
from modules import bai12_aideom_vn_integrated

# ---------------------------------------------------------------------
# CẤU HÌNH TRANG & STYLE (gọi một lần duy nhất ở điểm vào ứng dụng)
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

# ---------------------------------------------------------------------
# SIDEBAR — ĐIỀU HƯỚNG
# ---------------------------------------------------------------------
st.sidebar.title("🇻🇳 AIDEOM-VN")
st.sidebar.caption("Mô hình ra quyết định phát triển kinh tế VN trong kỉ nguyên AI")

PAGES = {
    "🏠 Trang chủ Tổng quan Hệ thống": home,
    "🌱 Bài 1 — Cobb-Douglas + AI": bai1_cobb_douglas,
    "💰 Bài 2 — LP ngân sách số": bai2_lp_budget,
    "📊 Bài 3 — Priority 10 ngành": bai3_priority_sectors,
    "🗺️ Bài 4 — LP ngành-vùng": bai4_lp_region_budget,
    "🎯 Bài 5 — MIP 15 dự án": bai5_mip_project_selection,
    "🏆 Bài 6 — TOPSIS 6 vùng": bai6_topsis_ai_regions,
    "🌐 Bài 7 — NSGA-II Pareto": bai7_nsga2_pareto,
    "⏳ Bài 8 — Động 2026-2035": bai8_dynamic_optimization,
    "👷 Bài 9 — Lao động & AI": bai9_ai_labor_market,
    "🎲 Bài 10 — Stochastic SP": bai10_stochastic_programming,
    "🤖 Bài 11 — Q-learning RL": bai11_qlearning_policy,
    "🧠 Bài 12 — AIDEOM tích hợp": bai12_aideom_vn_integrated,
}

choice = st.sidebar.radio("Điều hướng", list(PAGES.keys()), label_visibility="collapsed")

st.sidebar.markdown("---")
st.sidebar.markdown(
    "**Sinh viên thực hiện**  \n"
    "Trần Hoàng Bách  \n"
    "Viện Quản trị Kinh doanh  \n"
    "Đại học Kinh tế, ĐHQGHN"
)
st.sidebar.caption("Dữ liệu: NSO, MoST, MIC, MPI, WB, GII 2025")

# ---------------------------------------------------------------------
# ĐIỀU HƯỚNG: gọi render() của mô-đun tương ứng
# ---------------------------------------------------------------------
PAGES[choice].render()
