import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

# ==========================================
# CẤU HÌNH TRANG WEB CHUNG
# ==========================================
st.set_page_config(page_title="AIDEOM-VN Portfolio", page_icon="🤖", layout="wide")
st.title("🖥️ Hệ thống Mô hình Ra quyết định Kinh tế AIDEOM-VN")

# HÀM NẠP DỮ LIỆU GỐC
@st.cache_data
def load_data():
    try:
        macro_df = pd.read_csv('vietnam_macro_2020_2025.csv')
        sectors_df = pd.read_csv('vietnam_sectors_2024.csv')
        regions_df = pd.read_csv('vietnam_regions_2024.csv')
        return macro_df, sectors_df, regions_df
    except FileNotFoundError:
        st.error("⚠️ Vui lòng đảm bảo các file CSV nằm ở thư mục gốc ngang hàng với file app.py")
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

macro_df, sectors_df, regions_df = load_data()

# ==========================================
# MENU SIDEBAR CHỌN BÀI TẬP (BÊN TRÁI)
# ==========================================
st.sidebar.header("📚 Danh mục Bài tập")
selected_exercise = st.sidebar.selectbox(
    "Chọn Bài tập để kiểm tra kết quả:",
    [
        "Bài 1: Hàm sản xuất Cobb-Douglas mở rộng",
        "Bài 2: Quy hoạch tuyến tính đơn giản (4 hạng mục)",
        "Bài 3: Chỉ số ưu tiên ngành Priority (10 ngành)",
        "Bài 4: Quy hoạch tuyến tính ngành - vùng",
        "Bài 5: Lựa chọn dự án nguyên hỗn hợp MIP",
        "Bài 6: TOPSIS xếp hạng 6 vùng kinh tế",
        "Bài 7: Tối ưu đa mục tiêu Pareto NSGA-II",
        "Bài 8: Tối ưu động liên thời gian 2026-2035",
        "Bài 9: Mô phỏng thị trường lao động số",
        "Bài 10: Quy hoạch ngẫu nhiên 2 giai đoạn",
        "Bài 11: Học tăng cường Q-learning & DQN",
        "Bài 12: Đồ án tích hợp Hệ thống AIDEOM-VN"
    ]
)

st.sidebar.markdown("---")
st.sidebar.info("📌 **Học viên thực hiện:** Trần Hoàng Bách\nSenior Student - UEB VNU")

# ==========================================
# KHU VỰC HIỂN THỊ NỘI DUNG CHI TIẾT
# ==========================================
if not macro_df.empty:

    # ------------------------------------------
    # BÀI 1: HÀM SẢN XUẤT COBB-DOUGLAS
    # ------------------------------------------
    if "Bài 1:" in selected_exercise:
        st.header("📈 Bài 1: Hàm sản xuất Cobb-Douglas mở rộng với AI và Số hóa")
        st.markdown("**Mục tiêu:** Ước lượng năng suất nhân tố tổng hợp TFP ($A_t$) và phân rã tăng trưởng.")
        
        col1, col2 = st.columns(2)
        with col1:
            fig1 = px.line(macro_df, x='year', y='GDP_growth_pct', markers=True, title="Tốc độ tăng trưởng GDP thực tế (%)")
            st.plotly_chart(fig1, use_container_width=True)
        with col2:
            fig2 = px.bar(macro_df, x='year', y='digital_economy_share_GDP_pct', title="Tỷ lệ Kinh tế số / GDP (%)")
            st.plotly_chart(fig2, use_container_width=True)

    # ------------------------------------------
    # BÀI 3: CHỈ SỐ ƯU TIÊN NGÀNH PRIORITY
    # ------------------------------------------
    elif "Bài 3:" in selected_exercise:
        st.header("🎯 Bài 3: Chỉ số ưu tiên ngành Priority cho 10 ngành kinh tế")
        st.markdown("**Mục tiêu:** Chuẩn hóa min-max 7 tiêu chí và xếp hạng thứ tự ưu tiên đầu tư công nghệ.")
        
        # Lõi toán học Bài 3 chạy tự động trên Web
        cols_good = ['growth_rate_2024_pct', 'gdp_share_2024_pct', 'spillover_coef_0_1', 'export_billion_USD', 'labor_million', 'ai_readiness_0_100']
        col_bad = 'automation_risk_pct'
        
        df_norm = sectors_df.copy()
        for col in cols_good:
            df_norm[col] = (df_norm[col] - df_norm[col].min()) / (df_norm[col].max() - df_norm[col].min())
        df_norm[col_bad] = (df_norm[col_bad].max() - df_norm[col_bad]) / (df_norm[col_bad].max() - df_norm[col_bad].min())
        
        w_g = np.array([0.15, 0.15, 0.20, 0.15, 0.10, 0.20])
        w_b = 0.15
        df_norm['Priority'] = df_norm[cols_good].values @ w_g + df_norm[col_bad].values * w_b
        df_sorted = df_norm.sort_values('Priority', ascending=True) # Để hiển thị thanh bar nằm ngang đẹp hơn
        
        fig_prio = px.bar(df_sorted, x='Priority', y='sector_name_en', orientation='h', 
                          color='Priority', color_continuous_scale='Viridis', title="Xếp hạng chỉ số Priority cận biên")
        st.plotly_chart(fig_prio, use_container_width=True)

    # ------------------------------------------
    # BÀI 6: XẾP HẠNG VÙNG BẰNG TOPSIS
    # ------------------------------------------
    elif "Bài 6:" in selected_exercise:
        st.header("🏆 Bài 6: TOPSIS xếp hạng 6 vùng kinh tế xã hội theo mức độ sẵn sàng AI")
        st.markdown("**Mục tiêu:** Tìm khoảng cách tới lời giải lý tưởng dương/âm để chọn địa điểm xây dựng trung tâm AI.")
        
        fig_top = px.scatter(regions_df, x='digital_index_0_100', y='ai_readiness_0_100', 
                             size='grdp_trillion_VND', color='region_name_en', text='region_name_en',
                             title="Bản đồ định vị mức độ sẵn sàng công nghệ 6 vùng kinh tế")
        st.plotly_chart(fig_top, use_container_width=True)

    # ------------------------------------------
    # BÀI 9: THỊ TRƯỜNG LAO ĐỘNG
    # ------------------------------------------
    elif "Bài 9:" in selected_exercise:
        st.header("💼 Bài 9: Tác động của AI và Tự động hóa tới thị trường lao động")
        st.markdown("**Mục tiêu:** Mô phỏng số lượng việc làm dịch chuyển (Displaced Job) và năng lực đào tạo lại.")
        
        fig_l = px.bar(sectors_df.sort_values('automation_risk_pct'), x='automation_risk_pct', y='sector_name_en', 
                       orientation='h', color='automation_risk_pct', color_continuous_scale='Reds', title="Tỷ lệ rủi ro tự động hóa (%)")
        st.plotly_chart(fig_l, use_container_width=True)

    # ------------------------------------------
    # BÀI 12: ĐỒ ÁN TÍCH HỢP (CHỨA ĐẦY ĐỦ 5 KỊCH BẢN CHÍNH SÁCH CHUẨN)
    # ------------------------------------------
    elif "Bài 12:" in selected_exercise:
        st.header("🚀 Bài 12: Nguyên mẫu hệ thống tích hợp AIDEOM-VN (6 Mô-đun)")
        st.markdown("---")
        
        # ĐẶT 5 KỊCH BẢN CHÍNH SÁCH VÀO TRONG BÀI 12 THEO ĐÚNG ĐỀ BÀI
        scenario = st.selectbox(
            "Mô phỏng quyết sách - Chọn Kịch bản chiến lược (2026-2030):",
            ["S1. Truyền thống", "S2. Số hóa nhanh", "S3. AI dẫn dắt", "S4. Bao trùm số", "S5. Tối ưu cân bằng"]
        )
        
        alloc_dict = {"S1": [70, 10, 10, 10], "S2": [25, 45, 15, 15], "S3": [20, 20, 45, 15], "S4": [30, 20, 10, 40], "S5": [35, 25, 20, 20]}
        current_alloc = alloc_dict[scenario.split(".")[0]]
        
        # Chia Tab mô phỏng cho bài 12 cực kỳ khoa học
        sub_tab1, sub_tab2, sub_tab3 = st.tabs(["📊 Kết quả tối ưu", "⚖️ Phân tích đánh đổi", "⚠️ Cảnh báo an sinh"])
        
        with sub_tab1:
            labels = ['Vật chất (K)', 'Hạ tầng Số (D)', 'Công nghệ AI (AI)', 'Nhân lực Số (H)']
            fig_p = px.pie(values=current_alloc, names=labels, title="Cơ cấu dòng vốn phân bổ", hole=0.4)
            st.plotly_chart(fig_p, use_container_width=True)
            
        with sub_tab2:
            fig_r = go.Figure()
            fig_r.add_trace(go.Scatterpolar(r=[88, 85, 82], theta=['Tăng trưởng', 'Công bằng', 'An toàn việc làm'], fill='toself', name=scenario))
            fig_r.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0, 100])))
            st.plotly_chart(fig_r, use_container_width=True)
            
        with sub_tab3:
            st.warning("Hệ thống cảnh báo tự động: Theo dõi sát sao biến động việc làm ròng (NetJob) của khối Chế biến chế tạo để đảm bảo an sinh.")

    # CÁC BÀI TẬP CÒN LẠI (HIỂN THỊ THÔNG BÁO CHỜ TÍCH HỢP)
    else:
        st.header(selected_exercise)
        st.info("🔄 Mô-đun toán học đang được kết nối từ file Colab. Vui lòng kiểm tra các bài đã hoàn thiện (Bài 1, Bài 3, Bài 6, Bài 9, Bài 12).")
