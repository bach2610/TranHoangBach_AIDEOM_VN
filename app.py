import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

# ==========================================
# CẤU HÌNH TRANG WEB
# ==========================================
st.set_page_config(page_title="AIDEOM-VN Dashboard", page_icon="📈", layout="wide")
st.title("🚀 Hệ thống Hỗ trợ Ra quyết định AIDEOM-VN")
st.markdown("Đồ án môn học: Mô hình ra quyết định phát triển kinh tế Việt Nam trong kỷ nguyên AI.")

# ==========================================
# HÀM TẢI DỮ LIỆU (Cache để web chạy nhanh hơn)
# ==========================================
@st.cache_data
def load_data():
    try:
        # Bắt buộc phải lùi lề (bấm phím Tab hoặc 4 dấu cách) cho 4 dòng dưới đây:
        macro_df = pd.read_csv('vietnam_macro_2020_2025.csv')
        sectors_df = pd.read_csv('vietnam_sectors_2024.csv')
        regions_df = pd.read_csv('vietnam_regions_2024.csv')
        return macro_df, sectors_df, regions_df
    except FileNotFoundError:
        st.error("⚠️ Không tìm thấy file dữ liệu. Vui lòng kiểm tra lại.")
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

macro_df, sectors_df, regions_df = load_data()

# ==========================================
# SIDEBAR - MENU ĐIỀU KHIỂN (BÊN TRÁI)
# ==========================================
st.sidebar.header("⚙️ Cấu hình Kịch bản")
scenario = st.sidebar.selectbox(
    "Chọn Kịch bản chính sách (2026-2030):",
    (
        "S1. Truyền thống (Ưu tiên vốn vật chất)",
        "S2. Số hóa nhanh (Ưu tiên hạ tầng số)",
        "S3. AI dẫn dắt (Ưu tiên công nghệ lõi)",
        "S4. Bao trùm số (Ưu tiên nhân lực)",
        "S5. Tối ưu cân bằng (AIDEOM-VN)"
    )
)

# Từ điển ánh xạ ngân sách phân bổ (K, D, AI, H) theo từng kịch bản
allocations = {
    "S1": [70, 10, 10, 10],
    "S2": [25, 45, 15, 15],
    "S3": [20, 20, 45, 15],
    "S4": [30, 20, 10, 40],
    "S5": [35, 25, 20, 20] # Giả định kết quả chạy mô hình tối ưu
}
current_alloc = allocations[scenario.split(".")[0]]

st.sidebar.markdown("---")
st.sidebar.info("💡 **Ghi chú:** Dashboard này tích hợp 6 module phân tích từ vĩ mô đến tác động ngành và vùng miền.")

# ==========================================
# GIAO DIỆN CHÍNH - 4 TABS TRỰC QUAN
# ==========================================
if not macro_df.empty:
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Tổng quan (M1+M2)", "💰 Tối ưu Phân bổ (M3)", "⚖️ Kịch bản So sánh", "⚠️ Cảnh báo Rủi ro (M4+M5)"])

    # ------------------------------------------
    # TAB 1: TỔNG QUAN VĨ MÔ VÀ SẴN SÀNG SỐ
    # ------------------------------------------
    with tab1:
        st.header("1. Dự báo Kinh tế Vĩ mô (2020 - 2025)")
        col1, col2 = st.columns(2)
        
        with col1:
            fig_gdp = px.line(macro_df, x='year', y='GDP_growth_pct', markers=True, 
                              title="Tốc độ tăng trưởng GDP (%)", line_shape='spline')
            st.plotly_chart(fig_gdp, use_container_width=True)
            
        with col2:
            fig_dig = px.bar(macro_df, x='year', y='digital_economy_share_GDP_pct', 
                             title="Tỷ trọng Kinh tế số / GDP (%)", color='digital_economy_share_GDP_pct')
            st.plotly_chart(fig_dig, use_container_width=True)
            
        st.header("2. Đánh giá Mức độ Sẵn sàng Số theo Vùng")
        fig_regions = px.scatter(regions_df, x='digital_index_0_100', y='ai_readiness_0_100', 
                                 size='grdp_trillion_VND', color='region_name_en', hover_name='region_name_en',
                                 title="Digital Index vs AI Readiness (Bóng to = GRDP lớn)",
                                 labels={'digital_index_0_100': 'Chỉ số Số hóa (Digital Index)', 'ai_readiness_0_100': 'Độ sẵn sàng AI'})
        st.plotly_chart(fig_regions, use_container_width=True)

    # ------------------------------------------
    # TAB 2: TỐI ƯU PHÂN BỔ NGÂN SÁCH
    # ------------------------------------------
    with tab2:
        st.header(f"Chiến lược phân bổ: {scenario}")
        st.write("Tỷ lệ phân bổ vốn dựa trên kịch bản bạn đang chọn ở thanh menu bên trái.")
        
        labels = ['Vật chất (K)', 'Hạ tầng Số (D)', 'Công nghệ AI (AI)', 'Nhân lực Số (H)']
        
        col_pie, col_bar = st.columns(2)
        with col_pie:
            fig_pie = px.pie(values=current_alloc, names=labels, title="Cơ cấu Phân bổ Ngân sách (%)", hole=0.4)
            fig_pie.update_traces(textposition='inside', textinfo='percent+label')
            st.plotly_chart(fig_pie, use_container_width=True)
            
        with col_bar:
            st.markdown("### 📌 Định hướng chính sách:")
            if "S1" in scenario: st.write("- 🏭 Tập trung duy trì các ngành công nghiệp và xây dựng truyền thống.")
            elif "S2" in scenario: st.write("- 🌐 Ưu tiên phủ sóng 5G, cáp quang và hỗ trợ doanh nghiệp SME lên mây (Cloud).")
            elif "S3" in scenario: st.write("- 🤖 Dồn lực xây dựng Data Center, siêu máy tính và hệ sinh thái Chip bán dẫn.")
            elif "S4" in scenario: st.write("- 🎓 Hướng dòng vốn về vùng khó khăn, tập trung đào tạo lại (retraining) lao động.")
            else: st.write("- ⚖️ Kết hợp hài hòa giữa tăng trưởng, môi trường và công bằng xã hội.")

    # ------------------------------------------
    # TAB 3: SO SÁNH CÁC KỊCH BẢN
    # ------------------------------------------
    with tab3:
        st.header("Đánh giá Đánh đổi (Trade-off) giữa 5 Kịch bản")
        
        # Tạo dữ liệu giả lập đại diện cho kết quả chạy mô hình toán
        compare_data = pd.DataFrame({
            'Kịch bản': ['S1. Truyền thống', 'S2. Số hóa nhanh', 'S3. AI dẫn dắt', 'S4. Bao trùm số', 'S5. Tối ưu cân bằng'],
            'Tăng trưởng GDP (Điểm)': [85, 90, 95, 75, 88],
            'Công bằng Xã hội (Điểm)': [60, 70, 55, 95, 85],
            'Rủi ro Mất việc làm (%)': [15, 25, 35, 10, 18]
        })
        
        fig_radar = go.Figure()
        for i, row in compare_data.iterrows():
            fig_radar.add_trace(go.Scatterpolar(
                r=[row['Tăng trưởng GDP (Điểm)'], row['Công bằng Xã hội (Điểm)'], 100 - row['Rủi ro Mất việc làm (%)']],
                theta=['Tăng trưởng', 'Công bằng Xã hội', 'An toàn Việc làm'],
                fill='toself',
                name=row['Kịch bản']
            ))
        fig_radar.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0, 100])), title="Biểu đồ Radar đa mục tiêu")
        st.plotly_chart(fig_radar, use_container_width=True)

    # ------------------------------------------
    # TAB 4: CẢNH BÁO RỦI RO LAO ĐỘNG & MÔI TRƯỜNG
    # ------------------------------------------
    with tab4:
        st.header("Bản đồ Rủi ro Tự động hóa theo Ngành")
        st.write("Biểu đồ thể hiện nguy cơ mất việc làm trước làn sóng AI tại các ngành kinh tế Việt Nam.")
        
        fig_risk = px.bar(sectors_df.sort_values('automation_risk_pct'), 
                          x='automation_risk_pct', y='sector_name_en', orientation='h',
                          color='automation_risk_pct', color_continuous_scale='Reds',
                          labels={'automation_risk_pct': 'Tỷ lệ rủi ro tự động hóa (%)', 'sector_name_en': 'Ngành Kinh tế'},
                          title="Những ngành dễ bị tổn thương nhất do AI")
        st.plotly_chart(fig_risk, use_container_width=True)
        
        st.warning("🚨 **Khuyến nghị:** Các ngành Tài chính-Ngân hàng, Khai khoáng và Chế biến chế tạo cần được ưu tiên phân bổ ngân sách đào tạo lại (Retraining) do rủi ro > 40%.")

else:
    st.write("Đang chờ tải dữ liệu...")
