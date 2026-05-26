# 🇻🇳 AIDEOM-VN — Mô hình Ra quyết định Phát triển Kinh tế Việt Nam trong Kỷ nguyên AI

Ứng dụng web (Streamlit) giải **12 bài toán mô hình ra quyết định** phát triển kinh
tế Việt Nam giai đoạn 2020–2025, từ hàm sản xuất Cobb-Douglas mở rộng đến tối ưu
ngẫu nhiên hai giai đoạn và học tăng cường. Mỗi bài kèm phần **Câu hỏi thảo luận
chính sách** gắn kết quả mô hình với bối cảnh thể chế Việt Nam.

> Sinh viên: **Trần Hoàng Bách** — Viện Quản trị Kinh doanh, Đại học Kinh tế, ĐHQGHN

---

## 🏗️ Kiến trúc mô-đun hóa

Theo đúng yêu cầu Bài 12 ("mỗi module có file `.py` độc lập, có docstring"), dự án
tách thành gói `modules/`; `app.py` chỉ làm nhiệm vụ **điều hướng (router)** và gọi
hàm `render()` của từng mô-đun.

```
aideom_vn/
├── app.py                       # Router Streamlit (điểm vào ứng dụng)
├── requirements.txt
├── README.md
├── vietnam_macro_2020_2025.csv  # 3 tệp dữ liệu gốc
├── vietnam_sectors_2024.csv
├── vietnam_regions_2024.csv
├── modules/
│   ├── __init__.py
│   ├── common.py                # Hàm dùng chung: nạp dữ liệu, hằng số mô hình,
│   │                            #   Cobb-Douglas lõi, POLICY_QA, header, policy_box
│   ├── home.py                  # Trang chủ tổng quan
│   ├── bai1_cobb_douglas.py     # Bài 1  — Cobb-Douglas mở rộng + AI
│   ├── bai2_lp_budget.py        # Bài 2  — LP phân bổ ngân sách số
│   ├── bai3_priority_sectors.py # Bài 3  — Chỉ số ưu tiên 10 ngành
│   ├── bai4_lp_region_budget.py # Bài 4  — LP ngành–vùng (PuLP + CVXPY)
│   ├── bai5_mip_project_selection.py   # Bài 5  — MIP chọn 15 dự án
│   ├── bai6_topsis_ai_regions.py       # Bài 6  — TOPSIS + Entropy + AHP
│   ├── bai7_nsga2_pareto.py            # Bài 7  — NSGA-II đa mục tiêu
│   ├── bai8_dynamic_optimization.py    # Bài 8  — Tối ưu động 2026–2035
│   ├── bai9_ai_labor_market.py         # Bài 9  — Tác động AI tới lao động
│   ├── bai10_stochastic_programming.py # Bài 10 — Quy hoạch ngẫu nhiên (VSS/EVPI)
│   ├── bai11_qlearning_policy.py       # Bài 11 — Q-learning chính sách thích nghi
│   └── bai12_aideom_vn_integrated.py   # Bài 12 — Đồ án tích hợp 6 module
└── tests/
    └── test_modules.py          # Smoke test (pytest)
```

Mỗi mô-đun `baiN_*.py` tuân theo giao diện chung:

```python
from modules.common import (..., header, policy_box, prepared_data, ...)

def render():
    macro, sectors, regions = prepared_data()
    header("🔹 Bài N — ...", "Mục tiêu kinh tế ...")
    # ... lời giải các câu N.x.x ...
    policy_box(N)   # Câu hỏi thảo luận chính sách
```

---

## ⚙️ Cài đặt & chạy

```bash
# 1. Tạo môi trường ảo
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# 2. Cài thư viện
pip install -r requirements.txt

# 3. Chạy ứng dụng
streamlit run app.py
```

Đảm bảo 3 tệp CSV nằm cùng thư mục với `app.py`.

---

## 🧪 Kiểm thử

```bash
pytest -q
```

Bộ kiểm thử xác minh: mỗi mô-đun có `render()`; kích thước 3 tệp CSV đúng (6/10/6
dòng); hàm Cobb-Douglas thỏa lợi suất không đổi theo quy mô; nội dung thảo luận
chính sách đầy đủ Bài 1→12; và cả 13 trang render không phát sinh ngoại lệ.

---

## 📚 Nguồn dữ liệu

NSO/GSO, Bộ Khoa học & Công nghệ (MoST), Bộ TT&TT (MIC), Bộ KH&ĐT (MPI),
World Bank, Global Innovation Index 2025 (WIPO). Số liệu trong CSV được làm tròn
phục vụ giảng dạy; khi viết luận văn cần truy xuất số gốc từ https://www.nso.gov.vn.

## ⚖️ Liêm chính học thuật

Ứng dụng là công cụ **hỗ trợ ra quyết định**, không thay thế trách nhiệm chính trị —
xã hội (xem thảo luận Bài 11 & Bài 12). Mọi bộ trọng số đều là lựa chọn giá trị cần
được thảo luận công khai, không phải sự thật khách quan thuần kỹ thuật.
