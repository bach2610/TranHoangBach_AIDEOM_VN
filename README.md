[README.md](https://github.com/user-attachments/files/28173420/README.md)
# AIDEOM-VN — Mô hình Ra quyết định Phát triển Kinh tế Việt Nam trong Kỷ nguyên AI

> **Môn học:** Các Mô hình Ra Quyết định  
> **Sinh viên:** Trần Hoàng Bách  
> **Đơn vị:** Viện Quản trị Kinh doanh — Đại học Kinh tế, ĐHQGHN  
> **Dữ liệu:** Việt Nam 2020–2025 (NSO/GSO, MoST, MIC, MPI, World Bank, GII 2025)

---

## Giới thiệu

Repository này là bộ bài tập thực hành hoàn chỉnh cho môn **Các Mô hình Ra Quyết định**, triển khai nguyên mẫu hệ thống **AIDEOM-VN** (_AI-Driven Economic Optimization Model for Vietnam_) gồm 12 bài tập xuyên suốt 4 cấp độ:

| Cấp độ | Bài | Kỹ thuật chính |
|---|---|---|
| Dễ | 1–3 | Cobb-Douglas mở rộng, LP đơn giản, MCDM cơ bản |
| Trung bình | 4–6 | LP đầy đủ (ngành–vùng), MIP, TOPSIS + Entropy |
| Khá khó | 7–9 | Pareto NSGA-II, tối ưu động, mô phỏng lao động |
| Khó | 10–12 | Stochastic LP, Q-learning, Đồ án tích hợp |

Toàn bộ mô hình được trực quan hóa qua **một file Streamlit duy nhất** (`app.py`) — không dùng mock data, thuật toán lấy trực tiếp từ notebook (`AIDEOM_VN_Models_1.ipynb`).

---

## Cấu trúc thư mục

```
aideom_vn/
├── app.py                          # Dashboard Streamlit (Bài 1–12)
├── AIDEOM_VN_Models_1.ipynb        # Notebook giải đầy đủ Bài 1–11
├── requirements.txt                # Thư viện Python
├── README.md                       # Tài liệu này
│
├── vietnam_macro_2020_2025.csv     # Dữ liệu vĩ mô 2020–2025 (19 cột)
├── vietnam_sectors_2024.csv        # 10 ngành kinh tế 2024 (13 cột)
└── vietnam_regions_2024.csv        # 6 vùng kinh tế–xã hội 2024 (14 cột)
```

> **Lưu ý:** Ba file CSV phải nằm **cùng thư mục** với `app.py` khi chạy.

---

## Yêu cầu hệ thống

- **Python:** 3.10 hoặc 3.11 (khuyến nghị)
- **RAM:** tối thiểu 4 GB (NSGA-II & Q-learning chạy trên CPU)
- **GPU:** không bắt buộc (tất cả bài chạy được trên máy tính cá nhân)

---

## Cài đặt

### 1. Clone / tải repository

```bash
git clone https://github.com/<your-username>/aideom-vn.git
cd aideom-vn
```

### 2. Tạo môi trường ảo (khuyến nghị)

```bash
# Tạo venv
python -m venv venv

# Kích hoạt
# Windows (PowerShell):
venv\Scripts\Activate.ps1
# macOS / Linux:
source venv/bin/activate

# Nâng cấp pip
pip install --upgrade pip
```

### 3. Cài đặt thư viện

```bash
pip install -r requirements.txt
```

> **macOS Apple Silicon (M1/M2/M3):** Cài PyTorch riêng theo hướng dẫn tại [pytorch.org](https://pytorch.org) trước khi cài `stable-baselines3`.

### 4. Kiểm tra cài đặt

```bash
# Thư viện tối ưu hóa
python -c "import numpy, pandas, scipy, pulp, cvxpy, pymoo; import pyomo.environ; print('Optimization OK')"

# Dashboard
python -c "import streamlit, plotly; print('Dashboard OK')"

# Học tăng cường (tùy chọn — chỉ cần cho notebook, không cần cho app.py)
python -c "import gymnasium, stable_baselines3, torch; print('RL OK')"
```

---

## Chạy ứng dụng

### Dashboard Streamlit (Bài 1–12)

```bash
streamlit run app.py
```

Trình duyệt sẽ mở tự động tại `http://localhost:8501`.  
Dùng **sidebar** để điều hướng giữa 13 trang (Trang chủ + Bài 1 → Bài 12).

### Notebook (Bài 1–11)

```bash
jupyter lab AIDEOM_VN_Models_1.ipynb
```

Hoặc mở trực tiếp trên **Google Colab** — notebook có sẵn lệnh `pip install` ở đầu mỗi cell.

---

## Mô tả từng bài

### Bài 1 — Hàm sản xuất Cobb-Douglas mở rộng với AI và số hóa
Ước lượng TFP hàng năm từ dữ liệu macro 2020–2025, phân rã tăng trưởng GDP theo 6 yếu tố (K, L, D, AI, H, TFP), dự báo GDP Việt Nam đến 2030 theo kịch bản kinh tế số 30 %.

**Kỹ thuật:** `numpy`, `pandas`, hàm sản xuất $Y = A \cdot K^{\alpha} L^{\beta} D^{\gamma} AI^{\delta} H^{\theta}$, growth accounting.

### Bài 2 — LP phân bổ ngân sách 4 hạng mục đầu tư số
Tối đa hóa GDP kỳ vọng từ 100.000 tỷ VND ngân sách số quốc gia. Phân tích shadow price, sensitivity Z*(B), ảnh hưởng ràng buộc nhân lực.

**Kỹ thuật:** `scipy.optimize.linprog`, `pulp` (CBC), dual values.

### Bài 3 — Chỉ số ưu tiên ngành Priority cho 10 ngành
Chuẩn hóa min-max 7 tiêu chí, tính điểm ưu tiên có trọng số, phân tích độ nhạy trọng số AI Readiness, so sánh định hướng tăng trưởng vs bao trùm.

**Kỹ thuật:** Weighted scoring, min-max normalization, heatmap sensitivity.

### Bài 4 — LP phân bổ ngân sách số theo ngành–vùng
24 biến quyết định (6 vùng × 4 hạng mục), 6 nhóm ràng buộc bao gồm công bằng vùng miền (C5). So sánh PuLP vs CVXPY, đo chi phí kinh tế của công bằng.

**Kỹ thuật:** `pulp` (CBC), `cvxpy` (HiGHS), heatmap phân bổ.

### Bài 5 — MIP lựa chọn danh mục 15 dự án chuyển đổi số
Bài toán knapsack tổng quát với biến nhị phân, ràng buộc loại trừ (P1/P2), tiên quyết (P8 ← P12), bắt buộc (P14), ngân sách đa năm. Phân tích kịch bản nới ngân sách và rủi ro dự án.

**Kỹ thuật:** `pulp` (Binary MIP), CBC solver.

### Bài 6 — TOPSIS xếp hạng 6 vùng theo sẵn sàng AI
TOPSIS 5 bước từ đầu, trọng số chuyên gia vs Entropy khách quan, phân tích độ nhạy w_AI, so sánh với AHP.

**Kỹ thuật:** `numpy`, chuẩn hóa vector, khoảng cách Euclide, Entropy weight, Spearman correlation.

### Bài 7 — Tối ưu đa mục tiêu Pareto (NSGA-II)
4 mục tiêu xung đột: GDP gain, bất bình đẳng vùng, phát thải CO₂, rủi ro an ninh dữ liệu. Trích xuất Pareto front, chọn nghiệm thỏa hiệp bằng TOPSIS, phân tích chi phí cơ hội.

**Kỹ thuật:** `pymoo` (NSGA-II), 24 biến, 14 ràng buộc, scatter 3D, parallel coordinates.

### Bài 8 — Tối ưu động liên thời gian 2026–2035
Tối đa hóa tổng phúc lợi $\sum \rho^t \ln C_t$ qua 10 năm. Phương trình tích lũy vốn K/D/AI/H, TFP nội sinh, phân tích cú sốc 2028 (–8% GDP), so sánh front-load vs even-load.

**Kỹ thuật:** `cvxpy` (SCS/CLARABEL), geo_mean Cobb-Douglas, log-utility.

### Bài 9 — Tác động AI tới thị trường lao động
Tối đa hóa NetJob ròng cho 8 ngành, phân bổ x_AI / x_H trong 30.000 tỷ ngân sách. Tìm ngưỡng đào tạo tối thiểu, Sankey lao động dễ tổn thương, ràng buộc bảo vệ 5 %.

**Kỹ thuật:** `cvxpy` (LP), Plotly Sankey diagram.

### Bài 10 — Quy hoạch ngẫu nhiên hai giai đoạn
4 kịch bản kinh tế toàn cầu (lạc quan/cơ sở/bi quan/khủng hoảng). Quyết định here-and-now (65.000 tỷ) + recourse (15.000 tỷ). Tính VSS và EVPI, so sánh EV vs SP.

**Kỹ thuật:** `pyomo` (ConcreteModel), HiGHS solver, stochastic decomposition.

### Bài 11 — Q-learning cho chính sách kinh tế thích nghi
MDP 81 trạng thái × 5 hành động, epsilon-greedy 10.000 episodes. So sánh π* với 3 chính sách rule-based (a1, a3, random). Phân tích hành động tối ưu theo trạng thái kinh tế.

**Kỹ thuật:** Q-learning tabular (thuần Python / numpy), learning curve Plotly.

### Bài 12 — Đồ án AIDEOM-VN tích hợp
Dashboard 4 tab, selectbox 5 kịch bản chính sách (S1 Truyền thống → S5 Tối ưu cân bằng). Biểu đồ pie phân bổ vốn, radar so sánh 4 mục tiêu, bar NetJob, bảng tổng hợp KPI 2030.

**Kỹ thuật:** `streamlit` multi-tab, `plotly` (pie, radar, bar, Sankey), tích hợp M1–M5.

---

## Dữ liệu

| File | Phạm vi | Cột chính |
|---|---|---|
| `vietnam_macro_2020_2025.csv` | Vĩ mô 2020–2025 (6 năm) | GDP, tăng trưởng 3 khu vực, FDI, xuất nhập khẩu, kinh tế số/GDP, năng suất lao động |
| `vietnam_sectors_2024.csv` | 10 ngành 2024 | Tỷ trọng GDP, tăng trưởng, lao động, xuất khẩu, Digital Index, AI Readiness, rủi ro tự động hóa, R&D |
| `vietnam_regions_2024.csv` | 6 vùng KT–XH 2024 | GRDP, GRDP/người, FDI, Digital Index, AI Readiness, lao động qua đào tạo, Gini, R&D, internet |

**Nguồn:** NSO/GSO 2026, Bộ KH&CN (MoST), Bộ TT&TT (MIC), Bộ KH&ĐT (MPI), World Bank, WIPO GII 2025.

> Số liệu trong CSV được làm tròn phục vụ mục đích giảng dạy. Khi trích dẫn học thuật, vui lòng truy xuất số liệu gốc tại [gso.gov.vn](https://www.gso.gov.vn) và [nso.gov.vn](https://www.nso.gov.vn).

---

## Thư viện chính

| Thư viện | Phiên bản tối thiểu | Dùng cho |
|---|---|---|
| `numpy` | ≥ 1.24 | Tính toán ma trận toàn bộ bài |
| `pandas` | ≥ 2.0 | Đọc CSV, DataFrame |
| `scipy` | ≥ 1.10 | `linprog` (Bài 2), `minimize` (Bài 8 backup) |
| `pulp` | ≥ 2.7 | LP / MIP + CBC solver (Bài 2, 4, 5) |
| `cvxpy` | ≥ 1.4 | Convex opt + geo_mean (Bài 8, 9) |
| `pyomo` | ≥ 6.6 | Two-stage stochastic (Bài 10) |
| `highspy` | ≥ 1.7 | HiGHS solver backend |
| `pymoo` | ≥ 0.6.1 | NSGA-II (Bài 7) |
| `streamlit` | ≥ 1.28 | Dashboard (app.py) |
| `plotly` | ≥ 5.17 | Biểu đồ tương tác |
| `matplotlib` | ≥ 3.7 | Biểu đồ notebook |
| `gymnasium` | ≥ 0.29 | MDP env notebook (Bài 11) |
| `stable-baselines3` | ≥ 2.1 | DQN mở rộng notebook (Bài 11) |
| `torch` | ≥ 2.0 | Backend cho SB3 |

---

## Kết quả mô hình chính

| Bài | Kết quả nổi bật |
|---|---|
| Bài 1 | TFP tăng từ ~1.08 (2020) lên ~1.14 (2025); Số hóa D đóng góp ~14% tăng trưởng; Dự báo GDP 2030 ≈ 19.200 nghìn tỷ VND |
| Bài 2 | Z* = 123 nghìn tỷ; Shadow price ngân sách = 1,35; R&D chiếm toàn bộ ngân sách dôi dư |
| Bài 3 | Top-3 ưu tiên: CNTT-TT, Tài chính-NH, CN Chế biến; Khai khoáng xếp cuối dù năng suất cao |
| Bài 4 | Z* ≈ 63.500 tỷ VND GDP gain; Chi phí công bằng vùng ≈ 3–5% Z* |
| Bài 5 | Chọn 9/15 dự án; P3, P8, P12, P13 là cột sống; P15 (Open Data) bị loại do ràng buộc số lượng |
| Bài 6 | ĐNB + ĐBSH dẫn đầu; Tây Nguyên xếp cuối; Entropy thổi phồng khoảng cách FDI |
| Bài 7 | Pareto 100 nghiệm; Nghiệm thỏa hiệp TOPSIS: GDP ~48.000 tỷ, MAD ~1.800 tỷ |
| Bài 8 | Chiến lược front-load hiệu quả hơn even-load ≈ 4% welfare; H phải đi trước AI |
| Bài 9 | NetJob ròng dương toàn ngành; CN Chế biến cần x_H lớn nhất; NN không nên đầu tư AI |
| Bài 10 | Z* SP = 93.025; VSS > 0 (SP tốt hơn EV); H được ưu tiên trong kịch bản khủng hoảng |
| Bài 11 | π* hội tụ sau ~5.000 episodes; Trạng thái VN 2026 → a4 (Bao trùm); Reward +260% vs random |

---

## Liêm chính học thuật

Dự án này sử dụng công cụ AI hỗ trợ (Claude — Anthropic) trong quá trình phát triển code và viết tài liệu. Toàn bộ phân tích kinh tế, câu trả lời thảo luận chính sách và diễn giải kết quả là sản phẩm tư duy độc lập của sinh viên, có tham chiếu các văn bản pháp lý và tài liệu học thuật được trích dẫn trong đề bài.

---

## Tài liệu tham khảo chính

- Nghị quyết 57-NQ/TW (2024) — Đột phá KH&CN, đổi mới sáng tạo và chuyển đổi số
- QĐ 749/QĐ-TTg (2020) — Chương trình Chuyển đổi số quốc gia
- QĐ 127/QĐ-TTg (2021) — Chiến lược quốc gia về AI đến 2030
- QĐ 411/QĐ-TTg (2022) — Chiến lược phát triển kinh tế số và xã hội số
- Solow, R.M. (1956). *A Contribution to the Theory of Economic Growth*. QJE.
- Deb et al. (2002). *NSGA-II*. IEEE Trans. Evolutionary Computation.
- Birge & Louveaux (2011). *Introduction to Stochastic Programming*. Springer.
- Sutton & Barto (2018). *Reinforcement Learning: An Introduction*. MIT Press.
- World Bank (2024). *Vietnam Economic Update*.
- NSO/GSO (2026). *Niên giám Thống kê Việt Nam*.
