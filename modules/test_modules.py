# -*- coding: utf-8 -*-
"""
tests/test_modules.py — Kiểm thử khói (smoke test) cho hệ thống AIDEOM-VN.

Mục tiêu (đáp ứng yêu cầu unit test trong Bài 12):
  1. Mỗi mô-đun bài tập đều import được và có hàm render() gọi được.
  2. Dữ liệu 3 tệp CSV nạp đúng kích thước (shape) như đề bài quy định.
  3. Hàm sản xuất Cobb-Douglas lõi trả về giá trị dương hợp lệ.
  4. Toàn bộ 13 trang (Trang chủ + Bài 1->12) chạy render() không phát sinh
     ngoại lệ, kiểm tra bằng khung Streamlit AppTest.

Chạy:  pytest -q
"""
import importlib
import numpy as np
import pytest

MODULE_NAMES = [
    "home",
    "bai1_cobb_douglas", "bai2_lp_budget", "bai3_priority_sectors",
    "bai4_lp_region_budget", "bai5_mip_project_selection", "bai6_topsis_ai_regions",
    "bai7_nsga2_pareto", "bai8_dynamic_optimization", "bai9_ai_labor_market",
    "bai10_stochastic_programming", "bai11_qlearning_policy",
    "bai12_aideom_vn_integrated",
]

PAGE_LABELS = [
    "🏠 Trang chủ Tổng quan Hệ thống", "🌱 Bài 1 — Cobb-Douglas + AI",
    "💰 Bài 2 — LP ngân sách số", "📊 Bài 3 — Priority 10 ngành",
    "🗺️ Bài 4 — LP ngành-vùng", "🎯 Bài 5 — MIP 15 dự án",
    "🏆 Bài 6 — TOPSIS 6 vùng", "🌐 Bài 7 — NSGA-II Pareto",
    "⏳ Bài 8 — Động 2026-2035", "👷 Bài 9 — Lao động & AI",
    "🎲 Bài 10 — Stochastic SP", "🤖 Bài 11 — Q-learning RL",
    "🧠 Bài 12 — AIDEOM tích hợp",
]


@pytest.mark.parametrize("mod", MODULE_NAMES)
def test_module_has_render(mod):
    """Mỗi mô-đun phải import được và cung cấp hàm render() gọi được."""
    m = importlib.import_module(f"modules.{mod}")
    assert hasattr(m, "render"), f"{mod} thiếu hàm render()"
    assert callable(m.render)


def test_csv_shapes():
    """Kiểm tra kích thước 3 tệp dữ liệu gốc đúng như đề bài."""
    from modules.common import prepared_data
    macro, sectors, regions = prepared_data()
    assert macro.shape[0] == 6        # 2020 -> 2025
    assert sectors.shape[0] == 10     # 10 ngành
    assert regions.shape[0] == 6      # 6 vùng KT-XH
    # cột tên tiếng Việt được gắn thành công
    assert sectors["sector_name_vi"].notna().all()
    assert regions["region_name_vi"].notna().all()


def test_cobb_core_positive():
    """Hàm sản xuất Cobb-Douglas lõi trả về giá trị dương."""
    from modules.common import cobb_core, ALPHA, BETA, GAMMA, DELTA, THETA
    assert abs((ALPHA + BETA + GAMMA + DELTA + THETA) - 1.0) < 1e-9  # CRS
    val = cobb_core(23500, 52.9, 18.3, 73.8, 28.4)
    assert val > 0


def test_policy_qa_complete():
    """Mỗi bài (1->12) phải có đủ nội dung Câu hỏi thảo luận chính sách."""
    from modules.common import POLICY_QA
    assert set(POLICY_QA.keys()) == set(range(1, 13))
    for bai, qs in POLICY_QA.items():
        assert len(qs) >= 3, f"Bài {bai} có quá ít câu hỏi"
        for letter, q, a in qs:
            assert q.strip() and a.strip(), f"Bài {bai}{letter} trống nội dung"


@pytest.mark.parametrize("label", PAGE_LABELS)
def test_page_renders_without_exception(label):
    """Mỗi trang chạy render() qua AppTest mà không phát sinh ngoại lệ."""
    from streamlit.testing.v1 import AppTest
    at = AppTest.from_file("app.py", default_timeout=180).run()
    at.sidebar.radio[0].set_value(label).run()
    assert not at.exception, f"Trang '{label}' lỗi: {at.exception}"
