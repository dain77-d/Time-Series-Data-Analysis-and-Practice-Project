"""Team progress report (Word) from the committed results: python docs/make_report.py -> reports/Team06_진행보고_<date>.docx"""
from datetime import date
from pathlib import Path

import pandas as pd
from docx import Document
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Cm, Pt, RGBColor

REPO = Path(__file__).resolve().parents[1]
R = REPO / "results"
OUT = REPO / "reports" / f"Team06_진행보고_{date.today():%Y%m%d}.docx"
FONT = "맑은 고딕"

doc = Document()
sec = doc.sections[0]
sec.page_width, sec.page_height = Cm(21), Cm(29.7)
for side in ("left_margin", "right_margin"):
    setattr(sec, side, Cm(2))
sec.top_margin = sec.bottom_margin = Cm(1.8)
for name in ("Normal", "Heading 1", "Heading 2", "Heading 3", "Title"):
    st = doc.styles[name]
    st.font.name = FONT
    st.element.rPr.rFonts.set(qn("w:eastAsia"), FONT)
doc.styles["Normal"].font.size = Pt(10)


def h(text, level=1):
    doc.add_heading(text, level)


def p(text, bold_prefix=None, size=None):
    para = doc.add_paragraph()
    if bold_prefix:
        para.add_run(bold_prefix).bold = True
    r = para.add_run(text)
    if size:
        r.font.size = Pt(size)
    return para


def bullet(text, bold_prefix=None):
    para = doc.add_paragraph(style="List Bullet")
    if bold_prefix:
        para.add_run(bold_prefix).bold = True
    para.add_run(text)


def shade(cell, hex_color):
    tc = cell._tc.get_or_add_tcPr()
    s = OxmlElement("w:shd")
    s.set(qn("w:val"), "clear")
    s.set(qn("w:color"), "auto")
    s.set(qn("w:fill"), hex_color)
    tc.append(s)


def table(rows, header, widths=None, bold_rows=()):
    t = doc.add_table(rows=1, cols=len(header))
    t.style = "Table Grid"
    t.alignment = WD_TABLE_ALIGNMENT.CENTER
    for i, c in enumerate(header):
        cell = t.rows[0].cells[i]
        cell.text = ""
        cell.paragraphs[0].add_run(str(c)).bold = True
        shade(cell, "D9E2F3")
    for k, row in enumerate(rows):
        cells = t.add_row().cells
        for i, v in enumerate(row):
            cells[i].text = ""
            run = cells[i].paragraphs[0].add_run(str(v))
            run.bold = k in bold_rows
    for row in t.rows:
        for i, cell in enumerate(row.cells):
            for para in cell.paragraphs:
                for run in para.runs:
                    run.font.size = Pt(9)
            if widths:
                cell.width = Cm(widths[i])
    doc.add_paragraph()
    return t


def fig(path, caption, width=17):
    doc.add_picture(str(path), width=Cm(width))
    doc.paragraphs[-1].alignment = WD_ALIGN_PARAGRAPH.CENTER
    c = doc.add_paragraph()
    c.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r = c.add_run(caption)
    r.italic = True
    r.font.size = Pt(9)
    r.font.color.rgb = RGBColor(0x55, 0x55, 0x55)


def rmse_table(metrics_csv, keys):
    m = pd.read_csv(metrics_csv)
    e = m[m.prediction_kind.eq("ensemble")]
    piv = e.pivot_table(index=["model", "input_condition"], columns="horizon", values="RMSE")
    pers = m.groupby("horizon").persistence_RMSE.first()
    rows = [["Persistence", "—"] + [f"{pers[h]:.2f}" for h in (3, 6, 12, 24)]]
    for k in keys:
        rows.append(list(k) + [f"{piv.loc[k, h]:.2f}" for h in (3, 6, 12, 24)])
    return rows


# ---------------------------------------------------------------- title
doc.add_heading("Team06 서울 중구 PM2.5 예측: 진행 보고 (A·B·C)", 0)
p(f"{date.today():%Y-%m-%d} · 2023 검증 결과 · 담당: 모델 C (A·B 포함)", size=9)
p("코드·결과: github.com/dain77-d/Time-Series-Data-Analysis-and-Practice-Project (results/A, B, C)", size=9)

h("1. 한눈에 보기")
bullet("LightGBM(B)이 3·6·12시간에서 가장 정확하고, 24시간은 GRU(C)가 가장 좋다. 둘 다 Persistence 대비 오차를 약 20% 줄였다.", "정확도: ")
bullet("거리 그래프 GCN-GRU는 그래프 없는 GRU보다 오히려 나빴다(3·6시간 유의). 다만 무작위 그래프보다는 나아서, 그래프에 지리 정보는 담겨 있다.", "그래프: ")
bullet("멀리 볼수록(24 h) 모델이 중국 권역 정보를 많이 쓴다(B SHAP 비중: 내몽골 14%, 랴오닝 12%, 산둥 9%). 중국 권역을 넣으면 12–24시간 오차가 일관되게 조금 줄지만, 다중검정 보정 후 유의하지는 않다.", "국외 정보: ")
bullet("모델이 기대는 중국 정보(가리기)는 빼고 다시 학습하면 다른 측정소로 대체된다(DROP 재학습 차이 없음). 대체 불가능한 정보는 수도권 관측뿐이다.", "대체 가능성: ")
bullet("궤적이 거의 지나지 않는 중국 동부·일본은 B·C 모두 추가 가치가 없었다(설계대로).", "음성 대조: ")
bullet("중국 권역 의존도는 추운 계절에 크다(C, 12 h 산둥 4.7% vs 0.4%). '서쪽 경로일 때 크다'는 지지되지 않았다.", "언제: ")

h("2. 설정과 판정 규칙")
table([["데이터", "Dataset_11_07_regions_v5 (팀 11-07 규약 + 중국 동부·일본 동해 쪽 대조 지역)"],
       ["분할", "학습 2021–22 · 설정 선택 2023 · 최종 평가 2024(아직 열지 않음, --final로 한 번)"],
       ["목표", "서울 중구(KR:111121) PM2.5, 3·6·12·24시간 뒤, 3시간 간격 평가 격자의 고정 행"],
       ["조건", "D67(국내 67) · DN(국내 전국 514) · F(국내+국외 600) · ADD_r 8개 · DROP_r 15개 · CTRL 3개"],
       ["채점", "팀 공통 score.py: RMSE, DM 검정(HAC), 7일 블록 bootstrap 95% CI, Holm 보정"],
       ["개선 판정", "CI 상한 < 0 그리고 Holm p < 0.05 (둘 다 만족해야 '유의')"],
       ["시드", "11·22·33 앙상블(µg/m³ 평균). 예외: C GCN DROP은 1차에서 시드 11만(나머지 학습 중)"]],
      ["항목", "내용"], widths=[3, 14])

# ---------------------------------------------------------------- accuracy
h("3. 정확도 비교 (2023, µg/m³ RMSE)")
rows = rmse_table(R / "C/scores/validation/core/metrics.csv",
                  [("C_GRU", "D67"), ("C_GRU", "DN"), ("C_GRU", "F"), ("C_GCNGRU", "D67"), ("C_GCNGRU", "DN"), ("C_GCNGRU", "F"), ("C_GCNGRU_PERM", "F")])
rows += rmse_table(R / "B/scores/validation/core/metrics.csv", [("LightGBM", "D67"), ("LightGBM", "DN"), ("LightGBM", "F")])[1:]
table(rows, ["모델", "조건", "3 h", "6 h", "12 h", "24 h"], widths=[4.5, 2, 2.5, 2.5, 2.5, 2.5], bold_rows=(3, 10))
fig(R / "B/analysis/validation/b_vs_c_accuracy.png", "그림 1. B·C 대표 모델 정확도 (시드 앙상블)", 13)
fig(R / "C/analysis/validation/c_accuracy_by_horizon.png", "그림 2. C 모델 × 조건 정확도", 13)

# ---------------------------------------------------------------- C
h("4. 모델 C: GRU / GCN-GRU")
h("4.1 가설 판정", 2)
table([["C-H1", "GCN-GRU가 GRU보다 낫다", "GRU가 3·6시간에서 유의하게 우세(차이 +0.5~+0.8)", "기각"],
       ["C-H2", "GCN 이득은 실제 지리 덕", "실제 그래프 < 무작위 그래프 오차(−0.16 ~ −0.74), 유의는 3 h만", "방향 지지"],
       ["C-H3", "DN이 D67보다 낫다", "두 모델 모두 차이 없음", "기각"],
       ["C-H4", "중국 권역 추가 가치", "ADD·DROP 180개 중 유의 1개(수도권 DROP, 3 h). 랴오닝 ADD 24 h −0.95는 비유의", "기각"],
       ["C-H5", "저통과 지역 가치 없음", "CTRL 6개 비교 모두 비유의", "지지"],
       ["C-H6", "중국 의존도: 겨울·서쪽 경로↑", "계절은 지지(산둥 4.7% vs 0.4%), 경로는 동쪽이 더 큼", "부분 지지"]],
      ["ID", "가설", "결과", "판정"], widths=[1.3, 4.2, 9, 2.5])
fig(R / "C/analysis/validation/c_core_tests.png", "그림 3. C 핵심 쌍 검정 (RMSE 차이와 95% CI, 채운 점 = 유의)", 17)

h("4.2 지역 가리기: 모델이 어느 지역에 기대나", 2)
p("학습된 F 모델(600개 측정소)에서 한 권역의 입력을 가렸을 때 RMSE가 몇 % 늘어나는지입니다. 원 넓이가 클수록 의존도가 크고, 굵은 테두리는 팀 규칙상 유의합니다.")
fig(R / "C/analysis/validation/maps/map_dependence_by_horizon.png", "그림 4. C_GRU 권역 의존도 지도 (지평별)", 17)
table([["수도권", "+8.9", "+15.3", "+41.3", "+72.9", "3–24 h 유의"],
       ["강원", "+5.7", "+8.5", "+7.0", "+6.4", "3–12 h 유의"],
       ["호남", "+1.8", "+4.7", "+5.8", "+6.7", "6 h 유의"],
       ["랴오닝", "+1.5", "+3.2", "+3.6", "+3.8", "3·6 h 유의"],
       ["산둥", "+1.1", "+2.5", "+2.8", "+3.0", "비유의"],
       ["일본 서부", "+1.4", "+1.8", "+1.1", "−0.3", "6 h 유의"],
       ["몽골", "+0.0", "+0.1", "+0.0", "+0.0", "비유의"]],
      ["권역", "3 h", "6 h", "12 h", "24 h", "유의성"], widths=[3, 2.2, 2.2, 2.2, 2.2, 5])

h("4.3 의존도 vs 고유 정보", 2)
p("왼쪽은 학습된 모델에서 권역을 가린 결과, 오른쪽은 그 권역을 빼고 처음부터 다시 학습한 결과입니다. 중국 권역은 왼쪽에서 오차를 늘리지만 오른쪽에서는 효과가 거의 없습니다. 즉 모델이 쓰는 중국 정보는 다른 측정소로 대체될 수 있습니다.")
fig(R / "C/analysis/validation/maps/map_unique_vs_dependence_h12.png", "그림 5. 의존도(가리기) vs 고유 정보(DROP 재학습), 12 h", 17)
fig(R / "C/analysis/validation/maps/map_dependence_season_h12.png", "그림 6. 계절별 의존도 (12 h): 중국 권역은 추운 계절에만 뚜렷", 17)
fig(R / "C/analysis/validation/c_regional_forest.png", "그림 7. 권역 추가(ADD)·제외(DROP) 효과와 95% CI (C_GRU 위, C_GCNGRU 아래)", 17)

# ---------------------------------------------------------------- B
h("5. 모델 B: LightGBM + TreeSHAP")
h("5.1 권역 추가 가치", 2)
table([["유의한 비교", "92개 중 2개: F vs DROP_KR_CAPITAL (3 h −0.84, 6 h −0.55) → 수도권은 대체 불가"],
       ["중국 권역 ADD (12/24 h)", "랴오닝 −0.66/−0.88 · 내몽골 −0.48/−0.43 · 산둥 −0.28/−0.39 · 징진지 −0.11/−0.43 (모두 비유의)"],
       ["일본·몽골 ADD", "모든 지평에서 +0.03 ~ +0.19 (오히려 약간 나빠짐)"],
       ["F vs DN", "24 h −1.05로 가장 큰 차이지만 비유의"],
       ["대조 3개", "중국 동부 · 일본 동해 쪽 · 일본 서부 25곳: 모두 비유의"]],
      ["항목", "결과 (RMSE 차이, 음수 = 개선)"], widths=[4.5, 12.5])
fig(R / "B/analysis/validation/b_regional_forest.png", "그림 8. B 권역 추가·제외 효과와 95% CI", 17)

h("5.2 TreeSHAP: 예측을 무엇으로 만드나", 2)
p("F 모델 예측에서 측정소 특징 |SHAP|의 권역별 비중입니다(서울 자체 이력·기상·달력 제외). 정확도 기여가 아니라 '예측값을 만드는 데 쓰인 정도'입니다.")
fig(R / "B/analysis/validation/maps/b_shap_map_by_horizon.png", "그림 9. B SHAP 권역 비중 지도: 3 h 수도권 77% → 24 h 17%, 중국 비중 증가", 17)
sh = pd.read_csv(R / "B/analysis/validation/shap_region_horizon.csv", index_col=0)
st = sh[~sh.index.str.startswith("COMMON_")]
share = (100 * st / st.sum()).round(1)
share = share.sort_values("24", ascending=False)
table([[r] + [f"{share.loc[r, c]:.1f}%" for c in ("3", "6", "12", "24")] for r in share.index],
      ["권역", "3 h", "6 h", "12 h", "24 h"], widths=[5, 3, 3, 3, 3])
fig(R / "B/analysis/validation/shap_region_stat_h6.png", "그림 10. 권역 × 통계량 |SHAP| (6 h): 수도권 '현재값'이 압도적", 15)
fig(R / "B/analysis/validation/shap_high_pm_h6.png", "그림 11. 고농도 상위 10건(6 h)의 권역별 기여", 16)
fig(R / "B/analysis/validation/shap_region_condition_h6.png", "그림 12. 국외 권역 |SHAP| × 조건(계절·경로·바람) (6 h)", 17)

# ---------------------------------------------------------------- status
h("6. 진행 상황과 다음 단계")
table([["C 1차 (시드 11 기준 DROP)", "완료 · GitHub 반영"],
       ["C 나머지 시드 (GCN DROP 22·33 등 32회)", "학습 중 · 오늘 저녁 완료 후 재채점, 표·그림 갱신"],
       ["B 29개 조건 + TreeSHAP", "완료 · GitHub 반영"],
       ["A SARIMAX(권역별) + 그랜저", "적합 중(167개 중 ~140) · 완료 후 채점·그림"],
       ["README 정리", "A 완료 후"],
       ["2024 최종 평가 (--final)", "모든 설정 동결 후 한 번만"],
       ["D (DCRNN, 담당자)", "C의 거리 그래프 기준선 제공 예정"]],
      ["작업", "상태"], widths=[8, 9])
h("해석 시 주의", 2)
bullet("가리기·DROP·SHAP 모두 '예측 정보'에 대한 측정이며 배출 기여나 인과를 뜻하지 않는다.")
bullet("2023 한 해의 검증 결과다. 효과가 작아 연도에 따라 달라질 수 있으며, 결론은 2024 최종 평가로 확정한다.")
bullet("C의 GCN DROP 비교는 1차에서 시드 11끼리만 비교했다(공정 비교). 시드 3개가 모이면 갱신한다.")

OUT.parent.mkdir(exist_ok=True)
doc.save(OUT)
print(OUT)
