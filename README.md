# 서울 중구 PM2.5 예측: 어느 지역 정보가, 언제 도움이 되나

**Team06 · 시계열데이터분석및실습** · 담당: [dain77-d](https://github.com/dain77-d)

이 저장소는 팀 프로젝트에서 **제가 맡은 부분(EDA, 모델 A·B·C)의 코드와 결과 전체**입니다. 팀 공용 저장소는 [CHVRMER/TSA-PM2.5-Forecasting-in-Seoul](https://github.com/CHVRMER/TSA-PM2.5-Forecasting-in-Seoul)이고, D(DCRNN)는 팀원이 맡았습니다.

서울 중구(KR:111121) PM2.5의 **3·6·12·24시간 뒤 값**을 예측합니다. 국내 514곳, 중국·일본·몽골 86곳의 관측과 HYSPLIT 역궤적을 이용해 다음 세 가지를 검증합니다.

1. 어느 지역 정보가 예측에 도움이 되는가
2. 언제 도움이 되는가
3. 어떤 모델 구조가 그 정보를 잘 쓰는가

## 제가 한 일

| 부분 | 한 일 | 규모 | 코드 · 결과 |
|---|---|---|---|
| **EDA** | 자료 품질 점검부터 650개 측정소 네트워크까지 8개 유형으로 탐색 | 스크립트 13개, 그림 56장, 표 59개 | [`EDA/`](EDA/) |
| **A** | SARIMAX(권역 하나씩) + 그랜저 검정(계절·경로별): 선형 기준선, "앞서 움직이는 권역" | 20개 조건 | [`A_sarimax/`](A_sarimax/) · [결과](results/A/RESULTS_A.md) |
| **B** | LightGBM 29개 입력 조건 × 시드 3개 + TreeSHAP: 정확도와 예측 근거 분해 | 87회 학습 | [`B_lgbm/`](B_lgbm/) · [결과](results/B/RESULTS_B.md) |
| **C** | GRU / 거리 그래프 GCN-GRU / 무작위 그래프 대조 + 지역 가리기·DROP·ADD | 학습 168회 + 가리기 90회 | [`C_graph/`](C_graph/) · [결과](results/C/RESULTS_C.md) |
| 공통 | 팀 채점기 기반 가설별 검정 묶음(DM 검정·블록 bootstrap·Holm), 미래 정보 누출 검사, 파이프라인 문서, 팀 진행 보고서 | | [`docs/`](docs/) · [`reports/`](reports/) |

**설계에서 신경 쓴 점**
- **한 번만 여는 시험 구간:** 설정 선택은 2023 검증으로만 하고, 2024는 `--final`로 한 번만 엽니다.
- **대조 지역(음성 대조군):** 궤적이 거의 지나지 않는 중국 동부·일본 동해 쪽을 일부러 넣었습니다. 효과가 나오면 "수송"이 아니라 공통 기상 신호라는 경고로 씁니다.
- **의존 vs 고유 정보 구분:** 학습된 모델에서 지역을 가리는 실험(의존)과 빼고 다시 학습하는 실험(고유 정보)을 나눠, "쓰는 정보"와 "대체할 수 없는 정보"를 구분했습니다.
- **재현성:** 시드 3개(11·22·33)로 앙상블과 시드별 점수를 함께 보고합니다. C 모델은 미래 값을 바꿔도 예측이 변하지 않는지 검사합니다(`C_graph/test_causal.py`).

**한 줄 결론:** 해외 관측소를 얼마나 많이 넣는가보다, **이동 경로와 시차를 고려해 그 정보를 어떻게 쓰는가**가 서울 PM2.5 예측에서 더 중요합니다. EDA와 A·B·C가 같은 방향을 가리킵니다.

> **현재 상태 (2026-11-13):** 2023 검증 결과입니다. 2024 시험 구간은 아직 열지 않았습니다. C는 177회 중 168회를 완료했습니다(GCN DROP 4개 권역은 일부 시드만 있음). 궤적 고도 바람·혼합고 특징(H1c)은 B 코드에 조건만 추가했고 결과는 아직 없습니다.

---

## EDA: 데이터에서 먼저 확인한 것

학습 구간(2021–22)만 써서 8개 유형으로 나눴습니다. 목차와 전체 발견은 [`EDA/README.md`](EDA/README.md)에 있습니다.

| # | 유형 | 대표 발견 |
|---|---|---|
| 1 | [자료 구성과 품질](EDA/01_data_inventory_and_quality/) | 측정소 650곳 가용률 92–99%. 중국 오류값(9985), PM2.5 > PM10 1만 7천 시간 등 정제 대상 확인 |
| 2 | [서울 PM2.5 특성](EDA/02_seoul_pm25_characteristics/) | 주기로 설명되지 않는 변동 51%, 24시간 뒤 값과의 상관 0.45뿐 |
| 3 | [기상 요인](EDA/03_meteorological_drivers/) | 백령도 서풍이면 12시간 뒤 서울이 평균 +8.8 µg/m³ |
| 4 | [공간 상관](EDA/04_spatial_correlation_structure/) | 같은 시각 상관은 약하지만 시차를 맞추면 중국 권역이 12–24시간 앞섬(시속 35–46 km) |
| 5 | [이동경로와 사건](EDA/05_transport_pathways_and_events/) | 서해 횡단 경로 평균 27 vs 동쪽 경로 10 µg/m³. 2022 설날 불꽃놀이 급등이 백령도 +8시간 → 서울 +21시간 |
| 6 | [화학 성분](EDA/06_chemical_composition/) | 백령도와 서울은 PM2.5가 비슷하지만 PM2.5당 NO₂는 서울이 5배 |
| 7 | [예측 가능성과 모델 오차](EDA/07_predictability_and_model_errors/) | 해외 정보는 평균으로는 효과가 없지만, 정체·서쪽 경로의 24시간 예측에서 RMSE를 0.75–1.51 줄임 |
| 8 | [다지점 네트워크](EDA/08_multi_station_network/) | 650곳 모두 서쪽일수록 먼저 움직임. 큰 사례 4개 모두 중국 북부 → 한국 → 일본 순으로 전파 |

![route clusters](EDA/05_transport_pathways_and_events/figures/e5_cluster_paths.png)

![network lead lag](EDA/08_multi_station_network/figures/e11_station_lead_lag_map.png)

---

## 핵심 결과

### 1. 정확도: LightGBM은 12시간까지, GRU는 24시간에서 가장 좋다

![B vs C accuracy](results/B/analysis/validation/b_vs_c_accuracy.png)

| RMSE (µg/m³, 2023) | 3 h | 6 h | 12 h | 24 h |
|---|---|---|---|---|
| Persistence | 7.18 | 9.97 | 13.11 | 15.20 |
| A SARIMAX (ALL) | 6.46 | 9.35 | 11.36 | 12.48 |
| **B LightGBM (F)** | 5.75 | **8.03** | **10.20** | 12.05 |
| B LightGBM (DN, 국내만) | **5.70** | 8.08 | 10.79 | 13.10 |
| **C GRU (F)** | 6.19 | 8.28 | 10.43 | **11.82** |
| C GCN-GRU (F) | 6.76 | 8.98 | 10.99 | 12.28 |
| C GCN-GRU, 무작위 그래프 | 6.92 | 9.35 | 11.67 | 13.02 |

거리 그래프 GCN은 그래프 없는 GRU보다 나빴습니다(3·6시간 유의). 다만 무작위 그래프보다는 나아서, 그래프 자체에는 지리 정보가 담겨 있습니다.

### 2. 멀리 볼수록 예측이 중국 정보로 만들어진다 (B, TreeSHAP)

![B SHAP map](results/B/analysis/validation/maps/b_shap_map_by_horizon.png)

측정소 특징의 SHAP 비중을 보면, 수도권이 3시간 77%에서 24시간 17%로 줄어듭니다. 같은 24시간에 중국 권역 비중은 내몽골 14%, 랴오닝 12%, 산둥 9%로 커집니다.

### 3. 그러나 그 정보는 대부분 대체 가능하다 (C, 지역 가리기 vs 재학습)

![C dependence vs unique](results/C/analysis/validation/maps/map_unique_vs_dependence_h12.png)

- **왼쪽(지역 가리기):** 학습된 모델에서 랴오닝·산둥을 가리면 오차가 늘어납니다. 랴오닝은 3·6시간에서 유의합니다.
- **오른쪽(재학습):** 그 권역을 빼고 다시 학습하면 차이가 사라집니다. 대체 불가능한 정보는 **수도권 관측**뿐입니다(B·C 모두 유의).

### 4. 효과는 지평·계절·경로에 따라 다르다

![C dependence map](results/C/analysis/validation/maps/map_dependence_by_horizon.png)

| 질문 | 결과 |
|---|---|
| 중국 권역 추가가 오차를 줄이나? | **A(선형):** 24시간 랴오닝·징진지·산둥·지린, 12시간 산둥이 유의합니다. **B·C(비선형):** 12–24시간에 같은 방향으로 줄지만 유의하지 않습니다 |
| 언제 중국 정보를 더 쓰나? | **추운 계절**에 큽니다. C 가리기 12시간: 산둥 4.7% vs 0.4%. 그랜저 부분 R²는 1.5–6배입니다 |
| 서쪽 경로일 때 더 쓰나? | 그랜저(선형)는 **서쪽 경로에서 3–6배**입니다. C 가리기에서는 지지되지 않았습니다 |
| 궤적이 거의 안 지나는 지역은? | 일본 동해 쪽·일본 서부 대조는 효과가 없습니다(설계대로). 중국 동부는 그랜저만 유의해서, 공통 기상 변동 신호가 섞여 있음을 보여 줍니다 |

![A Granger](results/A/a_granger.png)

> 가리기·DROP·SHAP·그랜저는 모두 **예측 정보**를 재는 도구입니다. 배출 기여나 인과를 뜻하지 않습니다.

모델별 상세 결과와 그림:
- [`results/A/RESULTS_A.md`](results/A/RESULTS_A.md)
- [`results/B/RESULTS_B.md`](results/B/RESULTS_B.md)
- [`results/C/RESULTS_C.md`](results/C/RESULTS_C.md)

팀 보고용 Word 문서: [`reports/`](reports/)

---

## 실험 설계

| 항목 | 내용 |
|---|---|
| 데이터 | `Dataset_11_07_regions_v5.zip`. 팀 11-07 규약에 통과율 낮은 대조 지역(중국 동부 16곳, 일본 동해 쪽 18곳)을 더했습니다. 저장소에는 넣지 않습니다 |
| 분할 | 학습 2021–22 · 설정 선택 2023 · 최종 평가 2024 |
| 입력 조건 | D67(국내 67곳) · DN(국내 전국) · F(국내+국외) · ADD_r 8개 · DROP_r 15개 · CTRL 3개 |
| 채점 | 팀 공통 `score.py`. 고정 평가 행, DM 검정(HAC), 7일 블록 bootstrap, Holm 보정 |
| 개선 판정 | 95% 구간 상한 < 0 **그리고** Holm p < 0.05 |
| 시드 | 11·22·33. µg/m³ 시드 평균 앙상블과 시드별 점수를 따로 보고합니다 |

전체 흐름은 [`docs/PIPELINE.md`](docs/PIPELINE.md), 가설과 계획은 [`docs/C_PLAN.md`](docs/C_PLAN.md)와 [`docs/PLAN_A_to_D.md`](docs/PLAN_A_to_D.md)에 있습니다.

## 폴더 구조

```
EDA/         01–08 유형별 폴더 (스크립트, figures/, tables/, README), common.py, PLAN.md
A_sarimax/   asarimax.py        SARIMAX 권역별 + 그랜저 (+ --score, --figures)
B_lgbm/      blgbm.py           LightGBM 29조건 + TreeSHAP
             banalysis.py       채점, SHAP 표·그림
             bplots.py          SHAP 지도, 효과 그림, B vs C
C_graph/     cdata.py           651개 측정소 × 시간 캐시, 조건, 거리 그래프
             train.py           GRU / GCN-GRU / 무작위 그래프
             experiments.py     177회 실험 대기열 (재시작 가능, GPU 여러 장)
             occlude.py         지역 가리기
             score_all.py       가설별 채점 묶음
             region_contribution.py, plot_map.py, plot_summary.py
docs/        PIPELINE.md, C_PLAN.md, PLAN_A_to_D.md, make_report.py
results/     A/ B/ C/   예측(팀 형식), 점수, 표, 그림, RESULTS_*.md
reports/     팀 보고용 Word
assets/      Natural Earth 육지 경계 (지도용)
```

## 실행

```bash
pip install -r requirements.txt lightgbm statsmodels
export TSA_DATA=/path/to/data_v5          # 압축 푼 데이터 폴더
python C_graph/cdata.py                   # 캐시 (1–2분)
python C_graph/experiments.py --gpus 1 --per-gpu 4 --max-heavy 2
python B_lgbm/blgbm.py --jobs 8 --threads 14
python A_sarimax/asarimax.py --jobs 48
python C_graph/occlude.py && python C_graph/score_all.py && python C_graph/region_contribution.py
python B_lgbm/banalysis.py && python A_sarimax/asarimax.py --score && python A_sarimax/asarimax.py --figures
python C_graph/plot_map.py && python C_graph/plot_summary.py && python B_lgbm/bplots.py
python EDA/05_transport_pathways_and_events/eda_e5_trajectories.py   # EDA: 이것 먼저, 나머지 EDA 스크립트는 순서 무관
```

- 실행 시간(L40S 1장, 128코어 기준): C 약 7–8시간, B 약 3시간(C와 동시), A 약 1시간
- 2024 최종 채점은 같은 명령에 `--split test --final`을 붙여 **한 번만** 합니다.
- 모델 체크포인트(`model.pt`)는 용량 때문에 저장소에서 제외했습니다.
