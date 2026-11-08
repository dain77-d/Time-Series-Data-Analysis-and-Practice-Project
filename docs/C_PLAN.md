# 모델 C 계획: 가설 · 실험 · 실행 · 서버

작성 2026-11-08. 데이터는 `Dataset_11_07_regions_v5.zip`(팀 11-07 규약 + 중국 동부·일본 동해 쪽 추가)를 씁니다.

## 0. 가설 검토 (노션 초안에 대한 의견)

결론부터 말하면 **방향은 좋습니다.** 다만 팀원이 11-07에 정리한 공통 규약(`Model/Common/RESEARCH_PROTOCOL.txt`)이 더 최신이므로, C는 그 규약의 RQ·가설 번호를 따르고 아래를 보완했습니다.

| 초안 | 의견 | C에서의 처리 |
|---|---|---|
| RQ1/RQ2 일반화 문구 | 좋음. RQ2를 "조건에 따른 정보 가치(RQ2a) + 조건을 쓴 활용(RQ2b)"로 나눈 것이 명확함 | 그대로 사용 |
| H1a "중국 추가 → 오차 감소" | 국내 기준선이 67곳이면 "국내 정보 부족"이 섞임. 규약처럼 **전국 국내(DN) 기준선** 위에서 봐야 함 | DN 대비 ADD/DROP로 검증 |
| H1b "3<6<12<24 단조 증가" | 단조성을 가설로 고정하면 반증이 너무 쉬움. 규약도 단조 제약을 뺐음 | "지평에 따라 다르다"로 완화, 크기만 보고 |
| H1d "일본·몽골은 이득 0(음성 대조)" | 공통 기상 변동 때문에 0이 아닐 수 있음. **대조는 0을 가정하지 말고 "통과율 낮은 지역"으로 명시적으로 설계**하는 게 맞음 | 11-07 신규 지역(중국 동부 4.9%, 일본 동해 쪽 0.3% 통과)으로 대조 실험 |
| H2b "궤적 고도 바람이 지상 바람보다 조건을 더 잘 가른다" | 핵심 가설로 좋음. 단 Adaptive(B)만이 아니라 **조건별 정보 의존도(C의 가리기)로도** 확인 가능 | C-H6에서 두 조건 변수로 동시 보고 |
| H4a "GCN이 GRU보다 낫다" | 좋음. 그러나 이겨도 **그래프 덕인지 구조(노드별 GRU) 덕인지** 구분 불가 | 무작위 그래프 대조(C_GCNGRU_PERM) 추가 |
| H4b "D가 C보다 낫다" | 구조와 그래프가 동시에 다름. 규약처럼 **방향 그래프 대 그 대칭화**로 비교해야 함 | D 담당 몫. C는 거리 그래프 기준선 제공 |

중심 가설-보조 가설 구조는 **"정보가 어디에 있나(RQ1) → 언제 유용한가(RQ2a) → 어떻게 써야 하나(RQ2b·구조)"** 순서로 두면 자연스럽습니다. C는 이 중 RQ1·RQ2a를 **신경망 관점에서 재현**하고, 구조 질문(그래프의 가치)을 답합니다.

## 1. C의 가설 (결과 보기 전 고정)

| ID | 가설 | 비교 (같은 평가행, 시드 3개 앙상블) | 연결 |
|---|---|---|---|
| **C-H1** | 거리 그래프를 쓰는 GCN-GRU가 그래프 없는 GRU보다 오차가 작다 | `C_GCNGRU` vs `C_GRU` × {D67, DN, F} | H4a |
| **C-H2** | 그 이득은 **실제 지리** 때문이다 | `C_GCNGRU` vs `C_GCNGRU_PERM`(노드 번호를 섞은 같은 그래프) @F | H4a 보강 |
| **C-H3** | 전국 국내 관측(DN)이 국내 67곳(D67)보다 낫다 | DN vs D67 (두 모델) | H1a |
| **C-H4** | 강화된 국내 기준선 위에서도 일부 중국 권역은 추가 가치가 있고, 크기는 권역·지평마다 다르다 | ADD_r vs DN (8개), F vs DROP_r (15개) | H1b |
| **C-H5** | 궤적이 거의 지나지 않는 지역은 추가 가치가 없거나 작다 | CTRL_CN_EAST / CTRL_JP_EAST / CTRL_JP_WEST25 vs DN | 음성 대조 |
| **C-H6** | 중국 권역에 대한 의존도는 서쪽 접근(궤적 경로·궤적 고도 바람)과 겨울에 크고, 궤적 고도 바람이 지상 바람보다 이를 잘 가른다 | F 모델의 권역 가리기 오차 증가를 조건별로 | H2a, H2b |

**판정 규칙(팀 규약):** RMSE 차이의 7일 블록 bootstrap 95% 구간 상한 < 0 **그리고** Holm 보정 p < 0.05.
- 설정은 2023(검증)으로 동결합니다.
- 2024는 `--final`로 한 번만 채점합니다.
- 작거나 음수인 효과도 그대로 보고합니다.

**해석 원칙:**
- 가리기는 "모델이 기대는 정도"(의존도)를 잽니다.
- DROP 재학습은 "대체할 수 없는 정보"를 잽니다.
- 둘이 다르면 그 권역의 정보가 다른 권역과 겹친다는 뜻입니다.
- 어느 쪽도 배출 기여나 인과를 뜻하지 않습니다.

## 2. 실험 목록 (`C_graph/experiments.py`)

| 묶음 | 모델 | 조건 | 실행 수 |
|---|---|---|---|
| E1 핵심 | C_GRU, C_GCNGRU | D67, DN, F | 18 |
| E2 그래프 대조 | C_GCNGRU_PERM | F | 3 |
| E4 저통과 대조 | C_GRU, C_GCNGRU | CTRL_CN_EAST, CTRL_JP_EAST, CTRL_JP_WEST25 | 18 |
| E3 권역 가치 | C_GRU, C_GCNGRU | ADD_* 8 + DROP_* 15 | 138 |
| 가리기 | F로 학습한 6개 모델 | 15개 권역 | 재학습 없음 |

- 시드는 11, 22, 33이고, 합계 177회 학습입니다. 중요도 순서(E1 → E2 → E4 → E3)로 실행하며, 중단해도 이어서 돌릴 수 있습니다.
- **노드:** 서울 중구 + 조건에 포함된 측정소입니다(D67=68, DN=515, F=601, CTRL=531~533). 전체 후보는 651곳입니다.
- **그래프:** 팀 규약의 고정값을 씁니다. 가우시안 σ=182.34 km, 500 km 이내만 연결합니다.
- **공통 설계(모든 조건 동일):**
  - 입력: 48시간 창, log1p 표준화(2021–22 통계), 결측 표시 채널, ASOS 108·102(기온·습도·기압·강수·u·v), 달력
  - 출력: 마지막 관측값 + 변화량(잔차) 형태의 4개 지평
  - 학습: Adam 1e-3, 최대 40 epoch, 2023 조기 종료(인내 6)

## 3. 실행 방법 (서버)

```bash
git clone https://github.com/dain77-d/Time-Series-Data-Analysis-and-Practice-Project.git && cd Time-Series-Data-Analysis-and-Practice-Project
pip install -r requirements.txt            # torch는 서버 CUDA에 맞는 것으로
unzip Dataset_11_07_regions_v5.zip -d ../data_v5
export TSA_DATA=$PWD/../data_v5
cd C_graph
python cdata.py                            # 노드 651 × 시간 캐시 (1~2분)
python experiments.py --list
nohup python experiments.py --workers 2 > ../results/C/queue.log 2>&1 &   # E1→E2→E4→E3
# GPU 여러 장: nohup python experiments.py --gpus 4 --per-gpu 2 > ../results/C/queue.log 2>&1 &
python occlude.py                          # F 모델 가리기
python score_all.py                        # 2023 검증 채점, 가설 표
python region_contribution.py              # 권역별 의존도·고유 정보 표와 히트맵
python score_all.py --split test --final   # 모든 설정 동결 후 한 번
```

첫 GCN 실행의 로그(`results/C/logs/C_GCNGRU__D67__seed11.log`)에 epoch당 초가 찍힙니다. 이 값으로 전체 시간을 다시 추정하세요.

## 4. 서버 사양과 예상 시간

**측정값 (로컬, 노트북 CPU 4스레드):**
- GCN-GRU, 노드 601개: 학습 표본 1개당 순전파 약 0.045초 → **한 epoch 약 45분**. 177회를 로컬로 돌리는 것은 불가능합니다.
- GRU, 노드 68개: 60배치 + 검증에 약 4초.

**GPU 추정:** 표본당 연산량(노드 600개 기준 약 4 GFLOP, 순전파+역전파)과 위 CPU 측정을 바탕으로 했습니다. 실측이 아니므로 첫 epoch 로그로 확인해야 합니다.

| GPU (1장) | GCN epoch당 | 전체 177회 + 가리기 | 비고 |
|---|---|---|---|
| **RTX 4090 24GB** | 약 10–20초 | **약 6–10시간** | 가성비 최고. vast.ai·RunPod 시간당 약 $0.4–0.7 → **약 $3–7** |
| A100 40/80GB | 약 8–15초 | 약 5–8시간 | 시간당 약 $1.2–2 |
| L4 24GB | 약 30–50초 | 약 15–25시간 | |
| **L4 24GB × 4 (gp4l16-g3)** | 장당 약 30–50초 | **약 4–7시간** | `--gpus 4 --per-gpu 2`로 8개 동시 실행. 프로세스당 RAM 약 2–3 GB |
| Colab T4 16GB | 약 60–100초 | 약 30–50시간 | 세션 끊김 때문에 비추천 |

**추천 사양:**
- GPU: **RTX 4090 24GB 1장** (없으면 A100 40GB)
- vCPU 8개, RAM 32 GB, 디스크 60 GB
- Ubuntu 22.04, PyTorch 2.4 이상 + CUDA 12

`--workers 2`로 같은 GPU에서 두 실행을 동시에 돌립니다. GPU 메모리는 실행당 약 3–6 GB입니다. B·D까지 같은 서버에서 할 계획이면 A100 80GB도 좋습니다.

## 5. 산출물

- `results/C/runs/<모델>/<조건>/seed<k>/`: `predictions.parquet`(팀 채점기 형식), `meta.json`(학습 곡선·설정·시간), `model.pt`
- `results/C/scores/<split>/<묶음>/`: 팀 `score.py` 결과(metrics, tests, conditions, seed_metric_summary), `summary_<split>.md`
- 그림·표는 `analysis/`에 추가 예정입니다.
  - 지평별 개선률
  - 권역 × 지평 가치 히트맵(ADD/DROP)
  - 가리기 × 조건 히트맵
  - 그래프 대조 결과
