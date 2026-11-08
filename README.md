# Time-Series Data Analysis and Practice: Team06 PM2.5 Project

서울 중구 PM2.5의 3·6·12·24시간 뒤 값을 예측합니다. 국내외 관측과 HYSPLIT 역궤적을 이용해, **어느 지역 정보가 언제 예측에 도움이 되는지**를 검증합니다.

- 데이터: `Dataset_11_07_regions_v5.zip`(저장소에 포함하지 않음). 압축을 푼 폴더를 `TSA_DATA` 환경변수로 지정합니다.
- 채점: 팀 공통 채점기 `Model/Common/score.py`(데이터 zip 안). 평가 행 고정, 시드별 점수와 앙상블 구분, DM 검정·7일 블록 bootstrap·Holm 보정을 씁니다.

| 폴더 | 내용 |
|---|---|
| `C_graph/` | 모델 C: GRU / GCN-GRU / 무작위 그래프 대조, 26개 팀 조건 + 3개 대조 조건, 지역 가리기, 채점 |
| `docs/C_PLAN.md` | C의 가설, 실험 목록, 서버 실행법과 사양 |
| `results/C/` | 실행별 예측·학습 기록·체크포인트, 채점 결과 |

A·B·D는 C 완료 후 같은 구조로 추가합니다.
