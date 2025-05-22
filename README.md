# -_-
제3회 전국 대학생 리스크 관리 경진대회 결과물 
팀명: 스꾸 방범대
주제: 인공지능 사고 보험 가능성 및 리스크 예측 모델을 활용한 인공지능 보험 상품 제안 
본 프로젝트는 AI 시스템의 사고 데이터를 기반으로 인공지능 사고 보험 상품의 설계 가능성을 탐색하고, 리스크 예측 모델을 구축하여 고위험 및 저위험 인공지능 시스템을 식별하는 것을 목표로 합니다.
이를 위해, 실제 AI 사고 사례 데이터(AIID)를 수집하고, 머신러닝 및 GAMLSS(Generalized Additive Models for Location, Scale and Shape) 기반 분석을 통해 frequency–severity modeling을 수행하였습니다.

#dataset 설명
| 파일명                       | 설명                                                                                                                                                         |
| ------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `AIID 원본 데이터`             | AI 사고 리포트를 수집한 원천 데이터. 컬럼: `['date', 'Alleged deployer of AI system', 'Alleged developer of AI system', 'description', 'Risk Domain', 'Entity', 'Intent']` |
| `training_input_data.csv` | 원본 데이터를 기반으로 GPT API를 활용해 `developer_category`, `deployer_category`, `severity`를 추가한 학습용 데이터셋                                                              |
| `gamlss_data.csv`         | GAMLSS 모델 학습을 위한 전처리 완료 데이터. `training_input_data.csv`를 가공                                                                                                 |
| `활용데이터_madeby.csv`        | 실제 보험 상품에 적용 가능한 가상의 테스트 케이스 데이터. 모델의 예측 성능 확인용                                                                                                            |
1. AIID에서 다운로드받은 데이터 columns = [’date’, ‘Alleged deployer of AI system’, ‘Alleged developer of AI system’, ‘description’, ‘Risk Domain’, ‘Entity’, ‘Intent’]
2. "training_input_data.csv" columns = [’date’,  ‘Alleged deployer of AI system’, ‘developer_category’, ‘deployer_category’, ‘severity’, ‘Risk Domain’, ‘Entity’, ‘Intent’]
     -> [‘developer_category’, ‘deployer_category’, ‘severity’]: 1번 데이터를 기반으로 chatGPT API(role: 일반보험 리스크 관리 전문가로 설)를 활용하여
   developer&deployer_category = ['regulated_corporation', 'unregulated_startup', 'public_agency', 'academic_lab', 'freelancer_or_individual', 'open_source_community']로, 
severity: [0, 1] 값으로 설정
3. "gamlss_data.csv": "training_input_data.csv"를 preprocessing.py에 넣어 전처리한 데이터. GAMLSS에 활용하기 위함
4. "활용데이터_madeby.csv": 전체적인 모델에 직접 적용하여 리스크 레벨 구분을 잘할 수 있는지 알아보기 위한 예시 데이터

#code 설명
| 파일명                         | 설명                                                |
| --------------------------- | ------------------------------------------------- |
| `preprocessing.py`          | AI 사고 데이터를 GAMLSS 모델에 적합하도록 전처리하는 파이프라인           |
| `GAMLSS.R`                  | GAMLSS 기반 리스크 예측 모델 구현 및 고/저위험군 기준 설정             |
| `knn_model_with_pycaret.py` | PyCaret 기반 KNN 모델로 frequency 및 severity 예측        |
| `main.py`                   | 실제 보험 상품 적용 시뮬레이션. 전체 데이터 파이프라인 실행 및 리스크 레벨 분류 수행 |

1. "preprocessing.py": AIID 데이터를 GAMLSS 모델에 넣기 전 전처리를 위한 코드
2. "GAMLSS.R": GAMLSS 모델 -> 인공지능 리스크 예측 모델에 활용될 고/저위험군 기준 확립 코드
3. "knn_model_with_pycaret.py": AIID 데이터를 활용하여 frequency와 severity를 예측하는 모델(pycaret -> knn)
4. "main.py": 실제 보험 상품에 적용되는 코드

#주요 기법 및 도구
- 자연어 처리 기반 분류: GPT API를 통해 비정형 텍스트로부터 category, severity 자동 분류
- 리스크 모델링:
  Frequency: Poisson 회귀
  Severity: GPD(Generalized Pareto Distribution) 및 GAMLSS
- 리스크 스코어링: 예측된 frequency, severity, RAF(위험 조정계수)를 기반으로 expected_loss 계산 및 위험군 분류

#참고
본 프로젝트는 제3회 전국 대학생 리스크 관리 경진대회 출품작입니다.
모델링 및 데이터 전처리에 사용된 언어: Python, R
주요 라이브러리: PyCaret, scikit-learn, statsmodels, GAMLSS, NumPy, pandas
