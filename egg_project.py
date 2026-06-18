import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
import math
import joblib

# ── 1. 데이터 로드 ────────────────────────────────────────────────
# 전처리 완료된 CSV 파일을 불러옴
# 날짜(날짜) 열을 인덱스로 설정하고 datetime 형식으로 파싱
df = pd.read_csv('egg_data_clean.csv', index_col='날짜', parse_dates=True)

# ── 2. 입력(X)과 정답(y) 분리 ────────────────────────────────────
# X: 모델이 학습에 사용할 독립변수 2개
#    - ai_outbreak : 조류독감 발생 여부 (0 = 없음, 1 = 발생)
#    - usd_krw     : 원달러 환율
# y: 예측 대상인 종속변수 (계란 30개 가격, 단위: 원)
X = df[['ai_outbreak', 'usd_krw']]
y = df['egg_price']

# ── 3. 학습 / 테스트 데이터 분리 ─────────────────────────────────
# 전체 132행 중 80%는 학습, 20%는 성능 평가에 사용
# shuffle=True  : 시간 순서를 무시하고 랜덤으로 섞어서 분리
#                 (시간 순으로 자르면 최근 데이터만 테스트 → 과소 평가 가능)
# random_state=42: 실행마다 동일한 결과를 얻기 위한 시드 고정
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, shuffle=True, random_state=42
)

# ── 4. 모델 성능 평가 ─────────────────────────────────────────────
# 학습용 데이터로만 모델을 훈련한 뒤 테스트 데이터로 성능을 측정
# 이 모델은 평가 목적으로만 사용 (실제 예측에는 아래 최종 모델 사용)
model_eval = LinearRegression().fit(X_train, y_train)
y_pred = model_eval.predict(X_test)

# R²  : 1에 가까울수록 예측력이 높음 (0이면 평균 예측과 동일, 음수면 평균보다 나쁨)
# RMSE: 예측값과 실제값의 평균 오차 (단위: 원)
print(f"R²  : {r2_score(y_test, y_pred):.4f}")
print(f"RMSE: {math.sqrt(mean_squared_error(y_test, y_pred)):.1f}원")

# ── 5. 최종 모델 학습 (전체 데이터 사용) ─────────────────────────
# 성능 평가 후, 예측 프로그램에 사용할 최종 모델은 전체 132행으로 학습
# 데이터를 최대한 활용해야 더 정확한 계수를 추정할 수 있음
model = LinearRegression().fit(X, y)

# 회귀 계수(coef_): 각 독립변수가 계란값에 미치는 영향
# 절편(intercept_): 모든 변수가 0일 때의 기본 예측값
print(f"\nAI 발생 시  : +{model.coef_[0]:.0f}원")
print(f"환율 100원↑ : +{model.coef_[1]*100:.0f}원")
print(f"절편        : {model.intercept_:.1f}")

# ── 6. 모델 저장 ──────────────────────────────────────────────────
# joblib을 사용해 학습된 모델을 파일로 저장
# app.py(웹 서버)에서 이 파일을 불러와 사용 → 서버 시작마다 재학습 불필요
joblib.dump(model, 'egg_model.pkl')
print("\n✅ egg_model.pkl 저장 완료")
