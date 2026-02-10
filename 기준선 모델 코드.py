import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score

# =========================
# 1. 데이터 불러오기
# =========================
df = pd.read_csv("영양 모델용.csv")

# 필요한 변수만 선택
df_base = df[["DX_NK_BMD", "age", "sex", "HE_BMI"]].dropna()

X = df_base[["age", "sex", "HE_BMI"]]
y = df_base["DX_NK_BMD"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)
model_base = LinearRegression()
model_base.fit(X_train, y_train)
y_pred = model_base.predict(X_test)

r2_base = r2_score(y_test, y_pred)
mae_base = mean_absolute_error(y_test, y_pred)

print("Baseline R²:", r2_base)
print("Baseline MAE:", mae_base)
result_base = pd.DataFrame({
    "Model": ["Baseline (age + sex + HE_BMI)"],
    "R2": [r2_base],
    "MAE": [mae_base]
})

result_base.to_csv("baseline_model_performance.csv", index=False)
