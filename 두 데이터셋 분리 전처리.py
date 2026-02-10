import pandas as pd
import numpy as np

# =========================
# 0. 데이터 불러오기
# =========================
df = pd.read_csv("2010-2011model2.csv")

print("원본 데이터 크기:", df.shape)

# =========================
# 1. 40세 이상만 포함
# =========================
df = df[df["age"] >= 20].copy()
print("40세 이상 필터 후:", df.shape)
def recode_age_group(age):
    if pd.isna(age):
        return np.nan
    if 20 <= age <= 39:
        return 1
    elif 40 <= age <= 59:
        return 2
    elif 60 <= age <= 79:
        return 3
    elif age >= 80:
        return 4
    else:
        return np.nan

df["AGE_GROUP"] = df["age"].apply(recode_age_group)
# MET 변수는 재코딩 없이 그대로 사용
df["MET"] = df["MET"]

# =========================
# 2. 변수 재코딩
# =========================

# -------------------------
# (1) 성별: 0→1, 1→2
# -------------------------
df["sex"] = df["sex"].map({0: 1, 1: 2})

# -------------------------
# (2) 음주
# 1,8 → 1 (비음주)
# 2 → 2 (음주)
# 9 → NaN
# -------------------------
df["BD1"] = df["BD1"].replace({
    1: 1, 8: 1,
    2: 2,
    9: np.nan
})

# -------------------------
# (3) 흡연
# 1,2 → 1 (흡연 경험 있음)
# 3,8 → 2 (비흡연)
# 9 → NaN
# -------------------------
df["BS1_1"] = df["BS1_1"].replace({
    1: 1, 2: 1,
    3: 2, 8: 2,
    9: np.nan
})

# -------------------------
# (4) 소득
# 1 → 1 (하)
# 2,3 → 2 (중)
# 4 → 3 (상)
# -------------------------
df["ho_incm"] = df["ho_incm"].replace({
    1: 1,
    2: 2, 3: 2,
    4: 3
})

# -------------------------
# (5) 교육 수준
# 1,2 → 1 (고졸 미만)
# 3 → 2 (고졸)
# 4 → 3 (고졸 초과)
# -------------------------
df["edu"] = df["edu"].replace({
    1: 1, 2: 1,
    3: 2,
    4: 3
})
def recode_bmi(bmi):
    if pd.isna(bmi):
        return np.nan
    if bmi < 18.5:
        return 1
    elif 18.5 <= bmi <= 24.9:
        return 2
    elif 25.0 <= bmi <= 29.9:
        return 3
    elif bmi >= 30.0:
        return 4
    else:
        return np.nan

df["BMI"] = df["HE_BMI"].apply(recode_bmi)

# -------------------------
# (6) 결혼 상태 (marri_1 + marri_2)
# -------------------------
def recode_marital(row):
    m1 = row["marri_1"]
    m2 = row["marri_2"]

    if m1 == 9 or m2 in [8, 9, 99]:
        return np.nan

    if m1 == 2:              # 미혼
        return 1
    if m1 == 1 and m2 == 1:  # 기혼 + 동거
        return 2
    if m1 == 1 and m2 in [2, 3, 4]:  # 별거/사별/이혼
        return 3

    return np.nan


df["MARITAL"] = df.apply(recode_marital, axis=1)

# =========================
# 3. 분석 변수 정의
# =========================

bmd_vars = ["DX_NK_BMD"]

diet_vars = [
    "N_EN", "N_WATER", "N_PROT", "N_FAT", "N_CHO",
    "N_FIBER", "N_ASH",
    "N_CA", "N_PHOS", "N_FE", "N_NA", "N_K",
    "N_VA", "N_CAROT", "N_RETIN",
    "N_B1", "N_B2", "N_NIAC", "N_VITC"
]

covariates = [
    "age", "sex", "HE_BMI",
    "BS1_1", "BD1",
    "ho_incm", "edu", "MARITAL","AGE_GROUP"
]

survey_vars = ["wt_itvex", "kstrata", "psu"]

analysis_vars = bmd_vars + diet_vars + covariates + survey_vars

df_analysis = df[analysis_vars].copy()

print("전처리 후 dropna 전:", df_analysis.shape)

# =========================
# 4. 결측 제거 (최종)
# =========================
df_analysis = df_analysis.dropna()

print("최종 분석 데이터:", df_analysis.shape)

# =========================
# 5. 요인/군집용 식이 데이터 분리
# =========================
df_diet = df_analysis[diet_vars].copy()

# =========================
# 6. CSV 저장
# =========================
df_analysis.to_csv(
    "knhanes_analysis_ready.csv",
    index=False,
    encoding="utf-8-sig"
)

df_diet.to_csv(
    "knhanes_diet_only.csv",
    index=False,
    encoding="utf-8-sig"
)

print("CSV 저장 완료")
