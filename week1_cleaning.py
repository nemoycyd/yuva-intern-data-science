# Week 1: Data Acquisition, Cleaning, and Preprocessing  (Dataset: Titanic)
import pandas as pd, numpy as np
import matplotlib.pyplot as plt, seaborn as sns
from sklearn.preprocessing import StandardScaler

URL = "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/titanic.csv"
df = pd.read_csv(URL)                      # use pd.read_csv("data/titanic.csv") if offline
print("Shape:", df.shape)
print(df.dtypes)
print("\nMissing values:\n", df.isnull().sum()[df.isnull().sum() > 0])
print("\nMissing % :\n", (df.isnull().mean() * 100).round(1)[df.isnull().sum() > 0])
print("\nDuplicates:", df.duplicated().sum())
print(df.describe().round(2))

# ---- Cleaning ----
clean = df.copy()
clean = clean.drop_duplicates()                                  # 1. duplicates
clean = clean.drop(columns=["deck"])                             # 2. ~77% missing -> drop
clean["age"] = clean["age"].fillna(clean.groupby(["pclass", "sex"])["age"].transform("median"))  # 3. group median
clean["embarked"] = clean["embarked"].fillna(clean["embarked"].mode()[0])                         # 4. mode
clean["embark_town"] = clean["embark_town"].fillna(clean["embark_town"].mode()[0])

# ---- Outliers (IQR rule on fare, age) ----
def iqr_bounds(s):
    q1, q3 = s.quantile([.25, .75]); i = q3 - q1
    return q1 - 1.5 * i, q3 + 1.5 * i
for col in ["fare", "age"]:
    lo, hi = iqr_bounds(clean[col])
    print(f"{col}: bounds=({lo:.1f}, {hi:.1f}), outliers={((clean[col] < lo) | (clean[col] > hi)).sum()}")
lo, hi = iqr_bounds(clean["fare"])
clean["fare_capped"] = clean["fare"].clip(lo, hi)                # winsorise (cap), keep rows
clean["fare_log"] = np.log1p(clean["fare"])                      # reduce skew

# ---- Consistency + feature engineering ----
for c in ["sex", "embarked", "who", "class", "embark_town"]:
    clean[c] = clean[c].astype(str).str.strip().str.lower()
clean["family_size"] = clean["sibsp"] + clean["parch"] + 1
clean["is_alone"] = (clean["family_size"] == 1).astype(int)

# ---- Encode + scale ----
model_df = pd.get_dummies(clean.drop(columns=["alive", "alone", "adult_male", "class", "embark_town", "who"]),
                          columns=["sex", "embarked"], drop_first=True)
num = ["age", "fare_capped", "family_size"]
model_df[num] = StandardScaler().fit_transform(model_df[num])

print("\nMissing after cleaning:", int(clean.isnull().sum().sum()))
print("Final shape:", clean.shape, "| model-ready shape:", model_df.shape)

# ---- Before / after plots ----
fig, ax = plt.subplots(1, 3, figsize=(14, 4))
sns.heatmap(df.isnull(), cbar=False, yticklabels=False, ax=ax[0]); ax[0].set_title("Missing values (before)")
sns.boxplot(x=clean["fare"], ax=ax[1]); ax[1].set_title("Fare before capping")
sns.boxplot(x=clean["fare_capped"], ax=ax[2]); ax[2].set_title("Fare after capping")
plt.tight_layout(); plt.savefig("figs/w1_cleaning.png", dpi=130)
clean.to_csv("data/titanic_clean.csv", index=False)
