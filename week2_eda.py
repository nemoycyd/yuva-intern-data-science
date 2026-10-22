# Week 2: Exploratory Data Analysis and Visualization (Dataset: cleaned Titanic from Week 1)
import pandas as pd, numpy as np
import matplotlib.pyplot as plt, seaborn as sns
sns.set_theme(style="whitegrid")

df = pd.read_csv("data/titanic_clean.csv")
print("Shape:", df.shape)
print(df[["age", "fare", "family_size", "survived"]].describe().round(2))
print("\nOverall survival rate: %.1f%%" % (df.survived.mean() * 100))
for col in ["sex", "pclass", "embarked", "is_alone"]:
    print(f"\nSurvival rate by {col}:\n", (df.groupby(col)["survived"].mean() * 100).round(1))
print("\nSkewness:\n", df[["age", "fare", "family_size"]].skew().round(2))
corr = df[["survived", "pclass", "age", "sibsp", "parch", "fare", "family_size"]].corr()
print("\nCorrelation with survived:\n", corr["survived"].round(2))

# Fig 1: distributions
fig, ax = plt.subplots(1, 3, figsize=(15, 4))
sns.histplot(df.age, kde=True, ax=ax[0], color="steelblue"); ax[0].set_title("Age distribution")
sns.histplot(df.fare, bins=40, ax=ax[1], color="darkorange"); ax[1].set_title("Fare distribution (right-skewed)")
sns.countplot(x="pclass", hue="pclass", data=df, ax=ax[2], palette="viridis", legend=False); ax[2].set_title("Passengers per class")
plt.tight_layout(); plt.savefig("figs/w2_distributions.png", dpi=130); plt.close()

# Fig 2: survival relationships
fig, ax = plt.subplots(1, 3, figsize=(15, 4))
sns.barplot(x="sex", y="survived", data=df, ax=ax[0], errorbar=None); ax[0].set_title("Survival rate by sex")
sns.barplot(x="pclass", y="survived", hue="sex", data=df, ax=ax[1], errorbar=None); ax[1].set_title("Survival by class and sex")
sns.boxplot(x="survived", y="age", data=df, ax=ax[2]); ax[2].set_title("Age vs survival")
plt.tight_layout(); plt.savefig("figs/w2_survival.png", dpi=130); plt.close()

# Fig 3: correlation heatmap and age/fare scatter
fig, ax = plt.subplots(1, 2, figsize=(13, 5))
sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0, ax=ax[0]); ax[0].set_title("Correlation matrix")
sns.scatterplot(x="age", y="fare_capped", hue="survived", data=df, alpha=.6, ax=ax[1]); ax[1].set_title("Age vs fare by survival")
plt.tight_layout(); plt.savefig("figs/w2_correlation.png", dpi=130); plt.close()
