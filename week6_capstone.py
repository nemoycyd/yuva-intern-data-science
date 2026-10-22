# Week 6: Capstone - End-to-end Data Science Pipeline (Dataset: Diamonds, predict price)
import pandas as pd, numpy as np
import matplotlib.pyplot as plt, seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.cluster import KMeans
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, silhouette_score
sns.set_theme(style="whitegrid")

# 1. Acquisition
df = pd.read_csv("https://raw.githubusercontent.com/mwaskom/seaborn-data/master/diamonds.csv")  # or data/diamonds.csv
print("Raw shape:", df.shape, "| missing:", int(df.isnull().sum().sum()), "| duplicates:", df.duplicated().sum())

# 2. Cleaning
bad = ((df.x == 0) | (df.y == 0) | (df.z == 0)).sum(); print("Rows with zero dimension (impossible):", bad)
df = df[(df.x > 0) & (df.y > 0) & (df.z > 0)]
df = df[(df.y < 20) & (df.z < 10) & (df.table < 90)]           # obvious measurement errors
df = df.drop_duplicates().reset_index(drop=True)
print("Clean shape:", df.shape)

# 3. EDA
print("\nPrice summary:\n", df.price.describe().round(0))
print("Price skewness: %.2f -> log skewness: %.2f" % (df.price.skew(), np.log(df.price).skew()))
print("\nCorrelation with price:\n", df[["carat", "depth", "table", "x", "y", "z", "price"]].corr()["price"].round(2))
print("\nMedian price by cut:\n", df.groupby("cut", observed=True)["price"].median())

# 4. Feature engineering
df["volume"] = df.x * df.y * df.z
df["log_price"] = np.log(df.price)
df["log_carat"] = np.log(df.carat); df["log_volume"] = np.log(df.volume)   # fixes skew + makes price ~ linear
cut_o = ["Fair", "Good", "Very Good", "Premium", "Ideal"]; col_o = list("JIHGFED"); cla_o = ["I1", "SI2", "SI1", "VS2", "VS1", "VVS2", "VVS1", "IF"]
num = ["log_carat", "depth", "table", "log_volume"]; cat = ["cut", "color", "clarity"]
pre = ColumnTransformer([("num", StandardScaler(), num), ("cat", OrdinalEncoder(categories=[cut_o, col_o, cla_o]), cat)])
X, y = df[num + cat], df["log_price"]
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42)

# 5. Supervised modelling (target = log price, metrics reported in dollars)
models = {"Linear Regression": LinearRegression(),
          "Random Forest": RandomForestRegressor(n_estimators=150, min_samples_leaf=2, n_jobs=-1, random_state=42),
          "Gradient Boosting": HistGradientBoostingRegressor(max_iter=400, learning_rate=0.08, random_state=42)}
rows, preds = [], {}
for n, m in models.items():
    pipe = Pipeline([("pre", pre), ("m", m)]).fit(Xtr, ytr)
    p = np.exp(pipe.predict(Xte)); preds[n] = p; t = np.exp(yte)
    cv = cross_val_score(Pipeline([("pre", pre), ("m", m)]), Xtr, ytr, cv=3, scoring="r2").mean()
    rows.append([n, cv, r2_score(t, p), mean_absolute_error(t, p), np.sqrt(mean_squared_error(t, p))])
res = pd.DataFrame(rows, columns=["Model", "CV R2 (log)", "Test R2 ($)", "MAE ($)", "RMSE ($)"]).round(3)
print("\n", res.to_string(index=False))
best_name = res.sort_values("CV R2 (log)").iloc[-1]["Model"]; print("Best model (chosen by cross-validation, not test score):", best_name)

# 6. Unsupervised: market segments
Z = StandardScaler().fit_transform(df[["carat", "volume", "price"]].assign(price=np.log(df.price)))
sample = np.random.RandomState(1).choice(len(Z), 5000, replace=False)
for k in [2, 3, 4, 5]:
    km = KMeans(k, n_init=10, random_state=42).fit(Z); print(f"k={k} silhouette={silhouette_score(Z[sample], km.labels_[sample]):.3f}")
km = KMeans(3, n_init=10, random_state=42).fit(Z); df["segment"] = km.labels_
print("\nSegment profile:\n", df.groupby("segment").agg(n=("price", "size"), carat=("carat", "mean"), price=("price", "median")).round(2))

# 7. Plots
fig, ax = plt.subplots(2, 2, figsize=(13, 9))
sns.histplot(df.price, bins=60, ax=ax[0, 0]); ax[0, 0].set_title("Price (right-skewed)")
sns.histplot(df.log_price, bins=60, ax=ax[0, 1], color="darkorange"); ax[0, 1].set_title("log(Price) is closer to normal")
sns.scatterplot(x="carat", y="price", data=df.sample(5000, random_state=1), hue="clarity", alpha=.5, ax=ax[1, 0]); ax[1, 0].set_title("Carat vs price by clarity")
sns.boxplot(x="cut", y="price", data=df, order=cut_o, ax=ax[1, 1]); ax[1, 1].set_title("Price by cut")
plt.tight_layout(); plt.savefig("figs/w6_eda.png", dpi=120); plt.close()

pipe = Pipeline([("pre", pre), ("m", models[best_name])]).fit(Xtr, ytr)
fig, ax = plt.subplots(1, 2, figsize=(13, 4.8))
t = np.exp(yte); ax[0].scatter(t, preds[best_name], s=3, alpha=.3); ax[0].plot([0, 19000], [0, 19000], "r--"); ax[0].set_title(f"{best_name}: actual vs predicted"); ax[0].set_xlabel("Actual $"); ax[0].set_ylabel("Predicted $")
from sklearn.inspection import permutation_importance
pi = permutation_importance(pipe, Xte.sample(3000, random_state=1), yte.loc[Xte.sample(3000, random_state=1).index], n_repeats=5, random_state=42)
imp = pd.Series(pi.importances_mean, index=num + cat).sort_values(); imp.plot.barh(ax=ax[1], color="seagreen"); ax[1].set_title("Permutation importance")
print("\nPermutation importance:\n", imp.sort_values(ascending=False).round(3))
plt.tight_layout(); plt.savefig("figs/w6_model.png", dpi=120); plt.close()
