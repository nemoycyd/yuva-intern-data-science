# Week 4: Supervised Learning - Classification (Dataset: Titanic, predict survival)
import pandas as pd, numpy as np
import matplotlib.pyplot as plt, seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, StratifiedKFold
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, classification_report, RocCurveDisplay

df = pd.read_csv("data/titanic_clean.csv")
num = ["age", "fare_capped", "family_size", "pclass"]; cat = ["sex", "embarked", "is_alone"]
X, y = df[num + cat], df["survived"]            # 'alive' etc. excluded -> no target leakage
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
print("Train:", Xtr.shape, "Test:", Xte.shape, "| survival rate train/test: %.2f / %.2f" % (ytr.mean(), yte.mean()))

pre = ColumnTransformer([("num", StandardScaler(), num), ("cat", OneHotEncoder(drop="first"), cat)])
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Decision Tree": DecisionTreeClassifier(max_depth=4, random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=300, random_state=42),
}
cv = StratifiedKFold(5, shuffle=True, random_state=42)
rows = []
for name, m in models.items():
    pipe = Pipeline([("pre", pre), ("m", m)])
    cvs = cross_val_score(pipe, Xtr, ytr, cv=cv, scoring="accuracy")
    pipe.fit(Xtr, ytr); p = pipe.predict(Xte); pr = pipe.predict_proba(Xte)[:, 1]
    rows.append([name, cvs.mean(), accuracy_score(yte, p), precision_score(yte, p), recall_score(yte, p), f1_score(yte, p), roc_auc_score(yte, pr)])
res = pd.DataFrame(rows, columns=["Model", "CV acc", "Test acc", "Precision", "Recall", "F1", "ROC-AUC"]).round(3)
print("\n", res.to_string(index=False))

# Hyper-parameter tuning on Random Forest
pipe = Pipeline([("pre", pre), ("m", RandomForestClassifier(random_state=42))])
grid = {"m__n_estimators": [200, 400], "m__max_depth": [4, 6, 8, None], "m__min_samples_leaf": [1, 3, 5]}
gs = GridSearchCV(pipe, grid, cv=cv, scoring="roc_auc", n_jobs=-1).fit(Xtr, ytr)
best = gs.best_estimator_
p = best.predict(Xte); pr = best.predict_proba(Xte)[:, 1]
print("\nBest params:", gs.best_params_, "| CV AUC: %.3f" % gs.best_score_)
print("Tuned RF  Test acc: %.3f  F1: %.3f  AUC: %.3f" % (accuracy_score(yte, p), f1_score(yte, p), roc_auc_score(yte, pr)))
print("\n", classification_report(yte, p, target_names=["Died", "Survived"]))

# Feature importance
names = best.named_steps["pre"].get_feature_names_out()
imp = pd.Series(best.named_steps["m"].feature_importances_, index=names).sort_values(ascending=False)
print("Top features:\n", imp.round(3).head(6))

fig, ax = plt.subplots(1, 3, figsize=(17, 4.5))
sns.heatmap(confusion_matrix(yte, p), annot=True, fmt="d", cmap="Blues", xticklabels=["Died", "Survived"], yticklabels=["Died", "Survived"], ax=ax[0])
ax[0].set_title("Confusion matrix (tuned RF)"); ax[0].set_xlabel("Predicted"); ax[0].set_ylabel("Actual")
for name, m in models.items():
    pp = Pipeline([("pre", pre), ("m", m)]).fit(Xtr, ytr)
    RocCurveDisplay.from_estimator(pp, Xte, yte, name=name, ax=ax[1])
ax[1].set_title("ROC curves"); ax[1].plot([0, 1], [0, 1], "k--", alpha=.4)
imp.head(8)[::-1].plot.barh(ax=ax[2], color="seagreen"); ax[2].set_title("Feature importance")
plt.tight_layout(); plt.savefig("figs/w4_models.png", dpi=130); plt.close()
