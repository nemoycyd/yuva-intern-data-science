# Week 5: Deep Learning Application with PyTorch (Dataset: sklearn handwritten digits, 8x8 images, 10 classes)
import numpy as np, torch, torch.nn as nn
import matplotlib.pyplot as plt, seaborn as sns
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

torch.manual_seed(42); np.random.seed(42)
digits = load_digits(); X, y = digits.data, digits.target
print("Images:", X.shape, "| classes:", np.unique(y))
Xtr, Xtmp, ytr, ytmp = train_test_split(X, y, test_size=0.4, stratify=y, random_state=42)
Xval, Xte, yval, yte = train_test_split(Xtmp, ytmp, test_size=0.5, stratify=ytmp, random_state=42)
sc = StandardScaler().fit(Xtr)                              # fit on train only
T = lambda a, dt=torch.float32: torch.tensor(a, dtype=dt)
Xtr_t, Xval_t, Xte_t = T(sc.transform(Xtr)), T(sc.transform(Xval)), T(sc.transform(Xte))
ytr_t, yval_t, yte_t = T(ytr, torch.long), T(yval, torch.long), T(yte, torch.long)
print("Train/Val/Test:", len(Xtr), len(Xval), len(Xte))

def make_model(dropout):
    return nn.Sequential(nn.Linear(64, 128), nn.ReLU(), nn.Dropout(dropout),
                         nn.Linear(128, 64), nn.ReLU(), nn.Dropout(dropout), nn.Linear(64, 10))

def train(dropout, weight_decay, epochs=60, lr=1e-3, patience=None):
    model = make_model(dropout); opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    lossf = nn.CrossEntropyLoss(); hist = {"tl": [], "vl": [], "ta": [], "va": []}
    best, best_state, wait = 1e9, None, 0
    for ep in range(epochs):
        model.train(); perm = torch.randperm(len(Xtr_t))
        for i in range(0, len(perm), 32):
            idx = perm[i:i + 32]; opt.zero_grad()
            loss = lossf(model(Xtr_t[idx]), ytr_t[idx]); loss.backward(); opt.step()
        model.eval()
        with torch.no_grad():
            lt, lv = model(Xtr_t), model(Xval_t)
            hist["tl"].append(lossf(lt, ytr_t).item()); hist["vl"].append(lossf(lv, yval_t).item())
            hist["ta"].append((lt.argmax(1) == ytr_t).float().mean().item()); hist["va"].append((lv.argmax(1) == yval_t).float().mean().item())
        if patience:                                        # early stopping on validation loss
            if hist["vl"][-1] < best - 1e-4: best, wait = hist["vl"][-1], 0; best_state = {k: v.clone() for k, v in model.state_dict().items()}
            else:
                wait += 1
                if wait >= patience: break
    if best_state: model.load_state_dict(best_state)
    return model, hist

base, h_base = train(dropout=0.0, weight_decay=0.0)                      # no regularisation
reg, h_reg = train(dropout=0.3, weight_decay=1e-4, patience=8)           # dropout + L2 + early stopping
for name, m in [("Baseline (no regularisation)", base), ("Regularised (dropout+L2+early stop)", reg)]:
    m.eval()
    with torch.no_grad(): p = m(Xte_t).argmax(1).numpy()
    print(f"{name}: test accuracy = {accuracy_score(yte, p):.4f}")
print("Epochs run (regularised):", len(h_reg["tl"]))
print("Baseline final train/val loss: %.4f / %.4f" % (h_base["tl"][-1], h_base["vl"][-1]))
reg.eval()
with torch.no_grad(): p = reg(Xte_t).argmax(1).numpy()
print("\n", classification_report(yte, p, digits=3))
print("Min val loss  baseline: %.4f | regularised: %.4f" % (min(h_base["vl"]), min(h_reg["vl"])))
print("Parameters:", sum(p.numel() for p in reg.parameters()))

fig, ax = plt.subplots(1, 3, figsize=(17, 4.5))
ax[0].plot(h_base["tl"], label="train"); ax[0].plot(h_base["vl"], label="validation"); ax[0].set_title("Baseline: loss"); ax[0].legend(); ax[0].set_xlabel("epoch")
ax[1].plot(h_reg["tl"], label="train"); ax[1].plot(h_reg["vl"], label="validation"); ax[1].set_title("Regularised: loss"); ax[1].legend(); ax[1].set_xlabel("epoch")
sns.heatmap(confusion_matrix(yte, p), annot=True, fmt="d", cmap="Blues", ax=ax[2]); ax[2].set_title("Confusion matrix (test)")
plt.tight_layout(); plt.savefig("figs/w5_training.png", dpi=130); plt.close()
fig, ax = plt.subplots(1, 8, figsize=(14, 2.2))
for i, a in enumerate(ax): a.imshow(digits.images[i], cmap="gray_r"); a.set_title(digits.target[i]); a.axis("off")
plt.tight_layout(); plt.savefig("figs/w5_samples.png", dpi=130); plt.close()
