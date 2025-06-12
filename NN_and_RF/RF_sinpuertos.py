#!/usr/bin/env python3
# rf_sinpuertos_savepng.py
import os, ast, time, numpy as np, pandas as pd, matplotlib.pyplot as plt, seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, confusion_matrix,
                             classification_report, roc_curve, auc)
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils import shuffle

# ---------- CONFIG ----------
CSV_PATH  = "ips_totales_volumen_processed.csv"
VOL_BIN   = "volume_distance_vector_binary"
VEC_LEN   = 24
N_ESTIM   = 400
SEED      = 42
# -----------------------------

def parse_vec(s, L=VEC_LEN):
    try:
        s = s.strip("[]").replace("  ", " ").replace(" ", ",")
        v = np.array(ast.literal_eval(f"[{s}]"), dtype=float)
        if len(v) < L: v = np.pad(v, (0, L-len(v)))
        elif len(v) > L: v = v[:L]
        return v
    except Exception:
        return np.zeros(L)

print("⏳ Leyendo CSV…")
df = pd.read_csv(CSV_PATH)
df["label"] = df["etiqueta"].map({"Usuario_Final":1,"No_Usuario_Final":0})

df["vol_alaw"] = df["volume_distance_vector_alaw"].apply(parse_vec)
df["vol_bin"]  = df[VOL_BIN].apply(parse_vec)

df["features"] = df.apply(lambda r:
            np.concatenate([r.vol_alaw, r.vol_bin]), axis=1)

X = np.vstack(df.features.values)
y = df.label.values
X, y = shuffle(X, y, random_state=SEED)

X_tr, X_te, y_tr, y_te = train_test_split(
    X, y, stratify=y, test_size=0.3, random_state=SEED)

rf = RandomForestClassifier(
        n_estimators=N_ESTIM, class_weight="balanced_subsample",
        max_features="sqrt", n_jobs=-1, random_state=SEED)

print("🌳 Entrenando RF sin puertos…")
t0 = time.time(); rf.fit(X_tr, y_tr)
print(f"⏱️ {time.time()-t0:.1f} s")

# ---------- Métricas ----------
proba = rf.predict_proba(X_te)[:,1]
y_pred = (proba>=0.5).astype(int)

acc = accuracy_score(y_te, y_pred)
print(f"\n🔍 Accuracy: {acc*100:.2f}%")
print("\n📊 Confusión:\n", confusion_matrix(y_te, y_pred))
print("\n📋 Reporte:\n", classification_report(y_te, y_pred))

# ----- Guardar matriz de confusión -----
cm = confusion_matrix(y_te, y_pred)
plt.figure(figsize=(6,6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Purples',
            xticklabels=['No Usuario','Usuario'],
            yticklabels=['No Usuario','Usuario'])
plt.title('RF sin puertos – Matriz de confusión')
plt.xlabel('Predicción'); plt.ylabel('Real')
plt.tight_layout(); plt.savefig('confusion_matrix_rf_sinpuertos.png', dpi=300); plt.close()

# ----- ROC curve -----
fpr, tpr, _ = roc_curve(y_te, proba)
roc_auc = auc(fpr, tpr)
plt.figure(figsize=(6,5))
plt.plot(fpr, tpr, label=f'ROC AUC = {roc_auc:.3f}')
plt.plot([0,1],[0,1],'k--',alpha=0.4)
plt.xlabel('FPR'); plt.ylabel('TPR'); plt.title('RF sin puertos – ROC')
plt.legend(); plt.grid(alpha=0.3); plt.tight_layout()
plt.savefig('roc_curve_rf_sinpuertos.png', dpi=300); plt.close()

print("\n✅ Imágenes guardadas: confusion_matrix_rf_sinpuertos.png / roc_curve_rf_sinpuertos.png")
