# --- backend JAX (opcional) ---
import os
os.environ["KERAS_BACKEND"] = "jax"
from jax.lib import xla_bridge          # pip install keras jax[cuda12]

#!/usr/bin/env python3
#  – red densa 48 features  + imágenes de salida

import ast, numpy as np, pandas as pd
import matplotlib.pyplot as plt, seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from keras.models import Sequential
from keras.layers import Dense, Dropout



# ---------- CONFIG ----------
CSV_PATH  = "IPs_totales_alaw_with_binary.csv"
VEC_LEN   = 24
INPUT_DIM = 48
EPOCHS    = 100
SEED      = 42
# -----------------------------

def parse_vec(s: str, L: int = VEC_LEN) -> np.ndarray:
    """Convierte cadena con números en vector de longitud fija."""
    try:
        s = s.strip("[]").replace("  ", " ").replace(" ", ",")
        v = np.array(ast.literal_eval(f"[{s}]"), dtype=float)
        if len(v) < L: v = np.pad(v, (0, L-len(v)))
        elif len(v) > L: v = v[:L]
        return v
    except Exception:
        return np.zeros(L)

# ---------- Carga ----------
df = pd.read_csv(CSV_PATH)

# Ajusta si la columna etiqueta tiene otro nombre/valores
df["label"] = df["etiqueta"].map({"Usuario_Final": 1,
                                  "No_Usuario_Final": 0})

df["vec_alaw"]   = df["diferencias_alaw_str"].apply(parse_vec)
df["vec_binary"] = df["binary_vector_str"].apply(parse_vec)

# ---------- Balanceo (todas + mismas negativas) ----------
df_pos = df[df.label == 1]
df_neg = df[df.label == 0]

n_samples = min(len(df_pos), len(df_neg))
df_bal = pd.concat([
    df_pos,
    df_neg.sample(n_samples, random_state=SEED, replace=False)
]).sample(frac=1, random_state=SEED)

print(f"👉 Muestras usadas: {len(df_pos)} positivas + {n_samples} negativas = {len(df_bal)}")

df_bal["features"] = df_bal.apply(
    lambda r: np.concatenate([r.vec_alaw, r.vec_binary]),
    axis=1
)

X = np.vstack(df_bal.features.values).astype("float32")
y = df_bal.label.values.astype("int32")

X_tr, X_te, y_tr, y_te = train_test_split(
    X, y, stratify=y, test_size=0.3, random_state=SEED
)

# ---------- Modelo ----------
model = Sequential([
    Dense(128, activation='relu', input_shape=(INPUT_DIM,)),
    Dropout(0.30),
    Dense(256, activation='relu'),
    Dropout(0.30),
    Dense(512, activation='relu'),
    Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

history = model.fit(X_tr, y_tr,
                    epochs=EPOCHS,
                    batch_size=32,
                    validation_split=0.2,
                    verbose=1)

# ---------- Evaluación ----------
loss, acc = model.evaluate(X_te, y_te, verbose=0)
print(f"\n🔍 Acc test: {acc*100:.2f}%")

y_pred = (model.predict(X_te, verbose=0) >= 0.5).astype(int).ravel()
cm = confusion_matrix(y_te, y_pred)
print("\n📊 Matriz de confusión:\n", cm)
print("\n📋 Reporte:\n", classification_report(y_te, y_pred))

# ---------- Guardar matriz de confusión ----------
plt.figure(figsize=(6,6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['No Usuario', 'Usuario'],
            yticklabels=['No Usuario', 'Usuario'])
plt.title('Matriz de confusión')
plt.xlabel('Predicción'); plt.ylabel('Real')
plt.tight_layout()
plt.savefig('confusion_matrix_alawbinary.png', dpi=300)
plt.close()

# ---------- Guardar curvas de entrenamiento ----------
plt.figure(figsize=(8,5))
plt.plot(history.history['loss'], label='Train loss')
plt.plot(history.history['val_loss'], label='Val loss')
plt.plot(history.history['accuracy'], label='Train acc')
plt.plot(history.history['val_accuracy'], label='Val acc')
plt.title('Evolución pérdida y accuracy')
plt.xlabel('Época'); plt.legend(); plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('training_curves_alawbinary.png', dpi=300)
plt.close()

print("\n✅ Imágenes guardadas:")
print("   • confusion_matrix_alawbinary.png")
print("   • training_curves_alawbinary.png")
