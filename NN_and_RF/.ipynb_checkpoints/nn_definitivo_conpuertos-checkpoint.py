import os
# Usar backend JAX. Poner antes del import keras
os.environ["KERAS_BACKEND"] = "jax"
from jax.lib import xla_bridge  # pip install keras jax[cuda12]
#!/usr/bin/env python3
# nn_balanceo_auto_equilibrado_100ep_conpuertos.py
import ast, numpy as np, pandas as pd, matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from keras.models import Sequential
from keras.layers import Dense, Dropout
import seaborn as sns

# ---------- CONFIG ----------
CSV_PATH  = "ips_totales_volumen_processed.csv"
VOL_BIN   = "volume_distance_vector_binary"
PORT_BIN  = "port_distance_vector_binary"
VEC_LEN   = 24
INPUT_DIM = 96
SEED      = 42
# ------------------------------

def parse_vec(s: str, L=VEC_LEN) -> np.ndarray:
    """Convierte '[1 2 3]' → np.array(L) rellenando o recortando."""
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
df["label"] = df["etiqueta"].map({"Usuario_Final": 1,
                                  "No_Usuario_Final": 0})

df["vol_alaw"]  = df["volume_distance_vector_alaw"].apply(parse_vec)
df["vol_bin"]   = df[VOL_BIN].apply(parse_vec)
df["port_alaw"] = df["port_distance_vector_alaw"].apply(parse_vec)
df["port_bin"]  = df[PORT_BIN].apply(parse_vec)

# ---------- Balanceo automático ----------
df_pos = df[df.label == 1]
df_neg = df[df.label == 0]

n_samples = min(len(df_pos), len(df_neg))
df_bal = pd.concat([
    df_pos.sample(n_samples, random_state=SEED, replace=False),
    df_neg.sample(n_samples, random_state=SEED, replace=False)
]).sample(frac=1, random_state=SEED)

print(f"👉 Conjunto equilibrado: {n_samples} + {n_samples} = {len(df_bal)} muestras")

df_bal["features"] = df_bal.apply(
    lambda r: np.concatenate([r.vol_alaw, r.vol_bin, r.port_alaw, r.port_bin]),
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
    Dense(64, activation='relu'),
    Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

# ----- ENTRENAMIENTO 100 ÉPOCAS -----
history = model.fit(X_tr, y_tr,
                    epochs=100,
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

# ---------- 1) Guardar matriz de confusión ----------
plt.figure(figsize=(6,6))
sns.heatmap(cm,
            annot=True, fmt='d', cmap='Blues',
            xticklabels=['No Usuario', 'Usuario'],
            yticklabels=['No Usuario', 'Usuario'])
plt.title('Matriz de confusión')
plt.xlabel('Predicción'); plt.ylabel('Real')
plt.tight_layout()
plt.savefig('confusion_matrix_conpuertos.png', dpi=300)
plt.close()

# ---------- 2) Guardar curvas de entrenamiento ----------
plt.figure(figsize=(8,5))
plt.plot(history.history['loss'], label='Train loss')
plt.plot(history.history['val_loss'], label='Val loss')
plt.plot(history.history['accuracy'], label='Train acc')
plt.plot(history.history['val_accuracy'], label='Val acc')
plt.title('Evolución de pérdida y accuracy')
plt.xlabel('Época'); plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('training_curves_conpuertos.png', dpi=300)
plt.close()

print("\n✅ Imágenes guardadas:")
print("   • confusion_matrix_conpuertos.png")
print("   • training_curves_conpuertos.png")
