# %% [markdown]
# # ✏️ VisionDigits — Entregable 1: Datos y Machine Learning (Google Colab)
#
# **Sistema inteligente de reconocimiento de números escritos a mano mediante Machine Learning**
#
# Este cuaderno implementa el flujo completo de aprendizaje supervisado:
#
# 1. Carga y partición del dataset MNIST (54 000 entrenamiento / 6 000 validación / 10 000 prueba).
# 2. Exploración de datos (distribución de clases y cuadrículas de ejemplo).
# 3. Preprocesamiento (normalización y formato de tensor).
# 4. Arquitectura **CNN principal** y **red densa comparativa**.
# 5. Entrenamiento con Adam + *sparse categorical crossentropy* + *accuracy*.
# 6. Evaluación con Scikit-learn: Accuracy, Precision, Recall, F1, tiempo de inferencia y Matriz de Confusión.
# 7. Exportación del modelo, las métricas y las muestras limpias de cada dígito para la interfaz infantil.
#
# > **Recomendación:** activa la GPU en *Entorno de ejecución → Cambiar tipo de entorno de ejecución → T4 GPU*.
# > Con GPU el cuaderno completo tarda unos 3–5 minutos.

# %%
# =============================================================================
# 0. CONFIGURACIÓN GENERAL E IMPORTACIONES
# =============================================================================
import os
import json
import time
import shutil
from datetime import datetime

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
)

# --- Reproducibilidad: misma semilla para NumPy, Python y TensorFlow ---
SEMILLA = 42
keras.utils.set_random_seed(SEMILLA)

# --- Hiperparámetros ---
EPOCAS_CNN = 15            # máximo; EarlyStopping puede detener antes
EPOCAS_DENSA = 15
TAMANO_LOTE = 128
TAMANO_VALIDACION = 6_000  # se separa del conjunto de entrenamiento
USAR_AUMENTO_DATOS = True  # rotación/zoom/traslación leves -> generaliza mejor a trazos del lienzo
MUESTRAS_POR_DIGITO = 5    # variantes de trazo que se mostrarán a los niños

# --- Carpetas de salida (misma estructura que espera la app de Streamlit) ---
DIR_SALIDA = "visiondigits_artefactos"
DIR_MODELOS = os.path.join(DIR_SALIDA, "modelos")
DIR_RECURSOS = os.path.join(DIR_SALIDA, "recursos")
DIR_MUESTRAS = os.path.join(DIR_RECURSOS, "muestras")
DIR_FIGURAS = os.path.join(DIR_RECURSOS, "figuras")
for carpeta in (DIR_MODELOS, DIR_MUESTRAS, DIR_FIGURAS):
    os.makedirs(carpeta, exist_ok=True)

NOMBRES_CLASES = [str(d) for d in range(10)]

print(f"TensorFlow {tf.__version__} | Keras {keras.__version__}")
print("GPU disponible:", tf.config.list_physical_devices("GPU") or "No (se usará CPU)")


def guardar_figura(nombre: str) -> None:
    """Guarda la figura actual en la carpeta de figuras y la muestra en pantalla."""
    plt.tight_layout()
    plt.savefig(os.path.join(DIR_FIGURAS, nombre), dpi=120, bbox_inches="tight")
    plt.show()


# %% [markdown]
# ## 1. Preparación del dataset MNIST
#
# MNIST contiene **70 000 imágenes en escala de grises de 28 × 28 píxeles** de dígitos manuscritos (0–9).
# Keras lo entrega dividido en 60 000 imágenes de entrenamiento y 10 000 de prueba.
# De las 60 000 de entrenamiento separamos **6 000 para validación** (ajuste de hiperparámetros y
# *early stopping*). Las 10 000 de prueba **no se tocan** hasta la evaluación final.

# %%
(x_entreno_total, y_entreno_total), (x_prueba, y_prueba) = keras.datasets.mnist.load_data()
print("Entrenamiento original:", x_entreno_total.shape, "| Prueba:", x_prueba.shape)
print(f"Total de imágenes: {len(x_entreno_total) + len(x_prueba):,}")

# Partición estratificada manual (mezcla aleatoria reproducible)
rng = np.random.default_rng(SEMILLA)
indices = rng.permutation(len(x_entreno_total))
idx_val, idx_ent = indices[:TAMANO_VALIDACION], indices[TAMANO_VALIDACION:]

x_entreno, y_entreno = x_entreno_total[idx_ent], y_entreno_total[idx_ent]
x_val, y_val = x_entreno_total[idx_val], y_entreno_total[idx_val]

print(f"Entrenamiento: {len(x_entreno):,} | Validación: {len(x_val):,} | Prueba: {len(x_prueba):,}")
print("Rango de píxeles:", x_entreno.min(), "-", x_entreno.max(), "| tipo:", x_entreno.dtype)

# %% [markdown]
# ## 2. Exploración de datos (EDA)
#
# Revisamos si las clases están balanceadas y cómo lucen los trazos de cada dígito.

# %%
# --- 2.1 Distribución de categorías en cada partición ---
fig, ejes = plt.subplots(1, 3, figsize=(15, 4))
for eje, (titulo, etiquetas, color) in zip(
    ejes,
    [("Entrenamiento", y_entreno, "#6366F1"), ("Validación", y_val, "#F59E0B"), ("Prueba", y_prueba, "#10B981")],
):
    conteo = np.bincount(etiquetas, minlength=10)
    barras = eje.bar(range(10), conteo, color=color)
    eje.bar_label(barras, fontsize=8)
    eje.set_xticks(range(10))
    eje.set_title(f"{titulo} ({len(etiquetas):,} imágenes)")
    eje.set_xlabel("Dígito")
    eje.set_ylabel("Cantidad")
plt.suptitle("Distribución de categorías en MNIST", fontsize=14, fontweight="bold")
guardar_figura("distribucion_clases.png")

conteo_ent = np.bincount(y_entreno, minlength=10)
print(f"Clase menos frecuente: {conteo_ent.argmin()} ({conteo_ent.min()}) | "
      f"más frecuente: {conteo_ent.argmax()} ({conteo_ent.max()}) -> dataset aproximadamente balanceado")

# %%
# --- 2.2 Cuadrícula de ejemplos: 10 trazos distintos por cada dígito ---
fig, ejes = plt.subplots(10, 10, figsize=(10, 10))
for digito in range(10):
    ejemplos = np.where(y_entreno == digito)[0][:10]
    for columna, idx in enumerate(ejemplos):
        eje = ejes[digito, columna]
        eje.imshow(x_entreno[idx], cmap="gray")
        eje.axis("off")
        if columna == 0:
            eje.set_title(f"Dígito {digito}", fontsize=8, loc="left")
plt.suptitle("Ejemplos de trazos por dígito (cada fila es una clase)", fontsize=13, fontweight="bold")
guardar_figura("cuadricula_ejemplos.png")

# %%
# --- 2.3 Imagen promedio por dígito: muestra el "trazo típico" que aprende el modelo ---
fig, ejes = plt.subplots(1, 10, figsize=(15, 2))
for digito in range(10):
    ejes[digito].imshow(x_entreno[y_entreno == digito].mean(axis=0), cmap="magma")
    ejes[digito].set_title(str(digito))
    ejes[digito].axis("off")
plt.suptitle("Imagen promedio de cada clase", fontweight="bold")
guardar_figura("promedio_por_clase.png")

# %% [markdown]
# ## 3. Preprocesamiento
#
# * **Normalización:** los píxeles pasan de [0, 255] a [0, 1] para estabilizar el descenso de gradiente.
# * **Canal explícito:** la CNN espera tensores `(alto, ancho, canales)` → `(28, 28, 1)`.
#
# MNIST ya viene con el dígito **blanco sobre fondo negro, recortado y centrado por centro de masa en 28 × 28**.
# La app de Streamlit replica exactamente ese formato con lo que dibuja el niño.

# %%
def preparar(x: np.ndarray) -> np.ndarray:
    """Normaliza a [0, 1] y añade el canal de color: (n, 28, 28) -> (n, 28, 28, 1)."""
    return (x.astype("float32") / 255.0)[..., np.newaxis]


X_ent, X_val, X_prueba = preparar(x_entreno), preparar(x_val), preparar(x_prueba)
print("Forma de entrada:", X_ent.shape, "| min:", X_ent.min(), "| max:", X_ent.max())

# %% [markdown]
# ## 4. Arquitecturas
#
# ### 4.1 CNN principal
# | Bloque | Capa | Función |
# |---|---|---|
# | Entrada | `Input(28, 28, 1)` | Imagen en escala de grises |
# | (Opcional) | Aumento de datos | Rotación, zoom y traslación leves, **solo durante el entrenamiento** |
# | Bloque 1 | `Conv2D(32, 3×3) + MaxPooling` | Detecta **trazos básicos**: bordes, líneas, curvas |
# | Bloque 2 | `Conv2D(64, 3×3) + MaxPooling` | Combina trazos en **patrones complejos**: lazos, cruces, esquinas |
# | Clasificador | `Flatten → Dense(128) → Dropout` | Integra los patrones detectados |
# | Salida | `Dense(10, softmax)` | Probabilidad de cada dígito 0–9 |
#
# ### 4.2 Red densa comparativa (MLP)
# `Flatten(784) → Dense(128) → Dense(64) → Dense(10)`: trata la imagen como un vector de 784 números y
# **pierde la información espacial** (qué píxel está al lado de cuál).

# %%
def construir_cnn(usar_aumento: bool = USAR_AUMENTO_DATOS) -> keras.Model:
    """CNN principal de VisionDigits (API funcional con capas nombradas)."""
    entradas = keras.Input(shape=(28, 28, 1), name="entrada")
    x = entradas
    if usar_aumento:
        # Estas capas solo actúan en entrenamiento; en inferencia son la identidad.
        x = layers.RandomRotation(0.05, fill_mode="constant", name="aumento_rotacion")(x)
        x = layers.RandomZoom(0.10, fill_mode="constant", name="aumento_zoom")(x)
        x = layers.RandomTranslation(0.08, 0.08, fill_mode="constant", name="aumento_traslacion")(x)

    # Bloque 1: trazos básicos
    x = layers.Conv2D(32, (3, 3), activation="relu", padding="same", name="conv1_trazos_basicos")(x)
    x = layers.MaxPooling2D((2, 2), name="maxpool1")(x)            # 28x28 -> 14x14
    # Bloque 2: patrones complejos
    x = layers.Conv2D(64, (3, 3), activation="relu", padding="same", name="conv2_patrones_complejos")(x)
    x = layers.MaxPooling2D((2, 2), name="maxpool2")(x)            # 14x14 -> 7x7
    # Clasificador
    x = layers.Flatten(name="aplanado")(x)                          # 7*7*64 = 3136
    x = layers.Dense(128, activation="relu", name="densa_oculta")(x)
    x = layers.Dropout(0.4, name="dropout")(x)
    salidas = layers.Dense(10, activation="softmax", name="salida")(x)
    return keras.Model(entradas, salidas, name="VisionDigits_CNN")


def construir_densa() -> keras.Model:
    """Red neuronal densa sencilla (perceptrón multicapa) para la comparación."""
    entradas = keras.Input(shape=(28, 28, 1), name="entrada")
    x = layers.Flatten(name="aplanado")(entradas)                  # 784 valores
    x = layers.Dense(128, activation="relu", name="densa_1")(x)
    x = layers.Dense(64, activation="relu", name="densa_2")(x)
    salidas = layers.Dense(10, activation="softmax", name="salida")(x)
    return keras.Model(entradas, salidas, name="VisionDigits_Densa")


def compilar(modelo: keras.Model) -> keras.Model:
    """Adam + sparse categorical crossentropy (etiquetas enteras 0-9) + accuracy."""
    modelo.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return modelo


modelo_cnn = compilar(construir_cnn())
modelo_densa = compilar(construir_densa())
modelo_cnn.summary()
modelo_densa.summary()

# %% [markdown]
# ## 5. Entrenamiento
#
# * **EarlyStopping** detiene el entrenamiento si la precisión de validación deja de mejorar y
#   restaura los mejores pesos.
# * **ReduceLROnPlateau** reduce la tasa de aprendizaje cuando la pérdida de validación se estanca.

# %%
def entrenar(modelo: keras.Model, epocas: int) -> tuple[keras.callbacks.History, float]:
    """Entrena el modelo y devuelve el historial y el tiempo total en segundos."""
    callbacks = [
        keras.callbacks.EarlyStopping(monitor="val_accuracy", patience=3, restore_best_weights=True),
        keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2, min_lr=1e-5),
    ]
    inicio = time.perf_counter()
    historial = modelo.fit(
        X_ent, y_entreno,
        validation_data=(X_val, y_val),
        epochs=epocas,
        batch_size=TAMANO_LOTE,
        callbacks=callbacks,
        verbose=2,
    )
    return historial, time.perf_counter() - inicio


print("=== Entrenando CNN principal ===")
hist_cnn, t_ent_cnn = entrenar(modelo_cnn, EPOCAS_CNN)
print(f"CNN entrenada en {t_ent_cnn:.1f} s\n")

print("=== Entrenando red densa comparativa ===")
hist_densa, t_ent_densa = entrenar(modelo_densa, EPOCAS_DENSA)
print(f"Red densa entrenada en {t_ent_densa:.1f} s")

# %%
# --- Curvas de aprendizaje ---
fig, ejes = plt.subplots(1, 2, figsize=(13, 4))
for hist, nombre, color in [(hist_cnn, "CNN", "#6366F1"), (hist_densa, "Densa", "#F59E0B")]:
    ejes[0].plot(hist.history["accuracy"], color=color, label=f"{nombre} entrenamiento")
    ejes[0].plot(hist.history["val_accuracy"], color=color, linestyle="--", label=f"{nombre} validación")
    ejes[1].plot(hist.history["loss"], color=color, label=f"{nombre} entrenamiento")
    ejes[1].plot(hist.history["val_loss"], color=color, linestyle="--", label=f"{nombre} validación")
ejes[0].set_title("Precisión (accuracy) por época")
ejes[1].set_title("Pérdida (loss) por época")
for eje in ejes:
    eje.set_xlabel("Época")
    eje.grid(alpha=0.3)
    eje.legend(fontsize=8)
guardar_figura("curvas_entrenamiento.png")

# %% [markdown]
# ## 6. Evaluación con Scikit-learn (conjunto de prueba: 10 000 imágenes nunca vistas)
#
# * **Accuracy:** proporción total de aciertos.
# * **Precision (macro):** de todo lo que el modelo dijo que era "k", ¿cuánto era realmente "k"? (promedio de las 10 clases).
# * **Recall (macro):** de todos los "k" reales, ¿cuántos encontró el modelo?
# * **F1-score (macro):** media armónica de precision y recall.
# * **Tiempo de inferencia:** por imagen en lote (rendimiento) y por imagen individual (latencia real de la app).

# %%
def medir_latencia_individual(modelo: keras.Model, repeticiones: int = 100) -> float:
    """Latencia media (s) de predecir UNA imagen, tal como ocurre en la app al pulsar 'Reconocer'."""
    imagen = X_prueba[:1]
    for _ in range(5):                       # calentamiento (la 1.ª llamada compila el grafo)
        modelo(imagen, training=False)
    inicio = time.perf_counter()
    for i in range(repeticiones):
        modelo(X_prueba[i:i + 1], training=False)
    return (time.perf_counter() - inicio) / repeticiones


def evaluar(modelo: keras.Model, nombre: str, hist, t_entreno: float) -> dict:
    """Calcula todas las métricas del modelo sobre el conjunto de prueba."""
    modelo.predict(X_prueba[:256], verbose=0)              # calentamiento
    inicio = time.perf_counter()
    probabilidades = modelo.predict(X_prueba, batch_size=512, verbose=0)
    t_lote = time.perf_counter() - inicio
    y_pred = probabilidades.argmax(axis=1)

    acc = accuracy_score(y_prueba, y_pred)
    prec, rec, f1, _ = precision_recall_fscore_support(y_prueba, y_pred, average="macro")
    prec_c, rec_c, f1_c, soporte_c = precision_recall_fscore_support(y_prueba, y_pred, average=None)
    matriz = confusion_matrix(y_prueba, y_pred)
    latencia = medir_latencia_individual(modelo)

    print(f"\n{'=' * 60}\n {nombre}\n{'=' * 60}")
    print(f"Accuracy : {acc:.4f}")
    print(f"Precision: {prec:.4f} (macro)")
    print(f"Recall   : {rec:.4f} (macro)")
    print(f"F1-score : {f1:.4f} (macro)")
    print(f"Inferencia en lote : {t_lote / len(X_prueba) * 1000:.4f} ms/imagen ({t_lote:.2f} s para 10 000)")
    print(f"Inferencia individual: {latencia * 1000:.2f} ms/imagen")
    print(f"Parámetros entrenables: {modelo.count_params():,}")
    print("\nReporte por clase:\n", classification_report(y_prueba, y_pred, digits=4))

    return {
        "nombre": nombre,
        "accuracy": float(acc),
        "precision_macro": float(prec),
        "recall_macro": float(rec),
        "f1_macro": float(f1),
        "errores": int((y_pred != y_prueba).sum()),
        "parametros": int(modelo.count_params()),
        "tiempo_entrenamiento_s": float(t_entreno),
        "epocas_entrenadas": len(hist.history["loss"]),
        "tiempo_inferencia_lote_s_por_imagen": float(t_lote / len(X_prueba)),
        "tiempo_inferencia_individual_s": float(latencia),
        "reporte_por_clase": {
            str(d): {"precision": float(prec_c[d]), "recall": float(rec_c[d]),
                     "f1": float(f1_c[d]), "soporte": int(soporte_c[d])}
            for d in range(10)
        },
        "matriz_confusion": matriz.tolist(),
        "historial": {k: [float(v) for v in vals] for k, vals in hist.history.items() if k != "learning_rate"},
        "_probabilidades": probabilidades,   # uso interno (no se exporta)
        "_y_pred": y_pred,
    }


res_cnn = evaluar(modelo_cnn, "CNN principal", hist_cnn, t_ent_cnn)
res_densa = evaluar(modelo_densa, "Red densa comparativa", hist_densa, t_ent_densa)

objetivo_cumplido = res_cnn["accuracy"] > 0.98
print(f"\n🎯 Objetivo > 98 % en prueba: {'CUMPLIDO ✅' if objetivo_cumplido else 'NO cumplido ❌'} "
      f"({res_cnn['accuracy'] * 100:.2f} %)")

# %%
# --- 6.1 Matriz de confusión de ambos modelos ---
fig, ejes = plt.subplots(1, 2, figsize=(15, 6.5))
for eje, res, cmap in [(ejes[0], res_cnn, "Blues"), (ejes[1], res_densa, "Oranges")]:
    ConfusionMatrixDisplay(np.array(res["matriz_confusion"]), display_labels=NOMBRES_CLASES).plot(
        ax=eje, cmap=cmap, colorbar=False, values_format="d")
    eje.set_title(f"{res['nombre']} — accuracy {res['accuracy'] * 100:.2f} %")
    eje.set_xlabel("Predicción del modelo")
    eje.set_ylabel("Dígito real")
guardar_figura("matrices_confusion_comparadas.png")

# Matriz de la CNN sola (es la que muestra la app)
fig, eje = plt.subplots(figsize=(7.5, 6.5))
ConfusionMatrixDisplay(np.array(res_cnn["matriz_confusion"]), display_labels=NOMBRES_CLASES).plot(
    ax=eje, cmap="Blues", colorbar=True, values_format="d")
eje.set_title("Matriz de Confusión — CNN VisionDigits (10 000 imágenes de prueba)")
eje.set_xlabel("Predicción del modelo")
eje.set_ylabel("Dígito real")
plt.tight_layout()
plt.savefig(os.path.join(DIR_RECURSOS, "matriz_confusion.png"), dpi=130, bbox_inches="tight")
plt.show()

# %%
# --- 6.2 ¿Qué números se confunden entre sí? ---
def pares_confundidos(matriz: np.ndarray, top: int = 10) -> list[dict]:
    """Devuelve los pares (real -> predicho) con más errores, fuera de la diagonal."""
    m = np.array(matriz).copy()
    np.fill_diagonal(m, 0)
    pares = [{"real": int(i), "predicho": int(j), "casos": int(m[i, j])}
             for i in range(10) for j in range(10) if m[i, j] > 0]
    return sorted(pares, key=lambda p: -p["casos"])[:top]


confusiones_cnn = pares_confundidos(res_cnn["matriz_confusion"])
print("Pares más confundidos por la CNN (real -> predicho):")
for p in confusiones_cnn:
    print(f"  {p['real']} -> {p['predicho']}: {p['casos']} casos")

m = np.array(res_cnn["matriz_confusion"])
for a, b in [(3, 8), (4, 9), (7, 1), (5, 3)]:
    print(f"Confusión simétrica {a}<->{b}: {m[a, b] + m[b, a]} casos")

# %%
# --- 6.3 Galería de errores: ayuda a entender POR QUÉ se equivoca ---
idx_errores = np.where(res_cnn["_y_pred"] != y_prueba)[0]
n_mostrar = min(24, len(idx_errores))
if n_mostrar:
    fig, ejes = plt.subplots(3, 8, figsize=(14, 6))
    for eje in ejes.ravel():
        eje.axis("off")
    for eje, idx in zip(ejes.ravel(), idx_errores[:n_mostrar]):
        conf = res_cnn["_probabilidades"][idx].max()
        eje.imshow(x_prueba[idx], cmap="gray")
        eje.set_title(f"Real {y_prueba[idx]} → Pred {res_cnn['_y_pred'][idx]}\n({conf * 100:.0f} %)",
                      fontsize=8, color="crimson")
    plt.suptitle(f"Errores de la CNN ({len(idx_errores)} de 10 000)", fontweight="bold")
    guardar_figura("galeria_errores.png")

# %% [markdown]
# ## 7. Comparación CNN vs red densa y justificación técnica

# %%
print(f"{'Métrica':<34}{'CNN':>14}{'Densa':>14}")
print("-" * 62)
filas = [
    ("Accuracy (%)", res_cnn["accuracy"] * 100, res_densa["accuracy"] * 100, ".2f"),
    ("Precision macro (%)", res_cnn["precision_macro"] * 100, res_densa["precision_macro"] * 100, ".2f"),
    ("Recall macro (%)", res_cnn["recall_macro"] * 100, res_densa["recall_macro"] * 100, ".2f"),
    ("F1 macro (%)", res_cnn["f1_macro"] * 100, res_densa["f1_macro"] * 100, ".2f"),
    ("Errores en prueba", res_cnn["errores"], res_densa["errores"], "d"),
    ("Parámetros", res_cnn["parametros"], res_densa["parametros"], ","),
    ("Inferencia individual (ms)", res_cnn["tiempo_inferencia_individual_s"] * 1000,
     res_densa["tiempo_inferencia_individual_s"] * 1000, ".2f"),
    ("Entrenamiento (s)", res_cnn["tiempo_entrenamiento_s"], res_densa["tiempo_entrenamiento_s"], ".1f"),
]
for nombre, a, b, fmt in filas:
    print(f"{nombre:<34}{format(a, fmt):>14}{format(b, fmt):>14}")

reduccion = (1 - res_cnn["errores"] / max(res_densa["errores"], 1)) * 100
print(f"\n➡ La CNN comete {reduccion:.0f} % menos errores que la red densa.")

fig, eje = plt.subplots(figsize=(8, 4))
metricas_cmp = ["accuracy", "precision_macro", "recall_macro", "f1_macro"]
pos = np.arange(len(metricas_cmp))
eje.bar(pos - 0.2, [res_cnn[m] * 100 for m in metricas_cmp], 0.4, label="CNN", color="#6366F1")
eje.bar(pos + 0.2, [res_densa[m] * 100 for m in metricas_cmp], 0.4, label="Densa", color="#F59E0B")
eje.set_xticks(pos, ["Accuracy", "Precision", "Recall", "F1"])
eje.set_ylim(95, 100)
eje.set_ylabel("%")
eje.set_title("CNN vs red densa (conjunto de prueba)")
eje.legend()
guardar_figura("comparacion_modelos.png")

# %% [markdown]
# **Justificación técnica de la CNN:**
#
# 1. **Conectividad local:** cada filtro mira una ventana de 3×3 píxeles, igual que nuestro ojo detecta
#    un trazo, en vez de conectar los 784 píxeles con todas las neuronas.
# 2. **Pesos compartidos:** el mismo filtro recorre toda la imagen, así detecta una curva esté arriba o abajo.
#    Por eso tiene menos parámetros en las capas convolucionales que una capa densa equivalente.
# 3. **Jerarquía de rasgos:** la primera capa aprende bordes y líneas; la segunda los combina en lazos
#    (0, 6, 8, 9), cruces (4, 7) y curvas (2, 3, 5).
# 4. **Robustez:** el MaxPooling da tolerancia a pequeños desplazamientos, algo clave cuando los niños
#    dibujan el número un poco movido o inclinado en el lienzo.
#
# La red densa "aplana" la imagen y pierde la vecindad entre píxeles, por eso comete más errores.

# %% [markdown]
# ## 8. Exportación de artefactos para la aplicación Streamlit
#
# Se genera la carpeta `visiondigits_artefactos/` con esta estructura (cópiala dentro de `app/`):
# ```
# modelos/modelo_cnn.keras      <- modelo principal (formato nativo Keras 3)
# modelos/modelo_cnn.h5         <- copia en formato HDF5 (compatibilidad)
# recursos/metricas.json        <- métricas, historial y matriz de confusión
# recursos/matriz_confusion.png
# recursos/muestras/digito_D_I.png  <- 5 variantes limpias de cada dígito para los niños
# recursos/figuras/*.png        <- gráficas del análisis
# ```

# %%
# --- 8.1 Guardar modelos ---
modelo_cnn.save(os.path.join(DIR_MODELOS, "modelo_cnn.keras"))
try:
    modelo_cnn.save(os.path.join(DIR_MODELOS, "modelo_cnn.h5"))
except Exception as e:  # algunas versiones futuras pueden retirar HDF5
    print("Aviso: no se pudo guardar en .h5:", e)
print("Modelos guardados en", DIR_MODELOS)

# Verificación: el modelo recargado debe predecir igual que el original
recargado = keras.models.load_model(os.path.join(DIR_MODELOS, "modelo_cnn.keras"))
assert np.allclose(recargado.predict(X_prueba[:50], verbose=0),
                   modelo_cnn.predict(X_prueba[:50], verbose=0), atol=1e-5)
print("✅ Modelo recargado correctamente: predicciones idénticas")


# %%
# --- 8.2 Extraer muestras limpias y variadas de cada dígito ---
def seleccionar_muestras(digito: int, n: int = MUESTRAS_POR_DIGITO, candidatos: int = 80) -> np.ndarray:
    """
    1) Toma las imágenes de prueba de 'digito' que la CNN clasifica bien con mayor confianza (trazos limpios).
    2) Entre ellas escoge 'n' lo más DIFERENTES posible (muestreo del punto más lejano),
       para mostrar a los niños varias formas válidas de escribir el mismo número.
    """
    probs = res_cnn["_probabilidades"]
    idx = np.where((y_prueba == digito) & (res_cnn["_y_pred"] == digito))[0]
    mejores = idx[np.argsort(-probs[idx, digito])[:candidatos]]
    vectores = X_prueba[mejores].reshape(len(mejores), -1)
    elegidos = [0]
    while len(elegidos) < n:
        distancias = np.min([np.linalg.norm(vectores - vectores[e], axis=1) for e in elegidos], axis=0)
        distancias[elegidos] = -1
        elegidos.append(int(np.argmax(distancias)))
    return mejores[elegidos]


muestras_npz = {}
fig, ejes = plt.subplots(10, MUESTRAS_POR_DIGITO, figsize=(MUESTRAS_POR_DIGITO * 1.4, 14))
for digito in range(10):
    indices_muestra = seleccionar_muestras(digito)
    muestras_npz[f"digito_{digito}"] = x_prueba[indices_muestra]
    for i, idx in enumerate(indices_muestra):
        keras.utils.save_img(os.path.join(DIR_MUESTRAS, f"digito_{digito}_{i + 1}.png"),
                             x_prueba[idx][..., np.newaxis], scale=False)
        ejes[digito, i].imshow(255 - x_prueba[idx], cmap="gray")   # vista "tinta sobre papel"
        ejes[digito, i].axis("off")
plt.suptitle("Muestras de referencia para la interfaz infantil", fontweight="bold")
guardar_figura("muestras_referencia.png")
np.savez_compressed(os.path.join(DIR_MUESTRAS, "muestras.npz"), **muestras_npz)
print(f"Muestras guardadas: {len(os.listdir(DIR_MUESTRAS))} archivos")


# %%
# --- 8.3 Guardar el archivo de métricas ---
def limpiar(res: dict) -> dict:
    """Quita los campos internos (arrays) antes de serializar a JSON."""
    return {k: v for k, v in res.items() if not k.startswith("_")}


metricas = {
    "proyecto": "VisionDigits",
    "fecha_entrenamiento": datetime.now().strftime("%Y-%m-%d %H:%M"),
    "version_tensorflow": tf.__version__,
    "version_keras": keras.__version__,
    "modelo_principal": "cnn",
    "objetivo_98_cumplido": bool(objetivo_cumplido),
    "hiperparametros": {
        "optimizador": "Adam (lr=0.001)",
        "perdida": "sparse_categorical_crossentropy",
        "metrica": "accuracy",
        "tamano_lote": TAMANO_LOTE,
        "aumento_datos": USAR_AUMENTO_DATOS,
        "semilla": SEMILLA,
    },
    "dataset": {
        "nombre": "MNIST",
        "total": int(len(x_entreno_total) + len(x_prueba)),
        "entrenamiento": int(len(x_entreno)),
        "validacion": int(len(x_val)),
        "prueba": int(len(x_prueba)),
        "dimensiones": [28, 28],
        "distribucion_entrenamiento": np.bincount(y_entreno, minlength=10).tolist(),
        "distribucion_validacion": np.bincount(y_val, minlength=10).tolist(),
        "distribucion_prueba": np.bincount(y_prueba, minlength=10).tolist(),
    },
    "modelos": {"cnn": limpiar(res_cnn), "densa": limpiar(res_densa)},
    "matriz_confusion": res_cnn["matriz_confusion"],
    "pares_mas_confundidos": confusiones_cnn,
}
with open(os.path.join(DIR_RECURSOS, "metricas.json"), "w", encoding="utf-8") as f:
    json.dump(metricas, f, ensure_ascii=False, indent=2)
print("✅ metricas.json guardado")

# %%
# --- 8.4 Comprimir y descargar ---
ruta_zip = shutil.make_archive("visiondigits_artefactos", "zip", DIR_SALIDA)
print("Archivo listo:", ruta_zip)
try:
    from google.colab import files  # solo existe dentro de Colab
    files.download(ruta_zip)
except ImportError:
    print("No estás en Colab: el zip quedó en la carpeta actual.")

# %% [markdown]
# ## ✅ Siguiente paso
# Descomprime `visiondigits_artefactos.zip` **dentro de la carpeta `app/`** del proyecto
# (deben quedar `app/modelos/` y `app/recursos/`) y ejecuta:
# ```
# streamlit run app/app.py
# ```
