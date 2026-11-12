"""
modelo.py — Módulos 3 y 4: Modelo predictivo y clasificación.

* Carga UNA sola vez la CNN entrenada en Colab (st.cache_resource), sin reentrenar.
* Predice las 10 puntuaciones de salida, elige el dígito ganador y la segunda opción.
* Mide el tiempo de inferencia en segundos.
* Expone un "extractor de rasgos" (capa densa oculta) para comparar el dibujo del niño
  con las muestras de referencia en el espacio que aprendió la red.
"""
from __future__ import annotations

import os
import time
from dataclasses import dataclass

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")   # silencia logs de TensorFlow

import numpy as np
import streamlit as st

import config


@dataclass
class Prediccion:
    """Resultado de clasificar una imagen."""
    probabilidades: np.ndarray   # (10,) suma 1
    digito: int                  # clase con mayor puntuación
    confianza: float             # probabilidad de esa clase
    segundo: int                 # segunda clase más probable
    confianza_segundo: float
    tiempo_s: float              # tiempo de inferencia (segundos)

    @property
    def margen(self) -> float:
        """Diferencia entre la 1.ª y la 2.ª opción: cuanto mayor, más segura la decisión."""
        return self.confianza - self.confianza_segundo


@st.cache_resource(show_spinner="Cargando la red neuronal… 🧠")
def cargar_modelo():
    """
    Carga el modelo exportado desde Colab. Intenta primero el formato nativo .keras
    y después el respaldo .h5. Devuelve (modelo, mensaje_error).
    """
    from tensorflow import keras

    errores = []
    for ruta in (config.RUTA_MODELO_KERAS, config.RUTA_MODELO_H5):
        if not ruta.exists():
            errores.append(f"No existe {ruta.name}")
            continue
        try:
            modelo = keras.models.load_model(ruta, compile=False)
            # Calentamiento: la primera llamada construye el grafo y es más lenta;
            # así el tiempo que ve el usuario refleja la inferencia real.
            modelo(np.zeros((1, 28, 28, 1), dtype=np.float32), training=False)
            return modelo, None
        except Exception as e:  # versión de Keras incompatible, archivo dañado, etc.
            errores.append(f"{ruta.name}: {e}")
    return None, " | ".join(errores)


@st.cache_resource(show_spinner=False)
def crear_extractor(_modelo):
    """
    Devuelve un sub-modelo que entrega la activación de la capa 'densa_oculta'
    (128 rasgos). Si el modelo no tiene esa capa, devuelve None.
    El guion bajo en '_modelo' le indica a Streamlit que no intente hashearlo.
    """
    from tensorflow import keras
    try:
        capa = _modelo.get_layer("densa_oculta")
        return keras.Model(inputs=_modelo.input, outputs=capa.output)
    except Exception:
        return None


def _softmax(x: np.ndarray) -> np.ndarray:
    e = np.exp(x - x.max())
    return e / e.sum()


def predecir(modelo, tensor: np.ndarray) -> Prediccion:
    """Clasifica un tensor (1, 28, 28, 1) y mide el tiempo de inferencia."""
    inicio = time.perf_counter()
    salida = np.asarray(modelo(tensor, training=False))[0]
    tiempo = time.perf_counter() - inicio

    # Si el modelo devolviera logits en lugar de probabilidades, se normalizan.
    if salida.min() < 0 or not np.isclose(salida.sum(), 1.0, atol=1e-3):
        salida = _softmax(salida)

    orden = np.argsort(salida)[::-1]
    return Prediccion(
        probabilidades=salida.astype(float),
        digito=int(orden[0]),
        confianza=float(salida[orden[0]]),
        segundo=int(orden[1]),
        confianza_segundo=float(salida[orden[1]]),
        tiempo_s=float(tiempo),
    )


def rasgos(extractor, imagenes_28: np.ndarray) -> np.ndarray | None:
    """Vector de rasgos para un lote de imágenes 28x28 uint8 -> (n, 128)."""
    if extractor is None:
        return None
    lote = (imagenes_28.astype(np.float32) / 255.0)[..., np.newaxis]
    return np.asarray(extractor(lote, training=False))


def resumen_capas(modelo) -> list[dict]:
    """Tabla con las capas del modelo cargado (para la sección académica)."""
    filas = []
    for capa in modelo.layers:
        try:
            forma = str(tuple(capa.output.shape)).replace("None", "lote")
        except Exception:
            forma = "—"
        filas.append({
            "Capa": capa.name,
            "Tipo": capa.__class__.__name__,
            "Forma de salida": forma,
            "Parámetros": int(capa.count_params()),
        })
    return filas
