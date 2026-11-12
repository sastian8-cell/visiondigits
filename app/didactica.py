"""
didactica.py — Módulo 5 (parte pedagógica).

* Carga las muestras de referencia exportadas desde Colab (5 variantes por dígito).
  Si no existen, genera muestras tipográficas de respaldo para que la app siga funcionando.
* Compara el dibujo del niño con las muestras (similitud en el espacio de rasgos de la CNN).
* Genera mensajes de retroalimentación amigables según el modo de juego.
* Carga el archivo de métricas del entrenamiento.
"""
from __future__ import annotations

import json
import random
from dataclasses import dataclass

import numpy as np
import streamlit as st
from PIL import Image, ImageDraw, ImageFont

import config
import preprocesamiento as pp


# ---------------------------------------------------------------------------
# Muestras de referencia
# ---------------------------------------------------------------------------
def _muestras_tipograficas(digito: int, n: int) -> list[np.ndarray]:
    """Respaldo: dibuja el dígito con una fuente, con variaciones de grosor e inclinación."""
    try:
        fuente = ImageFont.truetype("DejaVuSans.ttf", 200)
    except OSError:
        try:
            fuente = ImageFont.load_default(size=200)
        except TypeError:
            fuente = ImageFont.load_default()
    variantes = [(0, 6), (-10, 10), (8, 4), (-4, 14), (12, 8)]
    muestras = []
    for angulo, grosor in variantes[:n]:
        img = Image.new("L", (280, 280), 255)
        ImageDraw.Draw(img).text((140, 140), str(digito), fill=0, font=fuente,
                                 anchor="mm", stroke_width=grosor, stroke_fill=0)
        img = img.rotate(angulo, fillcolor=255)
        res = pp.preprocesar(img)
        if res.imagen_28 is not None:
            muestras.append(res.imagen_28)
    return muestras


@st.cache_data(show_spinner=False)
def cargar_muestras(digito: int, n: int = config.MUESTRAS_POR_DIGITO) -> tuple[list[np.ndarray], bool]:
    """
    Devuelve (lista de imágenes 28x28 en formato MNIST, son_reales).
    son_reales = False cuando se usaron las muestras tipográficas de respaldo.
    """
    rutas = sorted(config.DIR_MUESTRAS.glob(f"digito_{digito}_*.png"))[:n]
    if rutas:
        return [np.asarray(Image.open(r).convert("L")) for r in rutas], True
    return _muestras_tipograficas(digito, n), False


def a_tinta(imagen_28: np.ndarray, tamano: int = 110,
            tinta=(49, 46, 129), papel=(255, 251, 235)) -> Image.Image:
    """
    Convierte una imagen MNIST (blanco sobre negro) a "tinta sobre papel" de colores,
    más natural para un niño, y la amplía con suavizado.
    """
    t = (imagen_28.astype(np.float32) / 255.0)[..., None]
    rgb = np.array(papel) * (1 - t) + np.array(tinta) * t
    img = Image.fromarray(rgb.astype(np.uint8))
    return img.resize((tamano, tamano), Image.Resampling.BICUBIC)


# ---------------------------------------------------------------------------
# Comparación con las muestras
# ---------------------------------------------------------------------------
def similitud_con_muestras(rasgos_dibujo: np.ndarray | None, rasgos_muestras: np.ndarray | None,
                           dibujo_28: np.ndarray, muestras_28: list[np.ndarray]) -> np.ndarray:
    """
    Similitud coseno (0-1) entre el dibujo y cada muestra.
    Se usa el espacio de rasgos de la CNN (lo que la red "piensa" de la forma);
    si no está disponible, se comparan directamente los píxeles.
    """
    if rasgos_dibujo is not None and rasgos_muestras is not None:
        a, b = rasgos_dibujo.reshape(1, -1), rasgos_muestras
    else:
        a = dibujo_28.reshape(1, -1).astype(np.float32)
        b = np.stack(muestras_28).reshape(len(muestras_28), -1).astype(np.float32)
    a = a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-8)
    b = b / (np.linalg.norm(b, axis=1, keepdims=True) + 1e-8)
    return np.clip(b @ a.T, 0, 1).ravel()


# ---------------------------------------------------------------------------
# Retroalimentación pedagógica
# ---------------------------------------------------------------------------
@dataclass
class Retroalimentacion:
    tipo: str          # "exito" | "bien" | "cerca" | "intentar" | "info"
    titulo: str
    mensaje: str
    estrellas: int     # 0-3
    celebrar: bool = False


_FELICITACIONES = ["¡Excelente trabajo!", "¡Eres un campeón de los números!", "¡Increíble!",
                   "¡Lo hiciste genial!", "¡Súper bien!"]
_ANIMOS = ["¡Tú puedes!", "¡Sigue practicando, vas muy bien!", "¡Inténtalo otra vez!",
           "¡Cada intento te hace mejor!"]


def retro_desafio(pred, objetivo: int) -> Retroalimentacion:
    """Mensaje para el Modo Desafío: compara lo reconocido con el número pedido."""
    nombre = config.NOMBRES_NUMEROS[objetivo]
    prob_objetivo = float(pred.probabilidades[objetivo])

    if pred.digito == objetivo and pred.confianza >= config.UMBRAL_CONFIANZA_ALTA:
        return Retroalimentacion("exito", f"{random.choice(_FELICITACIONES)} ⭐⭐⭐",
                                 f"Tu <b>{objetivo}</b> se parece muchísimo a los que aprendió la computadora "
                                 f"({prob_objetivo * 100:.0f} % de parecido).", 3, celebrar=True)
    if pred.digito == objetivo and pred.confianza >= config.UMBRAL_CONFIANZA_MEDIA:
        return Retroalimentacion("bien", "¡Muy bien! ⭐⭐",
                                 f"La computadora reconoció tu <b>{nombre}</b>. Para que quede perfecto, "
                                 f"mira las muestras y hazlo un poquito más clarito.", 2, celebrar=True)
    if pred.digito == objetivo:
        return Retroalimentacion("cerca", "¡Casi casi! ⭐",
                                 f"Sí parece un <b>{objetivo}</b>, pero la computadora duda un poco. "
                                 f"{config.PISTAS_TRAZO[objetivo]}", 1)
    if pred.segundo == objetivo:
        return Retroalimentacion("cerca", "¡Estás muy cerca! ⭐",
                                 f"La computadora vio un <b>{pred.digito}</b>, pero su segunda idea fue tu "
                                 f"<b>{objetivo}</b>. {config.PISTAS_TRAZO[objetivo]}", 1)
    return Retroalimentacion("intentar", f"Mmm… {random.choice(_ANIMOS)} 💪",
                             f"Tu dibujo se parece más a un <b>{pred.digito}</b> "
                             f"({config.NOMBRES_NUMEROS[pred.digito]}). Observa cómo se escribe el "
                             f"<b>{objetivo}</b> en las muestras. {config.PISTAS_TRAZO[objetivo]}", 0)


def retro_libre(pred, esperado: int | None = None) -> Retroalimentacion:
    """Mensaje para Práctica Libre: el niño dibuja cualquier número."""
    nombre = config.NOMBRES_NUMEROS[pred.digito]
    if esperado is not None:   # el niño (o la docente) indicó qué número iba a dibujar
        return retro_desafio(pred, esperado)
    if pred.confianza >= config.UMBRAL_CONFIANZA_ALTA:
        return Retroalimentacion("exito", f"¡Dibujaste un {pred.digito}! 🎉",
                                 f"Se ve clarito: es el número <b>{nombre}</b>. ¡Mira otras formas de "
                                 f"escribirlo aquí abajo!", 3, celebrar=True)
    if pred.confianza >= config.UMBRAL_CONFIANZA_MEDIA:
        return Retroalimentacion("bien", f"Creo que es un {pred.digito} 🤔",
                                 f"También podría ser un <b>{pred.segundo}</b>. "
                                 f"Compara tu dibujo con las muestras.", 2)
    return Retroalimentacion("cerca", "¡Qué difícil! 🧐",
                             f"Dudo entre <b>{pred.digito}</b> y <b>{pred.segundo}</b>. "
                             f"¿Lo intentamos un poco más grande y en el centro?", 1)


# ---------------------------------------------------------------------------
# Métricas del entrenamiento
# ---------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def cargar_metricas() -> dict | None:
    """Lee recursos/metricas.json generado en Colab (None si no existe)."""
    if not config.RUTA_METRICAS.exists():
        return None
    try:
        return json.loads(config.RUTA_METRICAS.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
