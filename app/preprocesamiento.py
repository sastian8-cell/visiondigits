"""
preprocesamiento.py — Módulo 2: Preprocesamiento avanzado.

Convierte un dibujo del lienzo (o una imagen subida) al MISMO formato con el que
se entrenó la CNN (MNIST):

    1. Escala de grises
    2. Inversión de colores (MNIST = trazo blanco sobre fondo negro)
    3. Limpieza de ruido y estiramiento de contraste
    4. Recorte de bordes (caja delimitadora del trazo)
    5. Redimensionado manteniendo proporción a una caja de 20x20
    6. Centrado en un lienzo de 28x28 por centro de masa
    7. Normalización a [0, 1] y tensor (1, 28, 28, 1)

Cada paso intermedio se guarda para mostrarlo en la interfaz (valor didáctico).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from PIL import Image

import config


@dataclass
class ResultadoPreprocesamiento:
    """Salida del preprocesamiento."""
    tensor: np.ndarray | None                 # (1, 28, 28, 1) float32 listo para el modelo
    imagen_28: np.ndarray | None              # (28, 28) uint8, formato MNIST
    pasos: dict[str, np.ndarray] = field(default_factory=dict)  # imágenes intermedias
    vacio: bool = False                       # True si no hay trazo


# ---------------------------------------------------------------------------
# Pasos individuales
# ---------------------------------------------------------------------------
def a_escala_grises(imagen: np.ndarray | Image.Image) -> np.ndarray:
    """
    Convierte cualquier imagen (RGBA, RGB, L) a escala de grises uint8.
    Si tiene canal alfa, se compone sobre fondo blanco: el lienzo devuelve
    los trazos con transparencia y el fondo podría venir transparente.
    """
    if isinstance(imagen, Image.Image):
        imagen = np.asarray(imagen.convert("RGBA") if imagen.mode in ("P", "LA") else imagen)
    arr = np.asarray(imagen).astype(np.float32)

    if arr.ndim == 2:                                   # ya es gris
        return arr.clip(0, 255).astype(np.uint8)
    if arr.shape[-1] == 4:                              # RGBA -> RGB sobre blanco
        alfa = arr[..., 3:4] / 255.0
        arr = arr[..., :3] * alfa + 255.0 * (1.0 - alfa)
    rgb = arr[..., :3]
    gris = 0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2]   # luminancia ITU-R 601
    return gris.clip(0, 255).astype(np.uint8)


def invertir_colores(gris: np.ndarray) -> np.ndarray:
    """
    Deja el trazo claro sobre fondo oscuro (formato MNIST).
    Se decide mirando el BORDE de la imagen: si el fondo es claro (papel, lienzo blanco)
    se invierte; si ya es oscuro (pizarra, imagen tipo MNIST) se deja igual.
    """
    borde = np.concatenate([gris[0, :], gris[-1, :], gris[:, 0], gris[:, -1]])
    return (255 - gris) if borde.mean() > 127 else gris.copy()


def limpiar_ruido(invertida: np.ndarray, umbral: int = config.UMBRAL_TINTA) -> np.ndarray:
    """Elimina píxeles débiles (sombras de foto, antialiasing) y estira el contraste a 0-255."""
    limpia = invertida.astype(np.float32)
    limpia[limpia < umbral] = 0
    maximo = limpia.max()
    if maximo > 0:
        limpia = limpia * (255.0 / maximo)
    return limpia.astype(np.uint8)


def recortar_bordes(imagen: np.ndarray, umbral: int = config.UMBRAL_TINTA) -> np.ndarray | None:
    """Recorta a la caja delimitadora del trazo. Devuelve None si no hay trazo."""
    filas = np.where((imagen > umbral).any(axis=1))[0]
    columnas = np.where((imagen > umbral).any(axis=0))[0]
    if len(filas) == 0 or len(columnas) == 0:
        return None
    return imagen[filas[0]:filas[-1] + 1, columnas[0]:columnas[-1] + 1]


def redimensionar_y_centrar(recorte: np.ndarray) -> np.ndarray:
    """
    Escala el trazo para que su lado mayor mida 20 px (manteniendo proporción),
    lo pega en el centro de un lienzo negro de 28x28 y lo desplaza para que su
    centro de masa quede en el centro, exactamente como se construyó MNIST.
    """
    alto, ancho = recorte.shape
    escala = config.TAMANO_CAJA_DIGITO / max(alto, ancho)
    nuevo_alto = max(1, round(alto * escala))
    nuevo_ancho = max(1, round(ancho * escala))
    reducida = np.asarray(
        Image.fromarray(recorte).resize((nuevo_ancho, nuevo_alto), Image.Resampling.LANCZOS)
    )

    lienzo = np.zeros((config.TAMANO_MNIST, config.TAMANO_MNIST), dtype=np.uint8)
    y0 = (config.TAMANO_MNIST - nuevo_alto) // 2
    x0 = (config.TAMANO_MNIST - nuevo_ancho) // 2
    lienzo[y0:y0 + nuevo_alto, x0:x0 + nuevo_ancho] = reducida

    # Centrado por centro de masa (ponderado por la intensidad del trazo)
    total = lienzo.sum()
    if total > 0:
        ys, xs = np.indices(lienzo.shape)
        cy, cx = (ys * lienzo).sum() / total, (xs * lienzo).sum() / total
        centro = config.TAMANO_MNIST / 2 - 0.5
        dy = int(np.clip(round(centro - cy), -4, 4))
        dx = int(np.clip(round(centro - cx), -4, 4))
        lienzo = np.roll(lienzo, shift=(dy, dx), axis=(0, 1))
    return lienzo


def normalizar(imagen_28: np.ndarray) -> np.ndarray:
    """[0, 255] -> [0, 1] y forma de tensor (1, 28, 28, 1)."""
    return (imagen_28.astype(np.float32) / 255.0).reshape(1, config.TAMANO_MNIST, config.TAMANO_MNIST, 1)


# ---------------------------------------------------------------------------
# Pipeline completo
# ---------------------------------------------------------------------------
def preprocesar(imagen: np.ndarray | Image.Image) -> ResultadoPreprocesamiento:
    """Ejecuta todo el pipeline y guarda los pasos intermedios."""
    pasos: dict[str, np.ndarray] = {}

    gris = a_escala_grises(imagen)
    pasos["1. Escala de grises"] = gris

    invertida = invertir_colores(gris)
    pasos["2. Colores invertidos"] = invertida

    limpia = limpiar_ruido(invertida)
    recorte = recortar_bordes(limpia)
    if recorte is None or recorte.size < 4:
        return ResultadoPreprocesamiento(tensor=None, imagen_28=None, pasos=pasos, vacio=True)
    pasos["3. Recorte de bordes"] = recorte

    imagen_28 = redimensionar_y_centrar(recorte)
    pasos["4. Centrado 28×28 (MNIST)"] = imagen_28

    return ResultadoPreprocesamiento(tensor=normalizar(imagen_28), imagen_28=imagen_28, pasos=pasos)


def ampliar_para_mostrar(imagen: np.ndarray, tamano: int = 140) -> Image.Image:
    """Amplía una imagen pequeña sin suavizar (se ven los píxeles, útil para explicar 28×28)."""
    return Image.fromarray(imagen).resize((tamano, tamano), Image.Resampling.NEAREST)
