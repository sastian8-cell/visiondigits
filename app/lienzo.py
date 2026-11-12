"""
lienzo.py — Módulo 1 (entrada): adaptador del componente de dibujo.

El paquete `streamlit-drawable-canvas` cambió su API entre versiones
(0.9.x -> 0.13+: aparece `return_image_data` y desaparece `display_toolbar`).
Este adaptador detecta los parámetros disponibles para que la app funcione con ambas.
"""
from __future__ import annotations

import inspect

import numpy as np
import streamlit as st

import config

try:
    from streamlit_drawable_canvas import st_canvas
    DISPONIBLE, ERROR_IMPORTACION = True, ""
except Exception as e:  # paquete no instalado o incompatible con la versión de Streamlit
    st_canvas, DISPONIBLE, ERROR_IMPORTACION = None, False, str(e)


def dibujar(clave: str, grosor: int, color: str = config.COLOR_TRAZO) -> np.ndarray | None:
    """Muestra el lienzo y devuelve la imagen RGBA dibujada (o None si aún no hay datos)."""
    if not DISPONIBLE:
        st.error("No se pudo cargar el lienzo de dibujo. Instala `streamlit-drawable-canvas` "
                 f"(ver requirements.txt). Detalle: {ERROR_IMPORTACION}")
        return None

    parametros = dict(
        fill_color="rgba(0, 0, 0, 0)",
        stroke_width=grosor,
        stroke_color=color,
        background_color=config.COLOR_FONDO_LIENZO,
        height=config.TAMANO_LIENZO,
        width=config.TAMANO_LIENZO,
        drawing_mode="freedraw",
        update_streamlit=True,
        key=clave,
    )
    aceptados = inspect.signature(st_canvas).parameters
    if "return_image_data" in aceptados:
        parametros["return_image_data"] = True
    if "display_toolbar" in aceptados:
        parametros["display_toolbar"] = False

    resultado = st_canvas(**parametros)
    try:
        return resultado.image_data
    except Exception:
        return None
