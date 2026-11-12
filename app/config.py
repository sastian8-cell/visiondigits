"""
config.py — Parámetros globales de VisionDigits.

Centraliza rutas, tamaños y umbrales para que el resto de módulos
no tenga "números mágicos" repartidos por el código.
"""
from pathlib import Path

# --- Rutas ------------------------------------------------------------------
DIR_BASE = Path(__file__).resolve().parent
DIR_MODELOS = DIR_BASE / "modelos"
RUTA_MODELO_KERAS = DIR_MODELOS / "modelo_cnn.keras"   # exportado desde Colab
RUTA_MODELO_H5 = DIR_MODELOS / "modelo_cnn.h5"         # respaldo HDF5

DIR_RECURSOS = DIR_BASE / "recursos"
RUTA_METRICAS = DIR_RECURSOS / "metricas.json"
RUTA_MATRIZ_PNG = DIR_RECURSOS / "matriz_confusion.png"
DIR_MUESTRAS = DIR_RECURSOS / "muestras"
DIR_FIGURAS = DIR_RECURSOS / "figuras"

# --- Lienzo -----------------------------------------------------------------
TAMANO_LIENZO = 300          # píxeles (cuadrado)
GROSOR_TRAZO_DEFECTO = 22
COLOR_TRAZO = "#1E1B4B"      # tinta azul muy oscuro (contrasta bien al pasar a gris)
COLOR_FONDO_LIENZO = "#FFFFFF"

# --- Formato MNIST ----------------------------------------------------------
TAMANO_MNIST = 28            # imagen final 28x28
TAMANO_CAJA_DIGITO = 20      # MNIST encaja el dígito en una caja de 20x20 y lo centra en 28x28
UMBRAL_TINTA = 40            # nivel (0-255) a partir del cual un píxel se considera trazo

# --- Retroalimentación pedagógica ------------------------------------------
UMBRAL_CONFIANZA_ALTA = 0.80
UMBRAL_CONFIANZA_MEDIA = 0.50
MUESTRAS_POR_DIGITO = 5

NOMBRES_NUMEROS = ["cero", "uno", "dos", "tres", "cuatro",
                   "cinco", "seis", "siete", "ocho", "nueve"]

# Pista de trazo para cada número (lenguaje sencillo para niños de inicial y 1.er grado)
PISTAS_TRAZO = {
    0: "Empieza arriba y da una vuelta completa, como un huevito 🥚.",
    1: "Una rayita bajando derechito de arriba hacia abajo ⬇️.",
    2: "Haz una curva como un cuello de patito 🦆 y termina con una raya acostada.",
    3: "Dos pancitas mirando a la derecha, una arriba y otra abajo.",
    4: "Baja, cruza hacia la derecha y luego una raya larga hacia abajo.",
    5: "Rayita arriba, baja un poco y haz una pancita grande abajo.",
    6: "Baja con una curva como un tobogán 🛝 y cierra una bolita abajo.",
    7: "Una raya acostada arriba y luego baja inclinado, como una resbaladera.",
    8: "Dibuja una 'S' y regresa para cerrarla: ¡dos bolitas, una sobre otra! ⛄",
    9: "Haz una bolita arriba y luego una rayita bajando por la derecha 🎈.",
}

# --- Paleta de la interfaz --------------------------------------------------
COLOR_PRINCIPAL = "#6366F1"    # índigo: predicción principal
COLOR_SEGUNDA = "#F59E0B"      # ámbar: segunda opción
COLOR_RESTO = "#CBD5E1"        # gris: resto de categorías
COLOR_EXITO = "#10B981"
COLOR_ERROR = "#EF4444"

PENDIENTE, CORRECTO, INCORRECTO = "⏳ Pendiente", "✅ Correcto", "❌ Incorrecto"
