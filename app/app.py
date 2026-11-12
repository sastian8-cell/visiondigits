"""
VisionDigits — Entregable 2: Aplicación web en Streamlit.

Sistema inteligente de reconocimiento de números escritos a mano mediante Machine Learning,
con una interfaz didáctica para niños de inicial y primer grado.

Secciones:
    A. Reconocimiento y Modo Didáctico Infantil (lienzo, modos, muestras, retroalimentación)
    B. Probabilidades de clasificación (gráfica de barras interactiva)
    C. Información académica del modelo (teoría, MNIST, CNN vs densa, métricas, matriz de confusión)
    D. Historial de pruebas de la sesión (registro, confirmación docente y estadísticas)

Ejecutar:  streamlit run app/app.py
"""
from __future__ import annotations

import random

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from PIL import Image

st.set_page_config(page_title="VisionDigits", page_icon="✏️", layout="wide",
                   initial_sidebar_state="expanded")

import config                       # noqa: E402  (después de set_page_config)
import didactica as dd              # noqa: E402
import estilos                      # noqa: E402
import historial as hs              # noqa: E402
import lienzo                       # noqa: E402
import modelo as md                 # noqa: E402
import preprocesamiento as pp       # noqa: E402

MODO_LIBRE = "🎨 Práctica Libre"
MODO_DESAFIO = "🏆 Modo Desafío para Niños"
FUENTE_LIENZO = "🖍️ Dibujar en el lienzo"
FUENTE_ARCHIVO = "📁 Subir una imagen"

# =============================================================================
# Estado de la sesión
# =============================================================================
st.markdown(estilos.CSS, unsafe_allow_html=True)
hs.iniciar()
ss = st.session_state
ss.setdefault("version_lienzo", 0)       # cambiarla "reinicia" el lienzo (botón Borrar)
ss.setdefault("ultimo", None)            # último resultado mostrado
ss.setdefault("objetivo", random.randint(0, 9))
ss.setdefault("estrellas_total", 0)

modelo, error_modelo = md.cargar_modelo()
metricas = dd.cargar_metricas()


# =============================================================================
# Callbacks
# =============================================================================
def borrar_lienzo() -> None:
    ss.version_lienzo += 1
    ss.ultimo = None


def nuevo_objetivo() -> None:
    """Elige un número sorpresa distinto al actual y limpia el lienzo."""
    ss.objetivo = random.choice([d for d in range(10) if d != ss.objetivo])
    borrar_lienzo()


# =============================================================================
# Barra lateral
# =============================================================================
with st.sidebar:
    st.markdown("## ⚙️ Configuración")
    modo = st.radio("Modo de uso", [MODO_LIBRE, MODO_DESAFIO], key="modo", on_change=borrar_lienzo,
                    help="Práctica Libre: dibuja cualquier número. Desafío: la app te pide un número.")
    participante = st.text_input("👧 Nombre o estilo de escritura", key="participante",
                                 placeholder="Ej.: Ana · trazo grueso · con dedo",
                                 help="Se guarda en el historial para comparar estilos en vivo.")
    grosor = st.slider("✏️ Grosor del lápiz", 10, 40, config.GROSOR_TRAZO_DEFECTO)

    st.divider()
    st.markdown("### ⭐ Mis estrellas")
    zona_estrellas = st.empty()   # se rellena al final, cuando ya se sumaron las del último intento

    st.divider()
    if modelo is not None:
        st.success("🧠 Modelo CNN cargado")
        if metricas:
            cnn = metricas["modelos"]["cnn"]
            st.caption(f"Accuracy en prueba: **{cnn['accuracy'] * 100:.2f} %** · "
                       f"entrenado el {metricas.get('fecha_entrenamiento', '—')}")
    else:
        st.error("🧠 Modelo no disponible")

# =============================================================================
# Encabezado
# =============================================================================
st.markdown("<div class='vd-titulo'>✏️ VisionDigits</div>"
            "<div class='vd-subtitulo'>Aprende a escribir los números y descubre cómo una "
            "inteligencia artificial los reconoce</div>", unsafe_allow_html=True)

if modelo is None:
    st.error("No se encontró el modelo entrenado.")
    st.markdown(f"""
1. Ejecuta el cuaderno **`colab/visiondigits_entrenamiento.ipynb`** en Google Colab.
2. Descarga `visiondigits_artefactos.zip` y descomprímelo **dentro de la carpeta `app/`**.
3. Deben existir `app/modelos/modelo_cnn.keras` y `app/recursos/metricas.json`.
4. Vuelve a ejecutar `streamlit run app/app.py`.

<span class='vd-nota'>Detalle técnico: {error_modelo}</span>
""", unsafe_allow_html=True)
    st.stop()

extractor = md.crear_extractor(modelo)
es_desafio = modo == MODO_DESAFIO


# =============================================================================
# Lógica de reconocimiento
# =============================================================================
def reconocer(imagen, fuente: str, esperado: int | None) -> None:
    """Preprocesa, predice, genera la retroalimentación y registra el intento."""
    if imagen is None:
        st.warning("Primero dibuja o sube una imagen 🖍️")
        return
    res = pp.preprocesar(imagen)
    if res.vacio:
        st.warning("🖍️ ¡El lienzo está vacío! Dibuja un número grande en el centro.")
        return

    pred = md.predecir(modelo, res.tensor)

    # Muestras con las que se compara: el número pedido (o declarado) o, si no hay, el reconocido
    digito_ref = esperado if esperado is not None else pred.digito
    muestras, reales = dd.cargar_muestras(digito_ref)
    sims = dd.similitud_con_muestras(
        md.rasgos(extractor, res.imagen_28[np.newaxis]),
        md.rasgos(extractor, np.stack(muestras)) if muestras else None,
        res.imagen_28, muestras,
    ) if muestras else np.array([])

    retro = dd.retro_desafio(pred, esperado) if es_desafio else dd.retro_libre(pred, esperado)
    hs.registrar(modo="Desafío" if es_desafio else "Libre", fuente="Lienzo" if fuente == FUENTE_LIENZO else "Imagen",
                 participante=participante, esperado=esperado, pred=pred)
    ss.estrellas_total += retro.estrellas
    ss.ultimo = {"pred": pred, "pasos": res.pasos, "imagen_28": res.imagen_28, "retro": retro,
                 "digito_ref": digito_ref, "muestras": muestras, "muestras_reales": reales,
                 "sims": sims, "esperado": esperado}
    if retro.celebrar:
        st.balloons()


def fila_muestras(muestras: list[np.ndarray], sims: np.ndarray | None = None, tamano: int = 110) -> None:
    """Muestra las variantes de referencia en fila; si hay similitudes, resalta la más parecida."""
    if not muestras:
        return
    columnas = st.columns(len(muestras))
    mejor = int(np.argmax(sims)) if sims is not None and len(sims) else -1
    for i, (col, img) in enumerate(zip(columnas, muestras)):
        with col:
            tinta = (5, 150, 105) if i == mejor else (49, 46, 129)
            st.image(dd.a_tinta(img, tamano, tinta=tinta), width=tamano)
            if sims is not None and len(sims):
                clase = "vd-mejor" if i == mejor else "vd-muestra-pie"
                extra = " 🏅" if i == mejor else ""
                st.markdown(f"<div class='{clase}'>{sims[i] * 100:.0f} %{extra}</div>",
                            unsafe_allow_html=True)
            else:
                st.markdown(f"<div class='vd-muestra-pie'>Forma {i + 1}</div>", unsafe_allow_html=True)


def panel_resultado(u: dict | None) -> None:
    """Indicadores de predicción + retroalimentación pedagógica + comparación con muestras."""
    if u is None:
        st.markdown("<div class='vd-tarjeta' style='text-align:center;padding:2.5rem 1rem;'>"
                    "<div style='font-size:4rem'>🤖</div>"
                    "<div style='font-size:1.2rem;color:#64748B'>Dibuja un número y pulsa "
                    "<b>Reconocer</b>.<br>¡Aquí te diré qué número veo!</div></div>",
                    unsafe_allow_html=True)
        return

    pred, retro = u["pred"], u["retro"]
    c_num, c_ind = st.columns([1, 1.35])
    with c_num:
        st.markdown(f"<div class='vd-tarjeta'><div class='vd-numero'>{pred.digito}</div>"
                    f"<div class='vd-nombre'>{config.NOMBRES_NUMEROS[pred.digito]}</div>"
                    f"<div class='vd-estrellas'>{'⭐' * retro.estrellas or '💪'}</div></div>",
                    unsafe_allow_html=True)
    with c_ind:
        m1, m2 = st.columns(2)
        m1.metric("Confianza", f"{pred.confianza * 100:.1f} %")
        m2.metric("2.ª opción", f"{pred.segundo} · {pred.confianza_segundo * 100:.1f} %")
        m3, m4 = st.columns([1, 1])
        m3.metric("⏱️ Tiempo", f"{pred.tiempo_s:.4f} s", help="Tiempo de inferencia de la red, en segundos")
        with m4:
            st.image(pp.ampliar_para_mostrar(u["imagen_28"], 96), width=96,
                     caption="Así lo ve la IA (28×28)")

    st.markdown(f"<div class='vd-retro vd-{retro.tipo}'><h3>{retro.titulo}</h3>{retro.mensaje}</div>",
                unsafe_allow_html=True)

    titulo = (f"🔎 Tu dibujo comparado con las muestras del <b>{u['digito_ref']}</b>")
    st.markdown(f"<h4 style='margin-bottom:.3rem'>{titulo}</h4>", unsafe_allow_html=True)
    st.caption("El porcentaje indica cuánto se parece tu trazo a cada muestra según lo que aprendió la red.")
    fila_muestras(u["muestras"], u["sims"], tamano=96)
    if not u["muestras_reales"]:
        st.caption("ℹ️ Muestras tipográficas de respaldo (no se encontraron las exportadas desde Colab).")

    with st.expander("🔬 Ver los pasos del preprocesamiento"):
        columnas = st.columns(len(u["pasos"]))
        for col, (nombre, img) in zip(columnas, u["pasos"].items()):
            with col:
                vista = pp.ampliar_para_mostrar(img, 120) if img.shape == (28, 28) else img
                st.image(vista, caption=nombre, width=120)
        st.caption("Gris → invertir (trazo blanco sobre negro) → recortar bordes → escalar a 20×20 "
                   "y centrar por centro de masa en 28×28 → normalizar a [0, 1].")


# =============================================================================
# Pestañas
# =============================================================================
tab_a, tab_b, tab_c, tab_d = st.tabs([
    "🖍️ A · Reconocer y practicar",
    "📊 B · Probabilidades",
    "🎓 C · Información del modelo",
    "📋 D · Historial de la sesión",
])

# -----------------------------------------------------------------------------
# SECCIÓN A — Reconocimiento y Modo Didáctico Infantil
# -----------------------------------------------------------------------------
with tab_a:
    if es_desafio:
        c_banner, c_ctrl = st.columns([3, 1], vertical_alignment="center")
        with c_ctrl:
            st.selectbox("Elige el número", list(range(10)), key="objetivo", on_change=borrar_lienzo)
            st.button("🎲 Número sorpresa", on_click=nuevo_objetivo, width="stretch")
        objetivo = ss.objetivo
        with c_banner:
            st.markdown(f"<div class='vd-banner'><h2>🏆 Desafío: escribe el número "
                        f"<span class='vd-objetivo'>{objetivo}</span> ({config.NOMBRES_NUMEROS[objetivo]})</h2>"
                        f"<p>💡 {config.PISTAS_TRAZO[objetivo]}</p></div>", unsafe_allow_html=True)
        st.markdown("**👀 Observa cómo se escribe:**")
        fila_muestras(dd.cargar_muestras(objetivo)[0], tamano=100)
        st.write("")
    else:
        objetivo = None

    col_entrada, col_resultado = st.columns([1, 1.25], gap="large")

    with col_entrada:
        fuente = st.radio("¿Cómo quieres ingresar el número?", [FUENTE_LIENZO, FUENTE_ARCHIVO],
                          horizontal=True, label_visibility="collapsed")
        imagen = None
        if fuente == FUENTE_LIENZO:
            st.caption("Dibuja con el mouse, el dedo o un lápiz táctil 👆")
            imagen = lienzo.dibujar(f"lienzo_{ss.version_lienzo}", grosor)
        else:
            archivo = st.file_uploader("Sube una foto o imagen de un número",
                                       type=["png", "jpg", "jpeg", "bmp", "webp"],
                                       key=f"archivo_{ss.version_lienzo}")
            if archivo is not None:
                imagen = Image.open(archivo)
                st.image(imagen, width=config.TAMANO_LIENZO, caption="Imagen subida")

        esperado = objetivo
        if not es_desafio:
            opcion = st.selectbox("¿Qué número vas a dibujar? (opcional)", ["No lo diré 🤫"] + list(range(10)),
                                  help="Si lo indicas, la app te dirá si se parece y el historial "
                                       "podrá contar los aciertos automáticamente.")
            esperado = None if isinstance(opcion, str) else int(opcion)

        b1, b2 = st.columns(2)
        pulsado = b1.button("🔍 Reconocer número", type="primary", width="stretch")
        b2.button("🧽 Borrar", on_click=borrar_lienzo, width="stretch")
        if pulsado:
            reconocer(imagen, fuente, esperado)

    with col_resultado:
        panel_resultado(ss.ultimo)
        if es_desafio and ss.ultimo and ss.ultimo["retro"].celebrar:
            st.button("➡️ ¡Siguiente desafío!", on_click=nuevo_objetivo, type="primary", width="stretch")

# -----------------------------------------------------------------------------
# SECCIÓN B — Probabilidades de clasificación
# -----------------------------------------------------------------------------
with tab_b:
    st.subheader("📊 Puntuaciones de las 10 categorías")
    if ss.ultimo is None:
        st.info("Reconoce un número en la pestaña **A** para ver aquí sus probabilidades.")
    else:
        pred = ss.ultimo["pred"]
        probs = pred.probabilidades * 100
        colores = [config.COLOR_PRINCIPAL if d == pred.digito else
                   config.COLOR_SEGUNDA if d == pred.segundo else config.COLOR_RESTO for d in range(10)]
        etiquetas = [("🥇 " if d == pred.digito else "🥈 " if d == pred.segundo else "")
                     + (f"{p:.1f} %" if p >= 1 or d in (pred.digito, pred.segundo) else "") for d, p in enumerate(probs)]
        escala_log = st.toggle("Escala logarítmica (para ver las probabilidades muy pequeñas)", value=False)

        fig = go.Figure(go.Bar(
            x=[str(d) for d in range(10)], y=probs, marker_color=colores,
            text=etiquetas, textposition="outside", textfont=dict(size=15),
            hovertemplate="Dígito %{x}<br>Probabilidad: %{y:.4f} %<extra></extra>",
        ))
        fig.update_layout(
            height=430, margin=dict(t=40, b=40, l=40, r=20), plot_bgcolor="white",
            xaxis=dict(title="Dígito (categoría)", type="category"),
            yaxis=dict(title="Probabilidad (%)", type="log" if escala_log else "linear",
                       range=None if escala_log else [0, 115], gridcolor="#F1F5F9"),
            font=dict(family="Fredoka, sans-serif", size=14),
        )
        st.plotly_chart(fig, width="stretch")

        c1, c2, c3 = st.columns(3)
        c1.metric("🥇 Predicción principal", f"{pred.digito}  ·  {pred.confianza * 100:.2f} %")
        c2.metric("🥈 Segunda opción", f"{pred.segundo}  ·  {pred.confianza_segundo * 100:.2f} %")
        c3.metric("Margen de decisión", f"{pred.margen * 100:.2f} pp",
                  help="Diferencia entre la 1.ª y la 2.ª opción (puntos porcentuales).")
        if pred.margen > 0.8:
            st.success("El modelo está **muy seguro**: casi toda la probabilidad está en una sola clase.")
        elif pred.margen > 0.3:
            st.info(f"Decisión **clara**, aunque el modelo ve algo de parecido con el {pred.segundo}.")
        else:
            st.warning(f"Decisión **ajustada**: el trazo es ambiguo entre {pred.digito} y {pred.segundo}.")

        with st.expander("📋 Tabla de puntuaciones y explicación"):
            tabla = pd.DataFrame({"Dígito": range(10), "Probabilidad": pred.probabilidades}) \
                .sort_values("Probabilidad", ascending=False)
            st.dataframe(tabla, hide_index=True, width="stretch", column_config={
                "Probabilidad": st.column_config.ProgressColumn(format="percent", min_value=0, max_value=1)})
            st.markdown("La última capa de la CNN tiene **10 neuronas** (una por dígito) con activación "
                        "**softmax**, que convierte sus puntuaciones en probabilidades que suman 100 %: "
                        r"$P(k) = \dfrac{e^{z_k}}{\sum_{j=0}^{9} e^{z_j}}$. "
                        "La predicción es la clase con mayor probabilidad (*argmax*).")

# -----------------------------------------------------------------------------
# SECCIÓN C — Información académica del modelo
# -----------------------------------------------------------------------------
with tab_c:
    st.subheader("🎓 ¿Cómo aprende la computadora a reconocer números?")
    t1, t2 = st.columns(2)
    with t1:
        st.markdown("""
**Aprendizaje supervisado.** Le mostramos a la red miles de imágenes **junto con la respuesta correcta**
(la etiqueta). En cada ronda (*época*) la red compara su predicción con la etiqueta, calcula el error
(*función de pérdida*) y ajusta sus pesos con el optimizador **Adam** para equivocarse menos.
Después la evaluamos con imágenes que **nunca vio** para comprobar que generaliza.
""")
    with t2:
        st.markdown("""
**Clasificación multiclase.** Cada imagen pertenece a **una sola de 10 clases** (0–9).
La red produce 10 probabilidades con *softmax* y se entrena con
**sparse categorical crossentropy**, que penaliza dar poca probabilidad a la clase correcta.
La métrica principal es la **accuracy** (porcentaje de aciertos).
""")

    # --- Dataset MNIST ---
    st.markdown("### 🗂️ Dataset MNIST")
    ds = (metricas or {}).get("dataset", {})
    d1, d2, d3, d4 = st.columns(4)
    d1.metric("Imágenes totales", f"{ds.get('total', 70000):,}".replace(",", " "))
    d2.metric("Entrenamiento", f"{ds.get('entrenamiento', 54000):,}".replace(",", " "))
    d3.metric("Validación", f"{ds.get('validacion', 6000):,}".replace(",", " "))
    d4.metric("Prueba final", f"{ds.get('prueba', 10000):,}".replace(",", " "))
    st.caption("Imágenes en escala de grises de 28 × 28 píxeles, dígito blanco sobre fondo negro, "
               "centrado por centro de masa. Las imágenes de prueba solo se usan en la evaluación final.")
    if ds.get("distribucion_entrenamiento"):
        fig_ds = go.Figure()
        for clave, nombre, color in [("distribucion_entrenamiento", "Entrenamiento", "#6366F1"),
                                     ("distribucion_validacion", "Validación", "#F59E0B"),
                                     ("distribucion_prueba", "Prueba", "#10B981")]:
            fig_ds.add_bar(x=[str(d) for d in range(10)], y=ds[clave], name=nombre, marker_color=color)
        fig_ds.update_layout(barmode="group", height=300, margin=dict(t=30, b=30), plot_bgcolor="white",
                             title="Distribución de categorías (dataset balanceado)",
                             xaxis_title="Dígito", yaxis_title="Imágenes")
        st.plotly_chart(fig_ds, width="stretch")

    # --- Arquitecturas ---
    st.markdown("### 🧠 Arquitectura: CNN vs red densa")
    a1, a2 = st.columns([1.1, 1])
    with a1:
        st.markdown("""
**CNN principal (modelo de la app)**

`Entrada 28×28×1` → `Conv2D 32 (3×3) + MaxPooling` *(trazos básicos: bordes, líneas, curvas)* →
`Conv2D 64 (3×3) + MaxPooling` *(patrones complejos: lazos, cruces, esquinas)* →
`Flatten` → `Dense 128 + Dropout` → `Dense 10 softmax`

**Red densa comparativa**

`Entrada 28×28` → `Flatten (784)` → `Dense 128` → `Dense 64` → `Dense 10 softmax`

**¿Por qué la CNN?** Sus filtros miran **vecindarios de píxeles** y se **comparten** en toda la imagen,
así reconocen un trazo sin importar dónde esté. El *MaxPooling* tolera pequeños desplazamientos e
inclinaciones, algo muy común cuando un niño dibuja. La red densa aplana la imagen y **pierde la
información espacial**, por eso comete más errores.
""")
    with a2:
        st.markdown("**Capas del modelo cargado**")
        st.dataframe(pd.DataFrame(md.resumen_capas(modelo)), hide_index=True, width="stretch", height=380)
        st.caption(f"Parámetros totales: **{modelo.count_params():,}**. Las capas *Random…* (aumento de datos) "
                   "solo actúan durante el entrenamiento.")

    if metricas:
        cnn, densa = metricas["modelos"]["cnn"], metricas["modelos"]["densa"]
        comparativa = pd.DataFrame({
            "Métrica": ["Accuracy", "Precision (macro)", "Recall (macro)", "F1-score (macro)",
                        "Errores en 10 000 imágenes", "Parámetros", "Inferencia por imagen (ms)",
                        "Tiempo de entrenamiento (s)"],
            "CNN": [f"{cnn['accuracy'] * 100:.2f} %", f"{cnn['precision_macro'] * 100:.2f} %",
                    f"{cnn['recall_macro'] * 100:.2f} %", f"{cnn['f1_macro'] * 100:.2f} %",
                    f"{cnn['errores']}", f"{cnn['parametros']:,}",
                    f"{cnn['tiempo_inferencia_individual_s'] * 1000:.2f}", f"{cnn['tiempo_entrenamiento_s']:.1f}"],
            "Red densa": [f"{densa['accuracy'] * 100:.2f} %", f"{densa['precision_macro'] * 100:.2f} %",
                          f"{densa['recall_macro'] * 100:.2f} %", f"{densa['f1_macro'] * 100:.2f} %",
                          f"{densa['errores']}", f"{densa['parametros']:,}",
                          f"{densa['tiempo_inferencia_individual_s'] * 1000:.2f}",
                          f"{densa['tiempo_entrenamiento_s']:.1f}"],
        })
        st.dataframe(comparativa, hide_index=True, width="stretch")
        reduccion = (1 - cnn["errores"] / max(densa["errores"], 1)) * 100
        st.success(f"La CNN comete **{reduccion:.0f} % menos errores** que la red densa "
                   f"({cnn['errores']} vs {densa['errores']}).")

        # --- Métricas finales ---
        st.markdown("### 🎯 Métricas finales de la CNN (conjunto de prueba)")
        k1, k2, k3, k4, k5 = st.columns(5)
        k1.metric("Accuracy", f"{cnn['accuracy'] * 100:.2f} %",
                  "Meta > 98 % ✅" if cnn["accuracy"] > 0.98 else "Meta > 98 % ❌",
                  delta_color="normal" if cnn["accuracy"] > 0.98 else "inverse")
        k2.metric("Precision", f"{cnn['precision_macro'] * 100:.2f} %")
        k3.metric("Recall", f"{cnn['recall_macro'] * 100:.2f} %")
        k4.metric("F1-score", f"{cnn['f1_macro'] * 100:.2f} %")
        k5.metric("Inferencia", f"{cnn['tiempo_inferencia_individual_s']:.4f} s")

        hist_c, hist_d = cnn.get("historial", {}), densa.get("historial", {})
        if hist_c.get("val_accuracy"):
            fig_h = go.Figure()
            for h, nombre, color in [(hist_c, "CNN", "#6366F1"), (hist_d, "Densa", "#F59E0B")]:
                if not h.get("accuracy"):
                    continue
                ep = list(range(1, len(h["accuracy"]) + 1))
                fig_h.add_scatter(x=ep, y=[v * 100 for v in h["accuracy"]], name=f"{nombre} · entrenamiento",
                                  line=dict(color=color))
                fig_h.add_scatter(x=ep, y=[v * 100 for v in h["val_accuracy"]], name=f"{nombre} · validación",
                                  line=dict(color=color, dash="dash"))
            fig_h.update_layout(height=320, title="Curvas de aprendizaje (accuracy por época)",
                                xaxis_title="Época", yaxis_title="Accuracy (%)", plot_bgcolor="white",
                                margin=dict(t=40, b=30))
            st.plotly_chart(fig_h, width="stretch")

        with st.expander("📑 Métricas por clase (CNN)"):
            por_clase = pd.DataFrame(cnn["reporte_por_clase"]).T.reset_index() \
                .rename(columns={"index": "Dígito", "precision": "Precision", "recall": "Recall",
                                 "f1": "F1-score", "soporte": "Imágenes"})
            st.dataframe(por_clase, hide_index=True, width="stretch", column_config={
                c: st.column_config.NumberColumn(format="%.4f") for c in ("Precision", "Recall", "F1-score")})

        # --- Matriz de confusión ---
        st.markdown("### 🧩 Matriz de Confusión")
        matriz = np.array(metricas["matriz_confusion"])
        normalizar = st.toggle("Mostrar por porcentaje de cada fila (recall por clase)", value=False)
        valores = matriz / matriz.sum(axis=1, keepdims=True) * 100 if normalizar else matriz
        texto = [[f"{v:.1f}" if normalizar else f"{int(v)}" for v in fila] for fila in valores]
        # Escala logarítmica de color para que los errores (fuera de la diagonal) se distingan
        fig_m = go.Figure(go.Heatmap(
            z=np.log10(valores + 1), text=texto, texttemplate="%{text}", colorscale="Blues", showscale=False,
            x=[str(d) for d in range(10)], y=[str(d) for d in range(10)],
            customdata=valores,
            hovertemplate="Real %{y} → Predicho %{x}<br>%{customdata:.1f}<extra></extra>"))
        fig_m.update_layout(height=520, xaxis_title="Predicción del modelo", yaxis_title="Dígito real",
                            yaxis_autorange="reversed", margin=dict(t=20, b=40))
        cm1, cm2 = st.columns([1.6, 1])
        with cm1:
            st.plotly_chart(fig_m, width="stretch")
        with cm2:
            st.markdown("**Números que más se confunden**")
            pares = pd.DataFrame(metricas.get("pares_mas_confundidos", []))
            if not pares.empty:
                pares["Confusión"] = pares.apply(lambda r: f"{r['real']} → {r['predicho']}", axis=1)
                st.dataframe(pares[["Confusión", "casos"]].rename(columns={"casos": "Casos"}).head(8),
                             hide_index=True, width="stretch")
            st.caption("La diagonal son los aciertos. Fuera de ella aparecen los errores: típicamente "
                       "4↔9 (el lazo superior), 3↔5↔8 (curvas parecidas) y 7↔1 o 7↔2 (trazos rectos).")

        figuras = sorted(config.DIR_FIGURAS.glob("*.png"))
        if figuras or config.RUTA_MATRIZ_PNG.exists():
            with st.expander("🖼️ Gráficas generadas en Google Colab"):
                for ruta in ([config.RUTA_MATRIZ_PNG] if config.RUTA_MATRIZ_PNG.exists() else []) + figuras:
                    st.image(str(ruta), caption=ruta.stem.replace("_", " ").capitalize())
    else:
        st.warning("No se encontró `recursos/metricas.json`. Ejecuta el cuaderno de Colab y copia los "
                   "artefactos en `app/` para ver métricas, comparativa y matriz de confusión.")

# -----------------------------------------------------------------------------
# SECCIÓN D — Historial de pruebas de la sesión
# -----------------------------------------------------------------------------
with tab_d:
    st.subheader("📋 Historial de pruebas de la sesión")
    est = hs.estadisticas()
    if est["total"] == 0:
        st.info("Aún no hay intentos. Cada vez que pulses **Reconocer** se registrará aquí.")
    else:
        e1, e2, e3, e4, e5 = st.columns(5)
        e1.metric("Intentos", est["total"])
        e2.metric("Confirmados", f"{est['confirmados']} / {est['total']}", help=f"{est['pendientes']} pendientes de confirmar")
        e3.metric("Aciertos", est["aciertos"])
        e4.metric("Precisión en vivo", f"{est['precision'] * 100:.0f} %" if est["precision"] is not None else "—")
        e5.metric("Tiempo medio", f"{est['tiempo_medio']:.4f} s")

        df = hs.como_dataframe()
        st.dataframe(df.iloc[::-1], hide_index=True, width="stretch", column_config={
            "Confianza": st.column_config.ProgressColumn(format="percent", min_value=0, max_value=1),
            "Conf. 2.ª": st.column_config.NumberColumn(format="percent"),
            "Tiempo (s)": st.column_config.NumberColumn(format="%.4f"),
        })

        st.markdown("#### ✔️ Confirmación de la docente")
        st.caption("Indica qué número se dibujó realmente (si no se dijo) y marca si la IA acertó.")
        encabezado = st.columns([0.5, 1.6, 1.1, 0.9, 1, 1.3, 0.7, 0.7])
        for col, texto in zip(encabezado, ["N.º", "Estilo / hora", "Dibujado", "IA dijo", "Confianza",
                                           "Resultado", "", ""]):
            col.markdown(f"**{texto}**")
        opciones = ["?"] + [str(d) for d in range(10)]
        for fila in reversed(hs.intentos()):
            c = st.columns([0.5, 1.6, 1.1, 0.9, 1, 1.3, 0.7, 0.7], vertical_alignment="center")
            c[0].markdown(f"**#{fila['id']}**")
            c[1].markdown(f"{fila['participante']}<br><span class='vd-nota'>{fila['hora']} · {fila['modo']}</span>",
                          unsafe_allow_html=True)
            clave = f"dibujado_{fila['id']}"
            ss[clave] = "?" if fila["dibujado"] is None else str(fila["dibujado"])   # sincroniza con el registro
            c[2].selectbox("Dibujado", opciones, key=clave, label_visibility="collapsed",
                           on_change=hs.fijar_dibujado, args=(fila["id"], clave))
            c[3].markdown(f"<span style='font-size:1.6rem;font-weight:700;color:#4F46E5'>{fila['prediccion']}</span>",
                          unsafe_allow_html=True)
            c[4].markdown(f"{fila['confianza'] * 100:.1f} %")
            c[5].markdown(fila["resultado"])
            c[6].button("✅", key=f"ok_{fila['id']}", on_click=hs.marcar, args=(fila["id"], True),
                        help="La IA acertó")
            c[7].button("❌", key=f"no_{fila['id']}", on_click=hs.marcar, args=(fila["id"], False),
                        help="La IA se equivocó")

        st.divider()
        g1, g2 = st.columns(2)
        with g1:
            st.markdown("#### 🔢 Precisión por número dibujado")
            pdig = est["por_digito"]
            if not pdig.empty:
                fig_pd = go.Figure(go.Bar(x=pdig["dibujado"].astype(str), y=pdig["precisión"] * 100,
                                          text=[f"{a}/{i}" for a, i in zip(pdig["aciertos"], pdig["intentos"])],
                                          textposition="outside", marker_color=config.COLOR_PRINCIPAL))
                fig_pd.update_layout(height=300, yaxis=dict(range=[0, 115], title="%"), xaxis_title="Dígito",
                                     plot_bgcolor="white", margin=dict(t=10, b=30))
                st.plotly_chart(fig_pd, width="stretch")
            else:
                st.caption("Confirma intentos con número conocido para ver esta gráfica.")
        with g2:
            st.markdown("#### ✍️ Comparación de estilos de escritura")
            pest = est["por_estilo"]
            if not pest.empty:
                st.dataframe(pest.rename(columns={"participante": "Estilo / participante",
                                                  "confianza_media": "Confianza media",
                                                  "tiempo_medio_s": "Tiempo medio (s)"}),
                             hide_index=True, width="stretch", column_config={
                                 "precisión": st.column_config.ProgressColumn("Precisión", format="percent",
                                                                              min_value=0, max_value=1),
                                 "Confianza media": st.column_config.NumberColumn(format="percent"),
                                 "Tiempo medio (s)": st.column_config.NumberColumn(format="%.4f")})
            else:
                st.caption("Escribe un nombre o estilo en la barra lateral y confirma intentos.")

        st.markdown("#### ⚠️ Errores frecuentes de la sesión")
        errs = est["errores_frecuentes"]
        if not errs.empty:
            errs = errs.assign(Error=lambda d: d["dibujado"].astype(str) + " → " + d["prediccion"].astype(str))
            st.dataframe(errs[["Error", "veces"]].rename(columns={"veces": "Veces"}), hide_index=True)
            peor = errs.iloc[0]
            st.caption(f"El error más común fue dibujar un **{peor['dibujado']}** y que la IA viera un "
                       f"**{peor['prediccion']}**. ¡Compáralo con la matriz de confusión de la sección C!")
        else:
            st.caption("Sin errores confirmados con número conocido. 🎉")

        h1, h2 = st.columns(2)
        h1.download_button("⬇️ Descargar historial (CSV)", df.to_csv(index=False).encode("utf-8-sig"),
                           file_name="visiondigits_historial.csv", mime="text/csv", width="stretch")
        h2.button("🗑️ Borrar historial", on_click=hs.borrar_todo, width="stretch")

with zona_estrellas.container():
    st.markdown(f"<div class='vd-estrellas'>{'⭐' * min(ss.estrellas_total, 15)}"
                f"{'…' if ss.estrellas_total > 15 else ''}</div>", unsafe_allow_html=True)
    st.metric("Estrellas ganadas", ss.estrellas_total)

st.markdown("<p class='vd-nota' style='text-align:center;margin-top:2rem'>VisionDigits · Proyecto académico "
            "de Inteligencia Artificial · CNN entrenada con MNIST (TensorFlow/Keras)</p>",
            unsafe_allow_html=True)
