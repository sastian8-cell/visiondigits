"""
historial.py — Módulo 6: Pruebas y estadísticas de la sesión.

Guarda cada intento en st.session_state (vive mientras la pestaña esté abierta)
y calcula aciertos, precisión en vivo, errores frecuentes y comparación por estilo.
"""
from __future__ import annotations

from datetime import datetime

import pandas as pd
import streamlit as st

import config

CLAVE = "historial_intentos"


def iniciar() -> None:
    st.session_state.setdefault(CLAVE, [])
    st.session_state.setdefault("siguiente_id", 1)


def intentos() -> list[dict]:
    return st.session_state[CLAVE]


def registrar(*, modo: str, fuente: str, participante: str, esperado: int | None, pred) -> int:
    """Añade un intento y devuelve su identificador."""
    ident = st.session_state["siguiente_id"]
    st.session_state["siguiente_id"] += 1
    intentos().append({
        "id": ident,
        "hora": datetime.now().strftime("%H:%M:%S"),
        "modo": modo,
        "fuente": fuente,
        "participante": participante or "Sin nombre",
        "dibujado": esperado,                    # número que se pretendía dibujar (None = no indicado)
        "prediccion": pred.digito,
        "confianza": round(pred.confianza, 4),
        "segunda": pred.segundo,
        "confianza_segunda": round(pred.confianza_segundo, 4),
        "tiempo_s": round(pred.tiempo_s, 4),
        "resultado": config.PENDIENTE,
    })
    return ident


def _buscar(ident: int) -> dict | None:
    return next((f for f in intentos() if f["id"] == ident), None)


def marcar(ident: int, correcto: bool) -> None:
    """Callback de los botones ✅/❌ de la docente."""
    fila = _buscar(ident)
    if fila is None:
        return
    fila["resultado"] = config.CORRECTO if correcto else config.INCORRECTO
    if correcto and fila["dibujado"] is None:      # si acertó, lo dibujado = lo predicho
        fila["dibujado"] = fila["prediccion"]
    if not correcto and fila["dibujado"] == fila["prediccion"]:
        fila["dibujado"] = None                     # contradicción: se pide corregir el número real


def fijar_dibujado(ident: int, clave_widget: str) -> None:
    """Callback del selector 'número dibujado'."""
    fila = _buscar(ident)
    valor = st.session_state.get(clave_widget)
    if fila is None:
        return
    fila["dibujado"] = None if valor in (None, "?") else int(valor)
    # Si ya se conoce el número real, el resultado se deduce automáticamente
    if fila["dibujado"] is not None and fila["resultado"] != config.PENDIENTE:
        fila["resultado"] = config.CORRECTO if fila["dibujado"] == fila["prediccion"] else config.INCORRECTO


def borrar_todo() -> None:
    st.session_state[CLAVE] = []
    st.session_state["siguiente_id"] = 1


def como_dataframe() -> pd.DataFrame:
    """Historial con nombres de columna legibles."""
    df = pd.DataFrame(intentos())
    if df.empty:
        return df
    df["dibujado"] = df["dibujado"].apply(lambda v: "?" if v is None or pd.isna(v) else str(int(v)))
    return df.rename(columns={
        "id": "N.º", "hora": "Hora", "modo": "Modo", "fuente": "Fuente", "participante": "Estilo / participante",
        "dibujado": "Número dibujado", "prediccion": "Predicción", "confianza": "Confianza",
        "segunda": "2.ª opción", "confianza_segunda": "Conf. 2.ª", "tiempo_s": "Tiempo (s)",
        "resultado": "Resultado",
    })


def estadisticas() -> dict:
    """Indicadores agregados de la sesión."""
    df = pd.DataFrame(intentos())
    if df.empty:
        return {"total": 0}
    confirmados = df[df["resultado"] != config.PENDIENTE]
    aciertos = int((confirmados["resultado"] == config.CORRECTO).sum())

    errores = confirmados[(confirmados["resultado"] == config.INCORRECTO) & confirmados["dibujado"].notna()]
    errores_frec = (
        errores.groupby(["dibujado", "prediccion"]).size().reset_index(name="veces")
        .sort_values("veces", ascending=False)
        if not errores.empty else pd.DataFrame(columns=["dibujado", "prediccion", "veces"])
    )
    if not errores_frec.empty:
        errores_frec["dibujado"] = errores_frec["dibujado"].astype(int)

    por_estilo = pd.DataFrame()
    if not confirmados.empty:
        por_estilo = confirmados.groupby("participante").agg(
            intentos=("id", "count"),
            aciertos=("resultado", lambda s: int((s == config.CORRECTO).sum())),
            confianza_media=("confianza", "mean"),
            tiempo_medio_s=("tiempo_s", "mean"),
        ).reset_index()
        por_estilo["precisión"] = por_estilo["aciertos"] / por_estilo["intentos"]

    con_digito = confirmados[confirmados["dibujado"].notna()].copy()
    por_digito = pd.DataFrame()
    if not con_digito.empty:
        con_digito["dibujado"] = con_digito["dibujado"].astype(int)
        por_digito = con_digito.groupby("dibujado").agg(
            intentos=("id", "count"),
            aciertos=("resultado", lambda s: int((s == config.CORRECTO).sum())),
        ).reset_index()
        por_digito["precisión"] = por_digito["aciertos"] / por_digito["intentos"]

    return {
        "total": len(df),
        "confirmados": len(confirmados),
        "pendientes": len(df) - len(confirmados),
        "aciertos": aciertos,
        "precision": aciertos / len(confirmados) if len(confirmados) else None,
        "tiempo_medio": float(df["tiempo_s"].mean()),
        "confianza_media": float(df["confianza"].mean()),
        "errores_frecuentes": errores_frec,
        "por_estilo": por_estilo,
        "por_digito": por_digito,
    }
