"""
AIRHSP · Pronóstico del estado predominante futuro de perfiles institucional-laborales
Prototipo académico (Hito 3) — UTEC · Ingeniería Industrial

Ejecutar:
    pip install -r requirements.txt
    streamlit run app.py

Archivos esperados en la misma carpeta que app.py (salen del zip
AIRHSP_para_prototipo.zip que exporta la última celda del notebook):
    modelo_campeon_pipeline.joblib      <- obligatorio para pronosticar
    ejemplos_prototipo.csv              <- recomendado (perfiles reales del test)
    decisiones_antes_test.json          <- opcional (umbrales congelados)
    versiones_entrenamiento.json        <- opcional (versiones de librerías)
    resultados_no_supervisado/medianas_clusters.csv  <- opcional
"""

from __future__ import annotations

import io
import json
from pathlib import Path

import re
import sys

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, RobustScaler


# ---------------------------------------------------------------------------
# Clase personalizada del notebook (Hito 2, sección 0).
# El .joblib guarda solo la referencia "__main__.PreparadorAIRHSP", así que
# la clase debe existir con el mismo código para poder abrir el modelo.
# ---------------------------------------------------------------------------
class PreparadorAIRHSP(TransformerMixin, BaseEstimator):
    """Aprende todas las estadísticas en fit; nunca sobre validación o test."""
    def fit(self, X, y=None):
        self.cols_ = list(X.columns)
        self.num_ = [c for c in X if c.endswith('_PER_CAPITA') or c in ['CANTIDAD_TOTAL', 'MES_SIN', 'MES_COS']]
        self.freq_ = [c for c in ['PLIEGO', 'UNIDAD_EJECUTORA', 'DESC_CARGO_ESTRUCUTURAL'] if c in X]
        self.cat_ = [c for c in X if c not in self.num_ + self.freq_ + ['ESTADO_REGISTRO']]
        self.estado_ = 'ESTADO_REGISTRO' in X
        if self.num_:
            z = X[self.num_].apply(pd.to_numeric, errors='coerce').replace([np.inf, -np.inf], np.nan)
            self.med_ = z.median().fillna(0)
            self.scaler_ = RobustScaler().fit(z.fillna(self.med_))
        self.maps_ = {c: X[c].fillna('SIN_DATO').astype(str).value_counts(normalize=True).to_dict() for c in self.freq_}
        if self.cat_:
            self.ohe_ = OneHotEncoder(handle_unknown='ignore', sparse_output=False, dtype=np.float32)
            self.ohe_.fit(X[self.cat_].fillna('SIN_DATO').astype(str))
        names = self.num_ + [c + '_FREQ' for c in self.freq_]
        if self.cat_:
            names += list(self.ohe_.get_feature_names_out(self.cat_))
        if self.estado_:
            names += ['ESTADO_REGISTRO_VACANTE']
        self.names_ = np.array([f'f{i:04d}_' + re.sub('[^0-9A-Za-z_]', '_', c) for i, c in enumerate(names)])
        return self

    def transform(self, X):
        if set(X.columns) != set(self.cols_):
            raise ValueError('Esquema distinto al de fit.')
        parts = []
        if self.num_:
            z = X[self.num_].apply(pd.to_numeric, errors='coerce').replace([np.inf, -np.inf], np.nan).fillna(self.med_)
            parts.append(self.scaler_.transform(z).astype(np.float32))
        for c in self.freq_:
            parts.append(X[c].fillna('SIN_DATO').astype(str).map(self.maps_[c]).fillna(0).to_numpy(np.float32)[:, None])
        if self.cat_:
            parts.append(self.ohe_.transform(X[self.cat_].fillna('SIN_DATO').astype(str)))
        if self.estado_:
            if not X.ESTADO_REGISTRO.isin(['OCUPADO', 'VACANTE']).all():
                raise ValueError('Estado actual inválido.')
            parts.append(X.ESTADO_REGISTRO.eq('VACANTE').to_numpy(np.float32)[:, None])
        out = np.concatenate(parts, axis=1)
        if not np.isfinite(out).all():
            raise ValueError('Valores no finitos tras preparación.')
        return out

    def get_feature_names_out(self, input_features=None):
        return self.names_.copy()


def crear_pipeline(modelo):
    return Pipeline([('prep', PreparadorAIRHSP()), ('modelo', modelo)])


# Streamlit no ejecuta app.py como __main__: se registra la clase allí
# para que joblib/pickle la encuentre al deserializar el modelo.
sys.modules['__main__'].PreparadorAIRHSP = PreparadorAIRHSP
sys.modules['__main__'].crear_pipeline = crear_pipeline

# ---------------------------------------------------------------------------
# Configuración general
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="AIRHSP · Pronóstico de vacancia",
    page_icon="🏛️",
    layout="wide",
    initial_sidebar_state="expanded",
)

BASE = Path(__file__).parent
RUTA_MODELO = BASE / "modelo_campeon_pipeline.joblib"
RUTA_EJEMPLOS = BASE / "ejemplos_prototipo.csv"
RUTA_DECISIONES = BASE / "decisiones_antes_test.json"
RUTA_VERSIONES = BASE / "versiones_entrenamiento.json"
RUTA_MEDIANAS_NS = BASE / "resultados_no_supervisado" / "medianas_clusters.csv"

# Paleta (la misma del Hito 3)
TEAL, NARANJA, AZUL, GRIS = "#146c67", "#d58c36", "#447c9e", "#a9bdcb"
ROJO, VERDE = "#c0504d", "#2e8b57"

# --- Variables del panel (FEATURES_PANEL del Hito 1, mismo orden) ----------
MONETARIAS = [
    "ING_IMPONIBLE_PERM_MENSUAL_PER_CAPITA",
    "ING_NO_IMPONIBLE_PERM_MENSUAL_PER_CAPITA",
    "INCENTIVO_UNICO_MENSUAL_PER_CAPITA",
    "GUARDIAS_HOSPITALARIAS_MENSUAL_PER_CAPITA",
    "ESCOLARIDAD_ANUAL_PER_CAPITA",
    "AGUINALDO_GRATIFICACION_ANUAL_PER_CAPITA",
    "OTROS_ING_OCASIONALES_ANUAL_PER_CAPITA",
    "APORTE_PERM_MENSUAL_PER_CAPITA",
    "APORTE_OCASIONAL_ANUAL_PER_CAPITA",
    "COSTO_TOTAL_ANUAL_PER_CAPITA",
]
ALTA_CARD = ["PLIEGO", "DESC_CARGO_ESTRUCUTURAL", "UNIDAD_EJECUTORA"]
ONE_HOT = [
    "NIVEL", "SECTOR", "TIPO_ESTABLECIMIENTO", "DESC_TIPO_REGISTRO",
    "DESC_SUB_TIPO_REGISTRO", "DESC_REGIMEN_LABORAL", "DESC_GRUPO_OCASIONAL",
    "DESC_CONDICION_LABORAL", "ESTADO_REGISTRO",
]
FEATURES = MONETARIAS + ["CANTIDAD_TOTAL", "MES_SIN", "MES_COS"] + ALTA_CARD + ONE_HOT

ETIQUETAS = {
    "ING_IMPONIBLE_PERM_MENSUAL_PER_CAPITA": "Ingreso imponible mensual",
    "ING_NO_IMPONIBLE_PERM_MENSUAL_PER_CAPITA": "Ingreso no imponible mensual",
    "INCENTIVO_UNICO_MENSUAL_PER_CAPITA": "Incentivo único mensual",
    "GUARDIAS_HOSPITALARIAS_MENSUAL_PER_CAPITA": "Guardias hospitalarias mensual",
    "ESCOLARIDAD_ANUAL_PER_CAPITA": "Escolaridad anual",
    "AGUINALDO_GRATIFICACION_ANUAL_PER_CAPITA": "Aguinaldo / gratificación anual",
    "OTROS_ING_OCASIONALES_ANUAL_PER_CAPITA": "Otros ingresos ocasionales anual",
    "APORTE_PERM_MENSUAL_PER_CAPITA": "Aporte permanente mensual",
    "APORTE_OCASIONAL_ANUAL_PER_CAPITA": "Aporte ocasional anual",
    "COSTO_TOTAL_ANUAL_PER_CAPITA": "Costo total anual",
    "CANTIDAD_TOTAL": "Cantidad total del perfil",
    "MES_SIN": "Mes (seno)", "MES_COS": "Mes (coseno)", "MES": "Mes del periodo",
    "PLIEGO": "Pliego", "DESC_CARGO_ESTRUCUTURAL": "Cargo estructural",
    "UNIDAD_EJECUTORA": "Unidad ejecutora", "NIVEL": "Nivel de gobierno",
    "SECTOR": "Sector", "TIPO_ESTABLECIMIENTO": "Tipo de establecimiento",
    "DESC_TIPO_REGISTRO": "Tipo de registro", "DESC_SUB_TIPO_REGISTRO": "Subtipo de registro",
    "DESC_REGIMEN_LABORAL": "Régimen laboral", "DESC_GRUPO_OCASIONAL": "Grupo ocupacional",
    "DESC_CONDICION_LABORAL": "Condición laboral", "ESTADO_REGISTRO": "Estado actual (t)",
}

# Ayudas que aparecen al pasar el mouse por el ícono ❓ de cada campo
AYUDAS = {
    "ESTADO_REGISTRO": "Cómo está el perfil este mes (t). OCUPADO = la mayor parte de su cantidad está ocupada; VACANTE = la mayor parte está vacante.",
    "MES": "Mes al que corresponden los datos. El modelo pronostica el mes siguiente (t+1).",
    "CANTIDAD_TOTAL": "Cuántas plazas/registros agrupa este perfil en el mes (suma de CANTIDAD del AIRHSP). Grupos pequeños cambian de estado con más facilidad.",
    "NIVEL": "Nivel de gobierno al que pertenece: nacional o regional.",
    "SECTOR": "Sector del Estado (Educación, Salud, Interior, gobiernos regionales, etc.).",
    "PLIEGO": "Entidad que tiene el presupuesto (ministerio, gobierno regional, etc.).",
    "UNIDAD_EJECUTORA": "Oficina o dependencia concreta del pliego que paga la planilla.",
    "TIPO_ESTABLECIMIENTO": "Tipo de establecimiento donde está la plaza (general, hospital, colegio, etc.).",
    "DESC_TIPO_REGISTRO": "Tipo de registro en el AIRHSP: Activos, CAS, Pensionistas, Sobrevivientes, etc.",
    "DESC_SUB_TIPO_REGISTRO": "Detalle del tipo de registro.",
    "DESC_REGIMEN_LABORAL": "Régimen laboral (D. Leg. 276, 728, Carreras Especiales, CAS, etc.).",
    "DESC_GRUPO_OCASIONAL": "Grupo ocupacional: profesionales, técnicos, auxiliares, etc.",
    "DESC_CARGO_ESTRUCUTURAL": "Cargo de la plaza según la estructura de la entidad.",
    "DESC_CONDICION_LABORAL": "Condición de la relación laboral (nombrado, contratado, designado, viudez, etc.).",
}
AYUDA_MONTO = ("Monto por unidad: el total del grupo dividido entre su cantidad. "
               "Pon 0 si el perfil no recibe este concepto.")

MESES = ["Enero", "Febrero", "Marzo", "Abril", "Mayo", "Junio", "Julio",
         "Agosto", "Septiembre", "Octubre", "Noviembre", "Diciembre"]

# --- Umbrales fijados en validación (sección 10 del Hito 2) ----------------
UMBRALES = {
    "Máximo F1 (alertas precisas)": 0.5243,
    "Máximo F2 (prioriza recall)": 0.0656,
    "Recall ≥ 0.80": 0.0604,
    "Mínimo costo (FN=100, FP=5)": 0.0266,
}
UMBRALES_VAL = pd.DataFrame([
    ("Máximo F1", 0.5243, 0.7907, 0.6996, 84630),
    ("Máximo F2", 0.0656, 0.5907, 0.7958, 63335),
    ("Recall ≥ 0.80", 0.0604, 0.5755, 0.8002, 62665),
    ("Mínimo costo", 0.0266, 0.3957, 0.8675, 54305),
], columns=["Criterio", "Umbral", "Precisión", "Recall", "Costo validación"])

# --- Resultados guardados del notebook (instantánea H3_DATOS) --------------
TEST = pd.DataFrame([
    ("Regresión Logística", 0.6000, 0.9748, 0.6474, 0.8413, 0.7317, 0.7333),
    ("Árbol de decisión", 0.7322, 0.9793, 0.6837, 0.8179, 0.7448, 0.7434),
    ("Random Forest", 0.7028, 0.9807, 0.6426, 0.7963, 0.7112, 0.7103),
    ("XGBoost", 0.7619, 0.9820, 0.7303, 0.7577, 0.7437, 0.7397),
    ("LightGBM", 0.7648, 0.9829, 0.6940, 0.7976, 0.7422, 0.7396),
    ("CatBoost", 0.7707, 0.9813, 0.7197, 0.7695, 0.7438, 0.7400),
    ("Random Forest (ajustado)", 0.7567, 0.9806, 0.7459, 0.7053, 0.7251, 0.7211),
    ("XGBoost (ajustado)", 0.7789, 0.9833, 0.7005, 0.7940, 0.7443, 0.7415),
    ("LightGBM (ajustado)", 0.7831, 0.9830, 0.8074, 0.6584, 0.7254, 0.7253),
    ("CatBoost (ajustado)", 0.7736, 0.9826, 0.7327, 0.7532, 0.7428, 0.7387),
    ("Persistencia", 0.5526, 0.9211, 0.6477, 0.8496, 0.7350, 0.7372),
    ("Dummy", 0.0158, 0.5000, 0.0, 0.0, 0.0, 0.0),
], columns=["Modelo", "AP test", "ROC-AUC test", "Precisión", "Recall", "F1", "MCC"])

AP_VAL = {
    "LightGBM (ajustado)": 0.7637, "XGBoost (ajustado)": 0.738, "LightGBM": 0.7292,
    "Random Forest (ajustado)": 0.72, "XGBoost": 0.7171, "Random Forest": 0.716,
    "CatBoost (ajustado)": 0.713, "CatBoost": 0.709, "Árbol de decisión": 0.677,
    "Regresión Logística": 0.673, "Persistencia": 0.558, "Dummy": 0.027,
}
TEST["AP validación"] = TEST["Modelo"].map(AP_VAL)

ABLACION = pd.DataFrame([
    ("Todas las variables", 0.7501), ("Sin condición laboral", 0.7520),
    ("Sin CANTIDAD_TOTAL", 0.7428), ("Sin alta cardinalidad", 0.7414),
    ("Sin otras categóricas", 0.7347), ("Sin monetarias per cápita", 0.7350),
    ("Sin estacionalidad (mes)", 0.7104), ("Sin estado actual (t)", 0.5799),
    ("SOLO estado actual", 0.5584),
], columns=["Configuración", "AP validación"])

COSTOS = pd.DataFrame([
    ("Siempre OCUPADO", 531100), ("LightGBM · máx. F1", 185570),
    ("LightGBM · mín. costo", 97820), ("Persistencia", 92170),
], columns=["Alternativa", "Costo test"])

CLUSTERS = pd.DataFrame({
    "Segmento": [f"S{i}" for i in range(8)],
    "Perfiles": [12500, 25313, 1815, 11270, 4110, 75, 179, 1220],
})
METODOS_NS = pd.DataFrame([
    ("K-Means", 8, 0.0, 0.5707, 0.5300), ("Ward", 8, 0.0, 0.5625, 0.5457),
    ("DBSCAN", 4, 22.96, 0.6494, 0.4884),
], columns=["Método", "Grupos", "Ruido %", "Silhouette", "Davies-Bouldin"])

# --- Métricas de validación (umbral máx. F1 de cada modelo) · instantánea H3_DATOS ---
VAL = pd.DataFrame([
    ("LightGBM (ajustado)", 0.7637, 0.969, 0.791, 0.700, 0.742, 0.737),
    ("XGBoost (ajustado)", 0.738, 0.969, 0.763, 0.722, 0.742, 0.735),
    ("LightGBM", 0.7292, 0.967, 0.764, 0.722, 0.742, 0.736),
    ("Random Forest (ajustado)", 0.720, 0.955, 0.765, 0.720, 0.742, 0.735),
    ("XGBoost", 0.7171, 0.966, 0.766, 0.715, 0.739, 0.733),
    ("Random Forest", 0.716, 0.960, 0.746, 0.689, 0.717, 0.709),
    ("CatBoost (ajustado)", 0.713, 0.964, 0.767, 0.716, 0.740, 0.734),
    ("CatBoost", 0.709, 0.963, 0.757, 0.714, 0.735, 0.728),
    ("Árbol de decisión", 0.677, 0.949, 0.765, 0.720, 0.742, 0.735),
    ("Regresión Logística", 0.673, 0.960, 0.761, 0.711, 0.735, 0.728),
    ("Persistencia", 0.558, 0.857, 0.765, 0.720, 0.742, 0.735),
    ("Dummy", 0.027, 0.500, 0.027, 1.000, 0.053, 0.000),
], columns=["Modelo", "AP", "ROC-AUC", "Precisión", "Recall", "F1", "MCC"])
TEST_M = TEST.rename(columns={"AP test": "AP", "ROC-AUC test": "ROC-AUC"})[
    ["Modelo", "AP", "ROC-AUC", "Precisión", "Recall", "F1", "MCC"]]

EXPLICA_METRICA = {
    "AP": "Average Precision: qué tan bien ordena los perfiles de mayor a menor riesgo (1 = perfecto).",
    "Precisión": "De todas las alertas VACANTE, qué proporción acierta.",
    "Recall": "De todas las vacantes reales, qué proporción detecta.",
    "F1": "Equilibrio entre precisión y recall.",
    "MCC": "Calidad global de la clasificación, robusta al desbalance (1 = perfecto, 0 = azar).",
}

# --- Conteos de errores en test para el simulador de costos -------------------
N_VAC_TEST = 5311
def _conteos(prec, rec):
    tp = round(rec * N_VAC_TEST)
    fp = round(tp / prec - tp) if prec > 0 else 0
    return N_VAC_TEST - tp, fp

ERRORES_TEST = {"Siempre OCUPADO": (N_VAC_TEST, 0),
                "Persistencia": (799, 2454),            # exactos (costo 92,170 con 100/5)
                "LightGBM (ajustado)": (1814, 834)}     # exactos de la matriz de confusión
for _, r in TEST.iterrows():
    if r.Modelo not in ERRORES_TEST and r.Modelo != "Dummy":
        ERRORES_TEST[r.Modelo] = _conteos(r["Precisión"], r["Recall"])

EXPLICA_ABLACION = {
    "Sin estado actual (t)": "Es la variable más importante: sin saber si hoy está vacante, el modelo pierde gran parte de su capacidad.",
    "SOLO estado actual": "Usando solo el estado actual se obtiene 0.56: el resto de variables suma bastante información.",
    "Sin estacionalidad (mes)": "El mes importa: hay épocas del año con más movimientos de plazas.",
    "Sin monetarias per cápita": "Los montos por plaza aportan algo de señal.",
    "Sin otras categóricas": "Sector, régimen, tipo de registro, etc. aportan algo de señal.",
    "Sin alta cardinalidad": "Pliego, unidad ejecutora y cargo aportan poca señal adicional.",
    "Sin CANTIDAD_TOTAL": "El tamaño del grupo aporta poca señal adicional.",
    "Sin condición laboral": "Quitarla casi no cambia (incluso sube 0.002): su aporte es mínimo.",
}

# --- Segmentación no supervisada ---------------------------------------------
CARPETA_NS = BASE / "resultados_no_supervisado"
NOMBRE_VAR_SEG = {
    "CANTIDAD_TOTAL": "Grupos grandes",
    "GUARDIAS_HOSPITALARIAS_MENSUAL_PER_CAPITA": "Salud con guardias",
    "INCENTIVO_UNICO_MENSUAL_PER_CAPITA": "Con incentivo único",
    "ING_NO_IMPONIBLE_PERM_MENSUAL_PER_CAPITA": "Ingreso no imponible alto",
    "ING_IMPONIBLE_PERM_MENSUAL_PER_CAPITA": "Sueldo imponible alto",
    "ESCOLARIDAD_ANUAL_PER_CAPITA": "Con escolaridad",
    "AGUINALDO_GRATIFICACION_ANUAL_PER_CAPITA": "Con aguinaldo alto",
    "OTROS_ING_OCASIONALES_ANUAL_PER_CAPITA": "Ingresos ocasionales",
    "APORTE_PERM_MENSUAL_PER_CAPITA": "Aportes altos",
    "APORTE_OCASIONAL_ANUAL_PER_CAPITA": "Aportes ocasionales",
    "COSTO_TOTAL_ANUAL_PER_CAPITA": "Costo alto",
}


def leer_csv_ns(nombre: str, **kw):
    ruta = CARPETA_NS / nombre
    if ruta.exists():
        try:
            return pd.read_csv(ruta, **kw)
        except Exception:
            return None
    return None


def rango_relativo(med: pd.DataFrame) -> pd.DataFrame:
    """0 = el segmento con menor mediana en esa variable, 1 = el mayor (igual que el notebook)."""
    r = med.rank(axis=0, method="average")
    return r.sub(r.min()).div((r.max() - r.min()).replace(0, np.nan)).fillna(0.5)


def nombrar_segmentos(med: pd.DataFrame) -> dict:
    """Nombre automático según la variable en la que cada segmento más se distingue."""
    rel = rango_relativo(med)
    nombres, usados = {}, set()
    for c in rel.index:
        fila = rel.loc[c]
        otros = rel.drop(index=c).mean()
        orden = (fila - otros).sort_values(ascending=False).index
        if fila.mean() >= 0.75:
            candidatos = ["Alto costo general"]
        elif fila.mean() <= 0.2:
            candidatos = ["Montos bajos"]
        else:
            candidatos = []
        candidatos += [NOMBRE_VAR_SEG.get(v, v) for v in orden]
        nombre = next((n for n in candidatos if n not in usados), f"Segmento {c}")
        usados.add(nombre)
        nombres[c] = f"S{c} · {nombre}"
    return nombres


@st.cache_resource
def cargar_segmentacion():
    ruta = CARPETA_NS / "segmentacion_kmeans.joblib"
    if not ruta.exists():
        return None
    try:
        return joblib.load(ruta)
    except Exception:
        return None


def asignar_segmento(df: pd.DataFrame):
    seg = cargar_segmentacion()
    if not seg:
        return None
    try:
        X = df[seg["variables"]].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
        X = X.fillna(seg["medianas"])
        Z = np.log1p(X) if seg.get("log1p", True) else X
        Z = seg["pca"].transform(seg["scaler"].transform(Z))
        return seg["kmeans"].predict(Z)
    except Exception:
        return None



# ---------------------------------------------------------------------------
# Estilos
# ---------------------------------------------------------------------------
st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=Fraunces:opsz,wght@9..144,500;9..144,700&display=swap');
html, body, [class*="css"], .stMarkdown, .stText {{ font-family: 'Inter', sans-serif; }}
.block-container {{ padding-top: 1.6rem; padding-bottom: 3rem; max-width: 1250px; }}
h1, h2, h3 {{ font-family: 'Fraunces', serif !important; letter-spacing: -0.01em; }}

.hero {{
  background: radial-gradient(circle at 85% 10%, rgba(213,140,54,.35), transparent 40%),
              linear-gradient(135deg, #0d3f3c 0%, {TEAL} 55%, {AZUL} 100%);
  border-radius: 22px; padding: 2.4rem 2.6rem; color: #fff; margin-bottom: 1.4rem;
  box-shadow: 0 18px 40px -18px rgba(13,63,60,.55);
}}
.hero .kicker {{ text-transform: uppercase; letter-spacing: .18em; font-size: .75rem; opacity: .8; }}
.hero h1 {{ color: #fff; font-size: 2.35rem; margin: .35rem 0 .6rem 0; line-height: 1.15; }}
.hero p {{ font-size: 1.04rem; opacity: .92; max-width: 760px; margin: 0; }}

.card {{
  background: var(--secondary-background-color); border-radius: 16px; padding: 1.15rem 1.25rem;
  border: 1px solid rgba(128,128,128,.14); height: 100%;
}}
.card .lbl {{ font-size: .78rem; text-transform: uppercase; letter-spacing: .08em; opacity: .65; }}
.card .val {{ font-family: 'Fraunces', serif; font-size: 1.9rem; font-weight: 700; color: {TEAL}; line-height: 1.2; }}
.card .sub {{ font-size: .82rem; opacity: .7; }}

.step {{ border-left: 3px solid {NARANJA}; padding: .2rem 0 .2rem .9rem; margin-bottom: .9rem; }}
.step b {{ color: {TEAL}; }}

.verdict {{ border-radius: 18px; padding: 1.4rem 1.5rem; color: #fff; }}
.verdict.vac {{ background: linear-gradient(135deg, #8f2f2b, {ROJO}); }}
.verdict.ocu {{ background: linear-gradient(135deg, #0d3f3c, {TEAL}); }}
.verdict .big {{ font-family: 'Fraunces', serif; font-size: 2rem; font-weight: 700; }}
.pill {{ display:inline-block; padding:.18rem .65rem; border-radius:999px; font-size:.78rem;
        background: rgba(255,255,255,.18); margin-right:.35rem; margin-top:.4rem; }}
.note {{ font-size: .85rem; opacity: .75; }}
div[data-testid="stTabs"] button p {{ font-size: .98rem; font-weight: 500; }}
section[data-testid="stSidebar"] {{ background: linear-gradient(180deg, rgba(20,108,103,.08), transparent 60%); }}
ul[data-testid="stSelectboxVirtualDropdown"] li, div[data-baseweb="popover"] li {{
  white-space: normal !important; height: auto !important; min-height: 2.4rem; line-height: 1.3; }}
div[data-baseweb="popover"] li div {{ white-space: normal !important; overflow: visible !important; text-overflow: clip !important; }}
.rc-head {{ font-size: .95rem; opacity: .75; margin: .6rem 0 .6rem 0; letter-spacing: .02em; }}
.rc-grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 1rem; }}
.rc-card {{ border-radius: 18px; padding: 1.3rem 1.4rem 1.1rem; position: relative;
           border: 1px solid rgba(128,128,128,.18); background: var(--secondary-background-color);
           transition: all .2s ease; }}
.rc-top {{ display: flex; justify-content: space-between; align-items: center; }}
.rc-name {{ font-size: .8rem; font-weight: 600; letter-spacing: .14em; }}
.rc-badge {{ font-size: .7rem; font-weight: 600; letter-spacing: .06em; text-transform: uppercase;
            padding: .22rem .6rem; border-radius: 999px; background: rgba(255,255,255,.22); color: #fff; }}
.rc-pct {{ font-family: 'Fraunces', serif; font-size: 3rem; font-weight: 700; line-height: 1.1; margin: .35rem 0 .7rem; }}
.rc-pct small {{ font-size: 1.4rem; margin-left: .15rem; opacity: .8; }}
.rc-track {{ height: 6px; border-radius: 999px; background: rgba(128,128,128,.2); overflow: hidden; }}
.rc-fill {{ height: 100%; border-radius: 999px; }}
.rc-sub {{ font-size: .8rem; opacity: .7; margin-top: .55rem; }}
.rc-lose {{ opacity: .8; }}
.rc-lose.rc-ocu .rc-name, .rc-lose.rc-ocu .rc-pct {{ color: #6fbcb4; }}
.rc-lose.rc-vac .rc-name, .rc-lose.rc-vac .rc-pct {{ color: #e4928e; }}
.rc-lose.rc-ocu .rc-fill {{ background: #6fbcb4; }}
.rc-lose.rc-vac .rc-fill {{ background: #e4928e; }}
.rc-win {{ color: #fff; border: none; }}
.rc-win.rc-ocu {{ background: linear-gradient(135deg, #0d3f3c, {TEAL}); box-shadow: 0 14px 34px -16px rgba(20,108,103,.8); }}
.rc-win.rc-vac {{ background: linear-gradient(135deg, #8f2f2b, {ROJO}); box-shadow: 0 14px 34px -16px rgba(192,80,77,.8); }}
.rc-win .rc-track {{ background: rgba(255,255,255,.25); }}
.rc-win .rc-fill {{ background: #fff; }}
.rc-foot {{ font-size: .82rem; opacity: .7; margin-top: .75rem; }}
</style>
""", unsafe_allow_html=True)

# Estilos de las secciones rediseñadas (sin f-string para no duplicar llaves)
st.markdown("""
<style>
.card .val { color: #6fc2b8 !important; }
.sec-title { font-family: 'Fraunces', serif; font-size: 1.45rem; font-weight: 700; margin: 1.6rem 0 .3rem; }
.sec-sub { opacity: .7; font-size: .92rem; margin-bottom: .9rem; }
.grid { display: grid; gap: .9rem; }
.g2 { grid-template-columns: repeat(2, 1fr); } .g3 { grid-template-columns: repeat(3, 1fr); }
.g4 { grid-template-columns: repeat(4, 1fr); } .g5 { grid-template-columns: repeat(5, 1fr); }
@media (max-width: 900px) { .g3, .g4, .g5 { grid-template-columns: repeat(2, 1fr); } }
.icard { background: var(--secondary-background-color); border: 1px solid rgba(128,128,128,.15);
         border-radius: 16px; padding: 1rem 1.1rem; height: 100%; }
.icard .ico { font-size: 1.6rem; line-height: 1; }
.icard .t { font-weight: 600; margin: .5rem 0 .25rem; font-size: .98rem; }
.icard .d { font-size: .86rem; opacity: .78; line-height: 1.45; }
.icard .n { font-family: 'Fraunces', serif; font-size: 1.5rem; font-weight: 700; color: #6fc2b8; margin-top: .3rem; }
.icard.accent { border-left: 4px solid #d58c36; }
.ex { display: flex; align-items: stretch; gap: .8rem; flex-wrap: wrap; }
.ex .box { flex: 1; min-width: 180px; border-radius: 16px; padding: 1rem 1.2rem; border: 1px solid rgba(128,128,128,.18);
           background: var(--secondary-background-color); }
.ex .box .k { font-size: .72rem; text-transform: uppercase; letter-spacing: .12em; opacity: .65; }
.ex .box .v { font-size: 1.05rem; font-weight: 600; margin-top: .35rem; }
.ex .arrow { display: flex; align-items: center; font-size: 1.6rem; opacity: .6; }
.ex .ocu { background: linear-gradient(135deg, #0d3f3c, #146c67); color: #fff; border: none; }
.ex .q { background: linear-gradient(135deg, #6b4316, #d58c36); color: #fff; border: none; }
.chips { display: flex; flex-wrap: wrap; gap: .4rem; margin-top: .55rem; }
.chip { font-size: .78rem; padding: .25rem .6rem; border-radius: 999px; background: rgba(111,194,184,.14);
        border: 1px solid rgba(111,194,184,.3); }
.cm { display: grid; grid-template-columns: 110px 1fr 1fr; gap: .5rem; align-items: stretch; }
.cm .h { font-size: .78rem; opacity: .7; text-align: center; align-self: end; padding-bottom: .2rem; }
.cm .r { font-size: .8rem; opacity: .75; display: flex; flex-direction: column; align-items: flex-end; justify-content: center; text-align: right; line-height: 1.35; }
.cm .c { border-radius: 14px; padding: .9rem 1rem; color: #fff; }
.cm .c .lb { font-size: .82rem; font-weight: 600; }
.cm .c .num { font-family: 'Fraunces', serif; font-size: 1.7rem; font-weight: 700; margin: .2rem 0; }
.cm .c .pc { font-size: .78rem; opacity: .85; }
.cm .ok { background: linear-gradient(135deg, #0d3f3c, #146c67); }
.cm .bad { background: linear-gradient(135deg, #6e2a27, #c0504d); }
.cm .warn { background: linear-gradient(135deg, #6b4316, #b9782d); }
.lim { border-left: 3px solid #d58c36; padding: .5rem 0 .5rem .9rem; margin-bottom: .7rem; }
.lim b { display: block; margin-bottom: .15rem; }
.lim span { font-size: .88rem; opacity: .78; }
</style>
""", unsafe_allow_html=True)



def card_txt(label: str, valor: str, sub: str = "") -> str:
    """Tarjeta para valores de texto (más pequeños que una cifra)."""
    return (f'<div class="card"><div class="lbl">{label}</div>'
            f'<div class="val" style="font-size:1.25rem;line-height:1.3;margin:.25rem 0">{valor}</div>'
            f'<div class="sub">{sub}</div></div>')


def card(label: str, valor: str, sub: str = "") -> str:
    return f'<div class="card"><div class="lbl">{label}</div><div class="val">{valor}</div><div class="sub">{sub}</div></div>'


def estilo_fig(fig: go.Figure, alto: int = 380) -> go.Figure:
    fig.update_layout(
        height=alto, margin=dict(l=10, r=10, t=50, b=10),
        font=dict(family="Inter"), paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        legend=dict(orientation="h", y=-0.15),
    )
    fig.update_xaxes(gridcolor="rgba(128,128,128,.15)")
    fig.update_yaxes(gridcolor="rgba(128,128,128,.15)")
    return fig


# ---------------------------------------------------------------------------
# Carga de artefactos
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner="Cargando el modelo campeón…")
def cargar_modelo(origen):
    return joblib.load(origen)


@st.cache_data
def cargar_ejemplos() -> pd.DataFrame | None:
    if RUTA_EJEMPLOS.exists():
        return pd.read_csv(RUTA_EJEMPLOS)
    return None


@st.cache_data(show_spinner="Cargando perfiles de referencia…")
def cargar_referencia() -> pd.DataFrame | None:
    """Base para buscar valores reales en 'Desde cero': test_panel.csv si está, si no los ejemplos."""
    ruta = BASE / "test_panel.csv"
    if ruta.exists():
        try:
            return pd.read_csv(ruta, usecols=lambda c: c in FEATURES + ["PERIODO"])
        except Exception:
            pass
    return cargar_ejemplos()


def leer_json(ruta: Path):
    try:
        return json.loads(ruta.read_text(encoding="utf-8"))
    except Exception:
        return None


modelo, error_modelo = None, None
if RUTA_MODELO.exists():
    try:
        modelo = cargar_modelo(str(RUTA_MODELO))
    except Exception as e:  # versiones de sklearn/lightgbm distintas, etc.
        error_modelo = str(e)

ejemplos = cargar_ejemplos()
referencia = cargar_referencia()
decisiones = leer_json(RUTA_DECISIONES)
versiones = leer_json(RUTA_VERSIONES)

# Si el JSON congelado existe, sus umbrales mandan sobre los valores por defecto
if decisiones and isinstance(decisiones.get("criterios"), dict):
    mapa = {"Máx. F1": "Máximo F1 (alertas precisas)", "Máx. F2": "Máximo F2 (prioriza recall)",
            "Recall ≥ 0.80": "Recall ≥ 0.80", "Mínimo costo": "Mínimo costo (FN=100, FP=5)"}
    for k, v in decisiones["criterios"].items():
        if k in mapa:
            UMBRALES[mapa[k]] = float(v)


# ---------------------------------------------------------------------------
# Utilidades de predicción
# ---------------------------------------------------------------------------
def opciones(col: str) -> list[str]:
    if ejemplos is not None and col in ejemplos.columns:
        vals = ejemplos[col].dropna().astype(str).value_counts().index.tolist()
        if vals:
            return vals
    return {"ESTADO_REGISTRO": ["OCUPADO", "VACANTE"],
            "NIVEL": ["GOBIERNO REGIONAL", "GOBIERNO NACIONAL"]}.get(col, ["SIN_DATO"])


def mes_desde_sincos(s: float, c: float) -> int:
    ang = np.arctan2(s, c)
    m = int(round(ang / (2 * np.pi) * 12)) % 12
    return 12 if m == 0 else m


def preparar(df: pd.DataFrame) -> pd.DataFrame:
    """Completa MES_SIN/MES_COS si viene MES y ordena las 25 variables."""
    df = df.copy()
    if ("MES_SIN" not in df.columns or "MES_COS" not in df.columns) and "MES" in df.columns:
        df["MES_SIN"] = np.sin(2 * np.pi * df["MES"] / 12)
        df["MES_COS"] = np.cos(2 * np.pi * df["MES"] / 12)
    if ("MES_SIN" not in df.columns) and "PERIODO" in df.columns:
        mes = pd.to_datetime(df["PERIODO"].astype(str), format="%Y%m%d", errors="coerce").dt.month
        df["MES_SIN"] = np.sin(2 * np.pi * mes / 12)
        df["MES_COS"] = np.cos(2 * np.pi * mes / 12)
    faltan = [c for c in FEATURES if c not in df.columns]
    if faltan:
        raise ValueError("Faltan columnas: " + ", ".join(faltan))
    return df[FEATURES]


def predecir(df: pd.DataFrame) -> np.ndarray:
    return modelo.predict_proba(preparar(df))[:, 1]


def contribuciones(fila: pd.DataFrame) -> pd.Series | None:
    """Contribuciones SHAP nativas de LightGBM agregadas a las 25 variables originales."""
    try:
        prep = modelo.named_steps["prep"]
        lgbm = modelo.named_steps["modelo"]
        Xt = prep.transform(preparar(fila))
        if hasattr(Xt, "toarray"):
            Xt = Xt.toarray()
        contrib = lgbm.predict(Xt, pred_contrib=True)[0][:-1]
        nombres = list(prep.get_feature_names_out())
        agg = {}
        for nom, v in zip(nombres, contrib):
            limpio = re.sub(r"^f\d{4}_", "", nom.split("__", 1)[-1])
            origen = max((f for f in FEATURES if limpio.startswith(f)), key=len, default=limpio)
            agg[origen] = agg.get(origen, 0.0) + float(v)
        s = pd.Series(agg)
        if "MES_SIN" in s and "MES_COS" in s:
            s["MES"] = s.pop("MES_SIN") + s.pop("MES_COS")
        return s.reindex(s.abs().sort_values(ascending=False).index)
    except Exception:
        return None


def grafico_resultado(prob: float, umbral: float, mes_destino: str) -> go.Figure:
    """Barras verticales OCUPADO vs VACANTE; la ganadora en color pleno, la otra atenuada."""
    gana_vac = prob >= umbral
    vals = [(1 - prob) * 100, prob * 100]
    colores = [TEAL if not gana_vac else "rgba(20,108,103,.35)",
               ROJO if gana_vac else "rgba(192,80,77,.35)"]
    fig = go.Figure(go.Bar(
        x=["OCUPADO", "VACANTE"], y=vals, marker_color=colores, width=0.55,
        text=[f"<b>{v:.1f}%</b>" for v in vals], textposition="outside",
        textfont=dict(size=22, family="Fraunces"), cliponaxis=False,
        hovertemplate="%{x}: %{y:.1f}%<extra></extra>",
    ))
    fig.update_traces(marker_line_width=0)
    # Línea del umbral solo sobre la barra VACANTE
    fig.add_shape(type="line", x0=0.62, x1=1.38, y0=umbral * 100, y1=umbral * 100,
                  line=dict(color=NARANJA, width=3, dash="dash"))
    fig.add_annotation(x=1.38, y=umbral * 100, xanchor="left", showarrow=False,
                       text=f"umbral {umbral:.0%}", font=dict(color=NARANJA, size=13))
    fig.update_layout(
        title=dict(text=f"Pronóstico para {mes_destino}: <b>{'VACANTE' if gana_vac else 'OCUPADO'}</b>",
                   font=dict(size=18)),
        yaxis=dict(range=[0, 112], ticksuffix="%", title="Probabilidad"),
        xaxis=dict(tickfont=dict(size=15)), bargap=0.4,
    )
    return estilo_fig(fig, 400)


def tarjetas_resultado(prob: float, umbral: float, mes_destino: str, persist: str) -> str:
    """Dos tarjetas: OCUPADO y VACANTE. La que gana el pronóstico queda resaltada."""
    gana_vac = prob >= umbral
    def tarjeta(nombre, p, gana, clase):
        badge = '<span class="rc-badge">Pronóstico</span>' if gana else ''
        return (f'<div class="rc-card {clase} {"rc-win" if gana else "rc-lose"}">'
                f'<div class="rc-top"><span class="rc-name">{nombre}</span>{badge}</div>'
                f'<div class="rc-pct">{p * 100:.1f}<small>%</small></div>'
                f'<div class="rc-track"><div class="rc-fill" style="width:{p * 100:.1f}%"></div></div>'
                f'<div class="rc-sub">≈ {round(p * 100)} de cada 100 perfiles como este</div></div>')
    return (f'<div class="rc-head">Pronóstico para <b>{mes_destino}</b></div>'
            f'<div class="rc-grid">{tarjeta("OCUPADO", 1 - prob, not gana_vac, "rc-ocu")}'
            f'{tarjeta("VACANTE", prob, gana_vac, "rc-vac")}</div>'
            f'<div class="rc-foot">Se marca VACANTE solo si su probabilidad supera <b>{umbral:.0%}</b> '
            f'(umbral elegido) · Hoy el perfil está <b>{persist}</b></div>')


def barra_probabilidad(prob: float, umbral: float, mes_destino: str) -> go.Figure:
    """Barra dividida: OCUPADO (izq.) vs VACANTE (der.), con la línea del umbral."""
    p_ocu, p_vac = 1 - prob, prob
    fig = go.Figure()
    fig.add_bar(y=[""], x=[p_ocu * 100], orientation="h", marker_color=TEAL, name="OCUPADO",
                text=[f"OCUPADO  {p_ocu:.1%}"], textposition="inside", insidetextanchor="middle",
                textfont=dict(size=17, color="white"), hovertemplate="OCUPADO: %{x:.1f}%<extra></extra>")
    fig.add_bar(y=[""], x=[p_vac * 100], orientation="h", marker_color=ROJO, name="VACANTE",
                text=[f"VACANTE  {p_vac:.1%}"], textposition="inside", insidetextanchor="middle",
                textfont=dict(size=17, color="white"), hovertemplate="VACANTE: %{x:.1f}%<extra></extra>")
    fig.update_layout(barmode="stack", showlegend=False,
                      title=f"¿Cómo estará el perfil en {mes_destino}?",
                      xaxis=dict(range=[0, 100], visible=False),
                      yaxis=dict(showticklabels=False))
    return estilo_fig(fig, 210)


def gauge(prob: float, umbral: float) -> go.Figure:
    color = ROJO if prob >= umbral else TEAL
    fig = go.Figure(go.Indicator(
        mode="gauge+number", value=prob * 100,
        number={"suffix": "%", "font": {"size": 46, "family": "Fraunces"}},
        title={"text": "Probabilidad de VACANTE en t+1", "font": {"size": 15}},
        gauge={
            "axis": {"range": [0, 100], "ticksuffix": "%"},
            "bar": {"color": color, "thickness": 0.28},
            "steps": [{"range": [0, umbral * 100], "color": "rgba(20,108,103,.12)"},
                      {"range": [umbral * 100, 100], "color": "rgba(192,80,77,.12)"}],
            "threshold": {"line": {"color": NARANJA, "width": 4}, "value": umbral * 100},
        },
    ))
    return estilo_fig(fig, 300)


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown("### 🏛️ AIRHSP · ML")
    st.caption("Prototipo académico · Aprendizaje automático aplicado")

    st.markdown("#### Modelo")
    if modelo is not None:
        st.success("LightGBM ajustado cargado", icon="✅")
    else:
        if error_modelo:
            st.error("No se pudo abrir el modelo. Revisa versiones de scikit-learn / LightGBM.", icon="⚠️")
            with st.expander("Detalle"):
                st.code(error_modelo)
        else:
            st.warning("No se encontró `modelo_campeon_pipeline.joblib`.", icon="📦")
        subido = st.file_uploader("Sube el modelo (.joblib)", type=["joblib", "pkl"])
        if subido is not None:
            try:
                modelo = joblib.load(io.BytesIO(subido.getvalue()))
                st.success("Modelo cargado desde archivo", icon="✅")
            except Exception as e:
                st.error(f"Error al cargar: {e}")

    st.markdown("#### Punto de operación")
    criterio = st.radio("Criterio del umbral", list(UMBRALES) + ["Personalizado"], index=0,
                        help="Umbrales fijados en validación antes de abrir el test.")
    if criterio == "Personalizado":
        umbral = st.slider("Umbral", 0.0, 1.0, 0.30, 0.005)
    else:
        umbral = UMBRALES[criterio]
    st.metric("Umbral activo", f"{umbral:.4f}")

    if versiones:
        with st.expander("Versiones de entrenamiento"):
            st.json(versiones)
    st.markdown("---")
    st.caption("Fuente: AIRHSP – MEF · ene-2024 a ago-2026. "
               "La unidad es un perfil agregado, no una persona ni una plaza individual.")


# ---------------------------------------------------------------------------
# Hero
# ---------------------------------------------------------------------------
st.markdown("""
<div class="hero">
  <div class="kicker">UTEC · Ingeniería Industrial · Proyecto de Aprendizaje Automático</div>
  <h1>¿Estará ocupado o vacante el próximo mes?</h1>
  <p>Pronóstico del estado predominante futuro de perfiles institucional-laborales del sector público
  peruano, a partir de 3.65 millones de registros del AIRHSP y un modelo LightGBM ajustado.</p>
</div>
""", unsafe_allow_html=True)

tab_inicio, tab_pred, tab_lote, tab_res, tab_seg, tab_met = st.tabs([
    "🏠 El proyecto", "🔮 Pronosticar", "📂 Pronóstico por lote",
    "📊 Resultados", "🧩 Segmentación", "🧭 Metodología",
])

# ===========================================================================
# 1. EL PROYECTO
# ===========================================================================
with tab_inicio:
    c = st.columns(4)
    c[0].markdown(card("Registros AIRHSP", "3.65 M", "ene-2024 → ago-2026 · 32 meses"), unsafe_allow_html=True)
    c[1].markdown(card("Perfiles únicos", "91,885", "grupos de plazas con 10 rasgos comunes"), unsafe_allow_html=True)
    c[2].markdown(card("Perfil-mes analizados", "1.68 M", "con el mes siguiente conocido"), unsafe_allow_html=True)
    c[3].markdown(card("Vacantes el mes siguiente", "≈ 3 %", "2.83 % en train · 1.58 % en test"), unsafe_allow_html=True)

    st.markdown('<div class="sec-title">¿Qué predice el modelo?</div>'
                '<div class="sec-sub">Un ejemplo: a partir de cómo está un grupo de plazas este mes, '
                'se pronostica cómo estará el mes siguiente.</div>', unsafe_allow_html=True)
    st.markdown("""
<div class="ex">
  <div class="box"><div class="k">Perfil (grupo de plazas)</div>
    <div class="v">Enfermera · Hospital de Chota · D. Leg. 276</div>
    <div style="font-size:.82rem;opacity:.7;margin-top:.3rem">4 plazas en el grupo</div></div>
  <div class="arrow">→</div>
  <div class="box ocu"><div class="k" style="opacity:.85">Junio (mes t) · dato conocido</div>
    <div class="v">OCUPADO</div><div style="font-size:.82rem;opacity:.85;margin-top:.3rem">3 ocupadas · 1 vacante</div></div>
  <div class="arrow">→</div>
  <div class="box q"><div class="k" style="opacity:.85">Julio (mes t+1) · pronóstico</div>
    <div class="v">¿OCUPADO o VACANTE?</div><div style="font-size:.82rem;opacity:.85;margin-top:.3rem">lo responde el modelo</div></div>
</div>""", unsafe_allow_html=True)
    st.caption("Ejemplo ilustrativo. Como el AIRHSP no tiene un código único por plaza, se agrupan las plazas con las "
               "mismas 10 características en un perfil; su estado es el que concentra más plazas.")

    izq, der = st.columns([1, 1.15])
    with izq:
        st.markdown('<div class="sec-title">¿Por qué importa?</div>', unsafe_allow_html=True)
        st.markdown("""
El **AIRHSP** del MEF registra cada mes las plazas del sector público y si están **ocupadas** o **vacantes**.
Saber con anticipación qué grupos de plazas tenderán a quedar vacantes ayuda a:

- 🔎 **Priorizar revisiones** de los casos con más riesgo.
- 📋 **Planificar convocatorias** antes de que la vacante ocurra.
- 💰 **Supervisar el presupuesto** de personal.
""")
    with der:
        st.markdown('<div class="sec-title">De los datos al modelo</div>', unsafe_allow_html=True)
        fig = go.Figure(go.Funnel(
            y=["Registros originales", "Perfiles por mes", "Con mes siguiente conocido"],
            x=[3_652_603, 1_776_598, 1_678_575],
            texttemplate="%{x:,.0f}", textfont=dict(size=15, color="white"),
            marker=dict(color=[AZUL, "#2f7f86", TEAL]), connector=dict(line=dict(color="rgba(128,128,128,.3)")),
        ))
        fig.update_layout(margin=dict(l=10, r=10, t=10, b=10))
        st.plotly_chart(estilo_fig(fig, 250), width="stretch")
        st.caption("Las filas se agrupan por perfil y mes; luego se quedan solo los casos cuyo mes siguiente existe en los datos.")

    st.markdown('<div class="sec-title">Cómo se construyó</div>', unsafe_allow_html=True)
    pasos = [
        ("🔍", "Revisamos los datos", "Limpieza, valores faltantes y variables que podían “delatar” la respuesta.", "3.65 M filas"),
        ("🧩", "Armamos los perfiles", "Grupos de plazas con las mismas 10 características, mes a mes.", "91,885 perfiles"),
        ("📅", "Separamos por fechas", "Se entrena con el pasado y se prueba con meses futuros.", "feb–jul 2026 = test"),
        ("🤖", "Probamos modelos", "6 algoritmos y 2 reglas de referencia; se eligió el mejor.", "LightGBM"),
        ("🧭", "Agrupamos perfiles", "Complemento: segmentos de perfiles parecidos según sus montos.", "8 segmentos"),
    ]
    st.markdown('<div class="grid g5">' + "".join(
        f'<div class="icard"><div class="ico">{i}</div><div class="t">{t}</div><div class="d">{d}</div>'
        f'<div class="n" style="font-size:1.05rem">{n}</div></div>' for i, t, d, n in pasos) + "</div>",
        unsafe_allow_html=True)

    st.markdown('<div class="sec-title">Mensaje principal</div>', unsafe_allow_html=True)
    msgs = [
        ("📈", "Ordena mejor el riesgo", "0.783", "Average Precision, frente a 0.553 de la regla “todo sigue igual”."),
        ("🎯", "Alertas confiables", "8 de cada 10", "alertas VACANTE son correctas (precisión 80.7 %)."),
        ("🔦", "Detecta la mayoría", "2 de cada 3", "vacantes reales del mes siguiente (recall 65.8 %)."),
    ]
    st.markdown('<div class="grid g3">' + "".join(
        f'<div class="icard"><div class="ico">{i}</div><div class="t">{t}</div><div class="n">{n}</div>'
        f'<div class="d">{d}</div></div>' for i, t, n, d in msgs) + "</div>", unsafe_allow_html=True)
    st.info("**En resumen:** el modelo es una buena herramienta para **priorizar** qué perfiles revisar primero. "
            "No reemplaza a la regla simple en todos los criterios: con los costos supuestos, la persistencia "
            "sigue siendo competitiva (ver *Resultados*).", icon="💡")

    st.markdown('<div class="sec-title">¿Por dónde seguir?</div>', unsafe_allow_html=True)
    nav = [("🔮", "Pronosticar", "Prueba el modelo con un perfil real o arma uno."),
           ("📂", "Pronóstico por lote", "Sube muchos perfiles y obtén un ranking de riesgo."),
           ("📊", "Resultados", "Compara modelos, simula costos y explora umbrales."),
           ("🧭", "Metodología", "Cómo se separaron los datos y qué variables usa el modelo.")]
    st.markdown('<div class="grid g4">' + "".join(
        f'<div class="icard accent"><div class="ico">{i}</div><div class="t">{t}</div><div class="d">{d}</div></div>'
        for i, t, d in nav) + "</div>", unsafe_allow_html=True)
    st.caption("Usa las pestañas de arriba para navegar.")

# ===========================================================================
# 2. PRONOSTICAR
# ===========================================================================
with tab_pred:
    st.subheader("Pronóstico para un perfil")
    st.caption("Parte de un perfil real del periodo de test y modifica lo que quieras, o arma uno desde cero.")

    base_fila = None
    id_base = "cero"
    if ejemplos is not None:
        modo = st.segmented_control("Punto de partida", ["Perfil de ejemplo", "Desde cero"],
                                    default="Perfil de ejemplo")
        if modo == "Perfil de ejemplo":
            ej = ejemplos.reset_index(drop=True)
            filtro = st.selectbox("Filtrar ejemplos por estado actual", ["Todos", "OCUPADO", "VACANTE"])
            if filtro != "Todos" and "ESTADO_REGISTRO" in ej.columns:
                ej = ej[ej["ESTADO_REGISTRO"] == filtro].reset_index(drop=True)

            def etiqueta(i):
                r = ej.iloc[i]
                return (f"#{i + 1} · {str(r.get('SECTOR', ''))[:28]} · "
                        f"{str(r.get('DESC_CARGO_ESTRUCUTURAL', ''))[:38]} · {r.get('ESTADO_REGISTRO', '')}")

            if len(ej):
                idx = st.selectbox("Perfil", range(len(ej)), format_func=etiqueta)
                base_fila = ej.iloc[idx]
                id_base = f"ej_{filtro}_{idx}"
    else:
        st.info("Coloca `ejemplos_prototipo.csv` junto a app.py para tener perfiles reales y listas desplegables completas.", icon="💡")

    CAT_ORDEN = ["NIVEL", "SECTOR", "PLIEGO", "UNIDAD_EJECUTORA", "TIPO_ESTABLECIMIENTO",
                 "DESC_TIPO_REGISTRO", "DESC_SUB_TIPO_REGISTRO", "DESC_REGIMEN_LABORAL",
                 "DESC_GRUPO_OCASIONAL", "DESC_CARGO_ESTRUCUTURAL", "DESC_CONDICION_LABORAL"]
    desde_cero = base_fila is None

    with st.expander("📘 ¿Cómo usar este formulario?", expanded=False):
        st.markdown("""
**Perfil de ejemplo:** elige un perfil de la lista; todos los campos se llenan con sus **datos reales**. Solo pulsa **Pronosticar**.

**Desde cero:** elige el puesto en la sección 1, de arriba hacia abajo. Cada lista solo muestra opciones que **existen en los datos**.
Con eso, la app busca ese perfil y completa sola la **cantidad** y los **montos** reales.

- **1 · ¿Qué puesto es?** Entidad, régimen, cargo… (identifican al grupo).
- **2 · ¿Cómo está hoy?** Estado actual y mes de los datos.
- **3 · Tamaño y montos:** cuántas plazas agrupa y cuánto cuesta cada una. Se llenan solos.

Si cambias un dato a mano, el resultado pasa a ser una **simulación** ("¿qué pasaría si…?") y la app te lo avisa.
Pasa el mouse sobre el ícono ❓ de cada campo para ver qué significa.
""")

    # ---------- 1 · ¿Qué puesto es? ----------
    st.markdown("##### 1 · ¿Qué puesto es?")
    cat_vals = {}
    # Filas del formulario: los campos de nombres largos van a lo ancho completo
    FILAS = [["NIVEL", "SECTOR"], ["PLIEGO"], ["UNIDAD_EJECUTORA"],
             ["TIPO_ESTABLECIMIENTO", "DESC_TIPO_REGISTRO", "DESC_SUB_TIPO_REGISTRO"],
             ["DESC_REGIMEN_LABORAL", "DESC_GRUPO_OCASIONAL"],
             ["DESC_CARGO_ESTRUCUTURAL", "DESC_CONDICION_LABORAL"]]
    contenedor = {}
    for fila_cols in FILAS:
        for c_, slot in zip(fila_cols, st.columns(len(fila_cols))):
            contenedor[c_] = slot
    sub = referencia if (desde_cero and referencia is not None) else None
    for i, col in enumerate(CAT_ORDEN):
        if desde_cero and sub is not None and col in sub.columns:
            # Lista en cascada: solo combinaciones que existen en los datos reales
            ops = sub[col].dropna().astype(str).value_counts().index.tolist() or opciones(col)
            actual = ops[0]
            k = col + "_cero_" + str(abs(hash(tuple(cat_vals.values()))))
        else:
            ops = opciones(col)
            actual = str(base_fila[col]) if (base_fila is not None and pd.notna(base_fila.get(col))) else ops[0]
            if actual not in ops:
                ops = [actual] + ops
            k = col + f"_{id_base}"
        cat_vals[col] = contenedor[col].selectbox(ETIQUETAS[col], ops, index=ops.index(actual), key=k,
                                              help=AYUDAS.get(col))
        if desde_cero and sub is not None and col in sub.columns:
            filtrado = sub[sub[col].astype(str) == cat_vals[col]]
            sub = filtrado if len(filtrado) else sub

    # Valores de partida para la sección 2 y 3
    if desde_cero:
        if sub is not None and len(sub):
            fuente = sub
            ref_vals = {c: float(fuente[c].median()) for c in MONETARIAS + ["CANTIDAD_TOTAL"] if c in fuente}
            ref_vals["ESTADO_REGISTRO"] = fuente["ESTADO_REGISTRO"].mode().iat[0] if "ESTADO_REGISTRO" in fuente else "OCUPADO"
            if "MES_SIN" in fuente:
                ult = fuente.iloc[-1]
                ref_vals["MES"] = mes_desde_sincos(float(ult["MES_SIN"]), float(ult["MES_COS"]))
            n = len(fuente)
            if n == 1:
                st.success("✅ Encontramos este perfil en los datos reales: cantidad y montos son **sus valores reales**.")
            else:
                st.info(f"📊 Encontramos **{n:,} registros reales** de este perfil (en distintos meses). "
                        "Cantidad y montos se completaron con la **mediana** de esos registros.")
        else:
            ref_vals = {}
            st.warning("No hay datos de referencia: completa cantidad y montos manualmente (será una simulación).")
        clave = "_cero_" + str(abs(hash(tuple(cat_vals.values()))))
    else:
        ref_vals = {c: base_fila[c] for c in base_fila.index}
        ref_vals["MES"] = mes_desde_sincos(float(base_fila.get("MES_SIN", 0.5)), float(base_fila.get("MES_COS", 0.866)))
        clave = f"_{id_base}"

    def val(col, defecto):
        v = ref_vals.get(col, defecto)
        return defecto if (v is None or (isinstance(v, float) and np.isnan(v))) else v

    # ---------- 2 · ¿Cómo está hoy? ----------
    st.markdown("##### 2 · ¿Cómo está hoy?")
    a, b = st.columns(2)
    estado_ops = ["OCUPADO", "VACANTE"]
    est_def = str(val("ESTADO_REGISTRO", "OCUPADO"))
    estado = a.radio("Estado actual del perfil (t)", estado_ops, horizontal=True, help=AYUDAS["ESTADO_REGISTRO"],
                     index=estado_ops.index(est_def) if est_def in estado_ops else 0, key="est" + clave)
    mes = b.selectbox("Mes del periodo t", range(1, 13), index=int(val("MES", 6)) - 1,
                      format_func=lambda m: MESES[m - 1], key="mes" + clave, help=AYUDAS["MES"])

    # ---------- 3 · Tamaño y montos ----------
    st.markdown("##### 3 · Tamaño y montos por unidad (S/)")
    st.caption("Se completan solos con los datos reales del perfil. Normalmente no necesitas tocarlos.")
    cant_def = float(val("CANTIDAD_TOTAL", 1.0))
    cantidad = st.number_input("Cantidad total del perfil (plazas que agrupa)", min_value=1.0,
                               value=max(cant_def, 1.0), step=1.0, key="cant" + clave,
                               help=AYUDAS["CANTIDAD_TOTAL"])
    mon_vals = {}
    cols = st.columns(5)
    for i, col in enumerate(MONETARIAS):
        mon_vals[col] = cols[i % 5].number_input(ETIQUETAS[col], min_value=0.0,
                                                 value=float(val(col, 0.0)), step=50.0,
                                                 format="%.2f", key=col + clave, help=AYUDA_MONTO)

    # ¿El usuario cambió algo respecto a los datos reales?
    cambios = []
    if ref_vals:
        if abs(cantidad - max(cant_def, 1.0)) > 1e-6:
            cambios.append("cantidad")
        cambios += [ETIQUETAS[c].lower() for c in MONETARIAS if abs(mon_vals[c] - float(val(c, 0.0))) > 1e-6]
        if estado != est_def:
            cambios.append("estado actual")
        if mes != int(val("MES", 6)):
            cambios.append("mes")
        if not desde_cero:
            cambios += [ETIQUETAS[c].lower() for c in CAT_ORDEN if str(cat_vals[c]) != str(base_fila.get(c))]
    if cambios:
        st.warning("🧪 **Modo simulación:** cambiaste " + ", ".join(cambios) +
                   ". El resultado responde a *¿qué pasaría si…?*, no describe un perfil real.")

    if st.button("🔮  Pronosticar estado en t+1", type="primary", width="stretch"):
        st.session_state["mostrar_pred"] = True
    enviar = st.session_state.get("mostrar_pred", False)

    if enviar:
        if modelo is None:
            st.error("Primero carga el modelo (`modelo_campeon_pipeline.joblib`) desde la barra lateral.")
        else:
            fila = pd.DataFrame([{**mon_vals, "CANTIDAD_TOTAL": cantidad,
                                  "MES_SIN": np.sin(2 * np.pi * mes / 12),
                                  "MES_COS": np.cos(2 * np.pi * mes / 12),
                                  **cat_vals, "ESTADO_REGISTRO": estado}])
            prob = float(predecir(fila)[0])
            es_vac = prob >= umbral
            persist = estado  # regla de persistencia: t+1 = t

            txt = "VACANTE" if es_vac else "OCUPADO"
            st.plotly_chart(grafico_resultado(prob, umbral, MESES[mes % 12]), width="stretch")
            st.caption(f"De cada 100 perfiles como este, el modelo espera unos **{round((1 - prob) * 100)} OCUPADOS** y "
                       f"unos **{round(prob * 100)} VACANTES**. Se marca VACANTE solo si la barra roja supera la línea "
                       f"naranja (**{umbral:.0%}**). Hoy el perfil está **{persist}**.")
            if txt != persist:
                st.warning(f"Hoy el perfil está **{persist}** y el modelo pronostica **{txt}**: anticipa un "
                           "**cambio de estado**. Es el tipo de caso que conviene revisar primero.", icon="🔁")
            seg_id = asignar_segmento(fila)
            med_ns = leer_csv_ns("medianas_clusters.csv", index_col=0)
            if seg_id is not None:
                nom = nombrar_segmentos(med_ns).get(int(seg_id[0]), f"S{int(seg_id[0])}") if med_ns is not None else f"S{int(seg_id[0])}"
                st.info(f"🧭 Según sus montos y tamaño, este perfil pertenece al segmento **{nom}** "
                        "(ver pestaña *Segmentación*).")

            contrib = contribuciones(fila)
            if contrib is not None:
                top = contrib.head(10).iloc[::-1]
                nombres = [ETIQUETAS.get(k, k) for k in top.index]
                fig = go.Figure(go.Bar(
                    x=top.values, y=nombres, orientation="h",
                    marker_color=[ROJO if x > 0 else TEAL for x in top.values],
                    hovertemplate="%{y}: %{x:+.3f}<extra></extra>",
                ))
                fig.update_layout(title="¿Qué empujó este pronóstico? (contribución en log-odds)",
                                  xaxis_title="← hacia OCUPADO  ·  hacia VACANTE →")
                st.plotly_chart(estilo_fig(fig, 400), width="stretch")
                st.caption("Contribuciones SHAP nativas de LightGBM agregadas a las variables originales. "
                           "Son explicaciones del modelo, no relaciones causales.")

            # Sensibilidad al umbral
            with st.expander("¿Cómo cambia la decisión según el criterio de umbral?"):
                tabla = pd.DataFrame([(k, t, "VACANTE" if prob >= t else "OCUPADO") for k, t in UMBRALES.items()],
                                     columns=["Criterio", "Umbral", "Decisión"])
                st.dataframe(tabla, hide_index=True, width="stretch")

# ===========================================================================
# 3. LOTE
# ===========================================================================
with tab_lote:
    st.subheader("Pronóstico masivo")
    st.markdown("Sube un CSV con las **25 variables del panel** (como `test_panel.csv` o `ejemplos_prototipo.csv`). "
                "Si trae `MES` o `PERIODO` en lugar de `MES_SIN/MES_COS`, se calculan solos. "
                "Si incluye `TARGET`, se evalúa el desempeño.")
    with st.expander("Ver columnas requeridas"):
        st.code(", ".join(FEATURES))

    archivo = st.file_uploader("CSV de perfiles", type=["csv"], key="lote")
    usar_ej = ejemplos is not None and st.checkbox("Usar los perfiles de ejemplo incluidos", value=archivo is None)
    datos = pd.read_csv(archivo) if archivo is not None else (ejemplos if usar_ej else None)

    if datos is not None:
        if modelo is None:
            st.error("Carga el modelo para pronosticar.")
        else:
            try:
                res = datos.copy()
                res["PROB_VACANTE"] = predecir(datos)
                res["PRONOSTICO_T1"] = np.where(res["PROB_VACANTE"] >= umbral, "VACANTE", "OCUPADO")
                if "ESTADO_REGISTRO" in res:
                    res["CAMBIO_ESPERADO"] = res["PRONOSTICO_T1"] != res["ESTADO_REGISTRO"].astype(str)
                res = res.sort_values("PROB_VACANTE", ascending=False)

                k = st.columns(4)
                k[0].markdown(card("Perfiles", f"{len(res):,}"), unsafe_allow_html=True)
                k[1].markdown(card("Alertas VACANTE", f"{(res.PRONOSTICO_T1 == 'VACANTE').sum():,}",
                                   f"{(res.PRONOSTICO_T1 == 'VACANTE').mean():.1%} del lote"), unsafe_allow_html=True)
                k[2].markdown(card("Prob. media", f"{res.PROB_VACANTE.mean():.1%}"), unsafe_allow_html=True)
                if "CAMBIO_ESPERADO" in res:
                    k[3].markdown(card("Cambios de estado", f"{res.CAMBIO_ESPERADO.sum():,}",
                                       "pronóstico ≠ estado actual"), unsafe_allow_html=True)

                g1, g2 = st.columns(2)
                fig = px.histogram(res, x="PROB_VACANTE", nbins=40, color="PRONOSTICO_T1",
                                   color_discrete_map={"VACANTE": ROJO, "OCUPADO": TEAL}, log_y=True,
                                   title="Distribución de probabilidades (eje Y log)")
                fig.add_vline(x=umbral, line_dash="dash", line_color=NARANJA, annotation_text="umbral")
                g1.plotly_chart(estilo_fig(fig), width="stretch")
                if "SECTOR" in res:
                    por_sector = (res.groupby("SECTOR")["PROB_VACANTE"].mean()
                                  .sort_values(ascending=False).head(12).iloc[::-1])
                    fig = go.Figure(go.Bar(x=por_sector.values, y=por_sector.index, orientation="h",
                                           marker_color=AZUL))
                    fig.update_layout(title="Riesgo medio por sector (top 12)", xaxis_tickformat=".0%")
                    g2.plotly_chart(estilo_fig(fig), width="stretch")

                if "TARGET" in res:
                    from sklearn.metrics import (average_precision_score, f1_score,
                                                 precision_score, recall_score)
                    y = res["TARGET"].astype(int)
                    p = (res.PRONOSTICO_T1 == "VACANTE").astype(int)
                    st.markdown("##### Desempeño en este lote (tiene `TARGET`)")
                    e = st.columns(4)
                    if y.nunique() > 1:
                        e[0].metric("AP", f"{average_precision_score(y, res.PROB_VACANTE):.3f}")
                    e[1].metric("Precisión", f"{precision_score(y, p, zero_division=0):.3f}")
                    e[2].metric("Recall", f"{recall_score(y, p, zero_division=0):.3f}")
                    e[3].metric("F1", f"{f1_score(y, p, zero_division=0):.3f}")
                    tn_ = int(((y == 0) & (p == 0)).sum()); fp_ = int(((y == 0) & (p == 1)).sum())
                    fn_ = int(((y == 1) & (p == 0)).sum()); tp_ = int(((y == 1) & (p == 1)).sum())
                    st.markdown(f"""
<div class="cm" style="margin-top:.6rem">
  <div></div><div class="h">El modelo dijo <b>OCUPADO</b></div><div class="h">El modelo dijo <b>VACANTE</b></div>
  <div class="r">En realidad<br><b>OCUPADO</b></div>
  <div class="c ok"><div class="lb">✅ Ocupados bien predichos</div><div class="num">{tn_:,}</div></div>
  <div class="c warn"><div class="lb">⚠️ Falsas alarmas</div><div class="num">{fp_:,}</div></div>
  <div class="r">En realidad<br><b>VACANTE</b></div>
  <div class="c bad"><div class="lb">❌ Vacantes no detectadas</div><div class="num">{fn_:,}</div></div>
  <div class="c ok"><div class="lb">✅ Vacantes detectadas</div><div class="num">{tp_:,}</div></div>
</div>""", unsafe_allow_html=True)
                    st.caption(f"Calculada en vivo con el umbral activo ({umbral:.4f}). Cambia el criterio en la barra "
                               "lateral y mira cómo se mueven las falsas alarmas y las vacantes no detectadas.")

                st.markdown("##### Ranking de perfiles por riesgo")
                mostrar = [c for c in ["PROB_VACANTE", "PRONOSTICO_T1", "ESTADO_REGISTRO", "CAMBIO_ESPERADO",
                                       "SECTOR", "PLIEGO", "DESC_CARGO_ESTRUCUTURAL", "DESC_REGIMEN_LABORAL",
                                       "CANTIDAD_TOTAL", "TARGET"] if c in res.columns]
                st.dataframe(res[mostrar], hide_index=True, width="stretch", height=380,
                             column_config={"PROB_VACANTE": st.column_config.ProgressColumn(
                                 "Prob. VACANTE", format="%.3f", min_value=0, max_value=1)})
                st.download_button("⬇️ Descargar resultados (CSV)", res.to_csv(index=False).encode("utf-8-sig"),
                                   "pronosticos_airhsp.csv", "text/csv", width="stretch")
            except ValueError as e:
                st.error(str(e))

# ===========================================================================
# 4. RESULTADOS
# ===========================================================================
with tab_res:
    k = st.columns(3)
    k[0].markdown(card("Mejor modelo", "LightGBM", "versión ajustada · elegido por AP de validación"), unsafe_allow_html=True)
    k[1].markdown(card("Ordena el riesgo", "0.783", "AP en test · regla simple: 0.553"), unsafe_allow_html=True)
    k[2].markdown(card("Alertas correctas", "8 de 10", "precisión 80.7 % · detecta 2 de cada 3 vacantes"), unsafe_allow_html=True)

    def color_modelo(m):
        return TEAL if m == "LightGBM (ajustado)" else NARANJA if m == "Persistencia" else "#8fa3b3"

    # ---------------- 1 · Ranking de modelos ----------------
    st.markdown('<div class="sec-title">1 · ¿Qué modelo fue mejor?</div>'
                '<div class="sec-sub">El modelo se eligió con <b>validación</b>; el <b>test</b> se abrió al final, '
                'como prueba. Cambia la métrica para ver el ranking desde otro ángulo.</div>', unsafe_allow_html=True)
    c1, c2 = st.columns([1.4, 1])
    metrica = c1.segmented_control("Métrica", list(EXPLICA_METRICA), default="AP", key="res_met") or "AP"
    conjunto = c2.segmented_control("Datos", ["Validación", "Test", "Ambos"], default="Ambos", key="res_conj") or "Ambos"
    st.caption("ℹ️ " + EXPLICA_METRICA[metrica])

    v = VAL.set_index("Modelo")[metrica]
    t = TEST_M.set_index("Modelo")[metrica]
    base = t if conjunto in ("Test", "Ambos") else v
    base = base.drop(index="Dummy")  # referencia trivial: se omite para no aplastar la escala
    orden = base.sort_values().index.tolist()
    vals_vis = pd.concat([v.drop(index="Dummy"), t.drop(index="Dummy")]) if conjunto == "Ambos" else base
    x_min = max(0.0, float(vals_vis.min()) - 0.06)
    fig = go.Figure()
    if conjunto == "Ambos":
        for m in orden:
            fig.add_shape(type="line", x0=v[m], x1=t[m], y0=m, y1=m, line=dict(color="rgba(150,150,150,.45)", width=3))
        fig.add_scatter(x=[v[m] for m in orden], y=orden, mode="markers", name="Validación",
                        marker=dict(size=12, color="rgba(0,0,0,0)", line=dict(width=2.5, color=[color_modelo(m) for m in orden])),
                        hovertemplate="%{y} · validación: %{x:.3f}<extra></extra>")
        fig.add_scatter(x=[t[m] for m in orden], y=orden, mode="markers+text", name="Test",
                        marker=dict(size=14, color=[color_modelo(m) for m in orden]),
                        text=[f"  {t[m]:.3f}" for m in orden], textposition="middle right",
                        hovertemplate="%{y} · test: %{x:.3f}<extra></extra>")
        fig.update_layout(legend=dict(orientation="h", y=1.08, x=0))
        st.caption("○ validación  ·  ● test — si los dos puntos están cerca, el modelo mantuvo su rendimiento.")
    else:
        fig.add_scatter(x=[base[m] for m in orden], y=orden, mode="markers+text",
                        marker=dict(size=15, color=[color_modelo(m) for m in orden]),
                        text=[f"  {base[m]:.3f}" for m in orden], textposition="middle right",
                        hovertemplate="%{y}: %{x:.3f}<extra></extra>")
        for m in orden:
            fig.add_shape(type="line", x0=x_min, x1=base[m], y0=m, y1=m, line=dict(color="rgba(150,150,150,.3)", width=2))
    fig.update_layout(xaxis=dict(range=[x_min, min(1.0, float(vals_vis.max()) + 0.06)], title=metrica),
                      showlegend=(conjunto == "Ambos"))
    st.plotly_chart(estilo_fig(fig, 470), width="stretch")
    st.caption("🟢 LightGBM ajustado (elegido) · 🟠 Persistencia: regla simple “el próximo mes estará igual que hoy”. "
               "El eje está ampliado para ver las diferencias; el Dummy (predice siempre lo mismo) se omite.")

    # ---------------- Comparador ----------------
    st.markdown("##### Compara dos modelos")
    modelos = TEST_M.Modelo.tolist()
    a1, a2, a3 = st.columns([1, 1, .8])
    m1 = a1.selectbox("Modelo A", modelos, index=modelos.index("LightGBM (ajustado)"), key="cmp_a")
    m2 = a2.selectbox("Modelo B", modelos, index=modelos.index("Persistencia"), key="cmp_b")
    datos_cmp = a3.segmented_control("Datos", ["Validación", "Test"], default="Test", key="cmp_d") or "Test"
    tabla = (TEST_M if datos_cmp == "Test" else VAL).set_index("Modelo")
    mets = ["AP", "Precisión", "Recall", "F1", "MCC"]
    fig = go.Figure()
    for m, col in [(m1, TEAL), (m2, NARANJA if m2 != m1 else AZUL)]:
        fig.add_bar(name=m, x=mets, y=[tabla.loc[m, x] for x in mets], marker_color=col,
                    text=[f"{tabla.loc[m, x]:.3f}" for x in mets], textposition="outside", cliponaxis=False)
    fig.update_layout(barmode="group", yaxis=dict(range=[0, 1.1]), legend=dict(orientation="h", y=1.12, x=0))
    st.plotly_chart(estilo_fig(fig, 340), width="stretch")
    gana = sum(tabla.loc[m1, x] > tabla.loc[m2, x] for x in mets)
    if m1 != m2:
        st.caption(f"**{m1}** supera a **{m2}** en {gana} de {len(mets)} métricas ({datos_cmp.lower()}).")

    # ---------------- 2 · Simulador de costos ----------------
    st.markdown('<div class="sec-title">2 · ¿Cuánto cuestan los errores?</div>'
                '<div class="sec-sub">Hay dos errores posibles: <b>perder una vacante</b> (no avisar y que sí ocurra) y '
                'dar una <b>falsa alarma</b> (avisar y que no ocurra). El proyecto supuso 100 y 5; mueve los valores '
                'para ver qué alternativa saldría más barata con otros supuestos.</div>', unsafe_allow_html=True)
    s1, s2 = st.columns(2)
    c_fn = s1.slider("💸 Costo de perder una vacante (FN)", 1, 300, 100, 5)
    c_fp = s2.slider("🔔 Costo de una falsa alarma (FP)", 1, 100, 5, 1)
    por_defecto = ["Siempre OCUPADO", "Persistencia", "LightGBM (ajustado)", "XGBoost (ajustado)",
                   "CatBoost (ajustado)", "Árbol de decisión"]
    elegidos = st.multiselect("Alternativas a comparar", list(ERRORES_TEST), default=por_defecto)
    if elegidos:
        cost = pd.DataFrame([(m, *ERRORES_TEST[m]) for m in elegidos], columns=["Alternativa", "FN", "FP"])
        cost["Costo"] = cost.FN * c_fn + cost.FP * c_fp
        cost = cost.sort_values("Costo", ascending=False)
        mejor = cost.iloc[-1].Alternativa
        colores = [TEAL if m == mejor else "#5d6f7e" for m in cost.Alternativa]
        fig = go.Figure(go.Bar(
            x=cost.Costo, y=cost.Alternativa, orientation="h", marker_color=colores,
            text=[f"{c:,.0f}" for c in cost.Costo], textposition="outside", cliponaxis=False,
            customdata=cost[["FN", "FP"]].values,
            hovertemplate="%{y}<br>Vacantes perdidas: %{customdata[0]:,}<br>Falsas alarmas: %{customdata[1]:,}"
                          "<br>Costo: %{x:,.0f}<extra></extra>"))
        fig.update_layout(xaxis_title="Costo total en test (menor es mejor)", margin=dict(r=70))
        st.plotly_chart(estilo_fig(fig, 60 + 48 * len(cost)), width="stretch")
        st.success(f"Con FN = {c_fn} y FP = {c_fp}, la alternativa más barata es **{mejor}**.", icon="🏆")
        st.caption("Cada modelo usa su umbral fijado en validación (máx. F1). Los conteos de Persistencia y LightGBM "
                   "ajustado son exactos; los demás se reconstruyen desde su precisión y recall. La variante de LightGBM "
                   "con umbral de mínimo costo (97,820 con 100/5) no aparece porque sus conteos no se guardaron.")

    # ---------------- 3 · Umbrales ----------------
    st.markdown('<div class="sec-title">3 · ¿Qué umbral usar?</div>'
                '<div class="sec-sub">El umbral decide desde qué probabilidad se marca VACANTE. Cada criterio '
                'prioriza algo distinto (valores de validación).</div>', unsafe_allow_html=True)
    crits = UMBRALES_VAL.Criterio.tolist()
    crit_side = criterio.split(" (")[0]
    sel = st.segmented_control("Criterio", crits, default=crit_side if crit_side in crits else crits[0],
                               key="res_umbral") or crits[0]
    r = UMBRALES_VAL.set_index("Criterio").loc[sel]
    u = st.columns(4)
    u[0].markdown(card("Umbral", f"{r.Umbral:.1%}", "probabilidad mínima para marcar VACANTE"), unsafe_allow_html=True)
    u[1].markdown(card("Precisión", f"{r['Precisión']:.0%}", f"≈ {round(r['Precisión'] * 10)} de cada 10 alertas aciertan"), unsafe_allow_html=True)
    u[2].markdown(card("Recall", f"{r.Recall:.0%}", f"detecta ≈ {round(r.Recall * 10)} de cada 10 vacantes"), unsafe_allow_html=True)
    u[3].markdown(card("Costo (validación)", f"{r['Costo validación']:,.0f}", "con FN = 100 y FP = 5"), unsafe_allow_html=True)
    fig = go.Figure()
    for nombre, col in [("Precisión", AZUL), ("Recall", NARANJA)]:
        fig.add_bar(name=nombre, x=crits, y=UMBRALES_VAL[nombre], marker_color=col,
                    marker_line=dict(width=[3 if c == sel else 0 for c in crits], color="white"),
                    text=[f"{x:.0%}" for x in UMBRALES_VAL[nombre]], textposition="outside", cliponaxis=False)
    fig.update_layout(barmode="group", yaxis=dict(range=[0, 1.05], tickformat=".0%"),
                      legend=dict(orientation="h", y=1.12, x=0))
    st.plotly_chart(estilo_fig(fig, 320), width="stretch")
    st.caption("Al bajar el umbral sube el recall (se escapan menos vacantes) pero baja la precisión (más falsas alarmas). "
               "Maximizar F1 no es lo mismo que minimizar el costo.")

    # ---------------- 4 · Ablación ----------------
    st.markdown('<div class="sec-title">4 · ¿Qué información usa el modelo?</div>'
                '<div class="sec-sub">Se reentrenó quitando cada grupo de variables y se midió cuánto cae la AP '
                '(modelo completo: 0.750).</div>', unsafe_allow_html=True)
    abl = ABLACION[ABLACION["Configuración"] != "Todas las variables"].copy()
    abl["Cambio"] = abl["AP validación"] - 0.7501
    abl = abl.sort_values("Cambio", ascending=False)
    fig = go.Figure(go.Bar(
        x=abl.Cambio, y=abl["Configuración"], orientation="h",
        marker_color=[ROJO if x < -0.1 else NARANJA if x < -0.02 else "#8fa3b3" for x in abl.Cambio],
        text=[f"{x:+.3f}" for x in abl.Cambio], textposition="outside", cliponaxis=False,
        hovertemplate="%{y}: %{x:+.3f} de AP<extra></extra>"))
    fig.update_layout(xaxis=dict(title="Cambio en AP frente al modelo completo", range=[-0.23, 0.04]), margin=dict(l=10, r=60))
    st.plotly_chart(estilo_fig(fig, 380), width="stretch")
    ver = st.selectbox("¿Qué significa cada fila?", abl["Configuración"].tolist(), key="abl_sel")
    st.info(EXPLICA_ABLACION.get(ver, ""), icon="🔍")

    # ---------------- 5 · Matriz de confusión ----------------
    st.markdown('<div class="sec-title">5 · ¿En qué acierta y en qué se equivoca?</div>'
                '<div class="sec-sub">Matriz de confusión de LightGBM ajustado en test (336,748 perfiles-mes, umbral máx. F1).</div>',
                unsafe_allow_html=True)
    tn, fp, fn, tp = 330603, 834, 1814, 3497
    st.markdown(f"""
<div class="cm">
  <div></div><div class="h">El modelo dijo <b>OCUPADO</b></div><div class="h">El modelo dijo <b>VACANTE</b></div>
  <div class="r">En realidad<br><b>OCUPADO</b></div>
  <div class="c ok"><div class="lb">✅ Ocupados bien predichos</div><div class="num">{tn:,}</div>
    <div class="pc">{tn / (tn + fp):.1%} de los ocupados reales</div></div>
  <div class="c warn"><div class="lb">⚠️ Falsas alarmas</div><div class="num">{fp:,}</div>
    <div class="pc">{fp / (fp + tp):.1%} de las alertas</div></div>
  <div class="r">En realidad<br><b>VACANTE</b></div>
  <div class="c bad"><div class="lb">❌ Vacantes no detectadas</div><div class="num">{fn:,}</div>
    <div class="pc">{fn / (fn + tp):.1%} de las vacantes reales</div></div>
  <div class="c ok"><div class="lb">✅ Vacantes detectadas</div><div class="num">{tp:,}</div>
    <div class="pc">{tp / (fn + tp):.1%} de las vacantes reales</div></div>
</div>""", unsafe_allow_html=True)
    st.caption("Calibración sin reajustar: Brier 0.0060 · log loss 0.0241. Para una matriz que cambia con el umbral, "
               "usa la pestaña *Pronóstico por lote* con los perfiles de ejemplo.")

    with st.expander("Ver la tabla completa de métricas (validación y test)"):
        st.markdown("**Test**")
        st.dataframe(TEST_M.round(4), hide_index=True, width="stretch")
        st.markdown("**Validación**")
        st.dataframe(VAL.round(4), hide_index=True, width="stretch")

# ===========================================================================
# 5. SEGMENTACIÓN
# ===========================================================================
with tab_seg:
    st.markdown('<div class="sec-sub" style="font-size:1rem">El algoritmo <b>K-Means</b> juntó los <b>56,482 perfiles</b> '
                'de enero 2026 en <b>8 grupos</b> de perfiles parecidos según sus <b>montos y tamaño</b>, sin saber si '
                'estaban ocupados o vacantes. Sirve para describir qué tipos de perfiles existen; no interviene en el '
                'pronóstico.</div>', unsafe_allow_html=True)

    med = leer_csv_ns("medianas_clusters.csv", index_col=0)
    tam = leer_csv_ns("tamanos_clusters.csv", index_col=0)
    if tam is None or "perfiles" not in tam:
        tam = CLUSTERS.assign(cluster=range(8)).set_index("cluster")[["Perfiles"]].rename(columns={"Perfiles": "perfiles"})
    tam["porcentaje"] = 100 * tam["perfiles"] / tam["perfiles"].sum()
    nombres = nombrar_segmentos(med) if med is not None else {c: f"S{c}" for c in tam.index}
    tam["nombre"] = [nombres.get(c, f"S{c}") for c in tam.index]

    g1, g2 = st.columns([1.35, 1])
    with g1:
        t_ = tam.sort_values("perfiles")
        fig = go.Figure(go.Bar(
            x=t_.perfiles, y=t_.nombre, orientation="h", marker_color=TEAL,
            text=[f"{p:,} · {q:.1f}%" for p, q in zip(t_.perfiles, t_.porcentaje)],
            textposition="outside", cliponaxis=False,
            hovertemplate="%{y}<br>%{x:,} perfiles<extra></extra>"))
        fig.update_layout(title="¿Cuántos perfiles hay en cada segmento?", xaxis_title="Perfiles",
                          xaxis=dict(range=[0, t_.perfiles.max() * 1.35]), margin=dict(l=10, r=20))
        st.plotly_chart(estilo_fig(fig, 400), width="stretch")
    with g2:
        st.markdown("##### ¿Por qué K-Means?")
        st.dataframe(METODOS_NS, hide_index=True, width="stretch")
        st.markdown("""
- **Silhouette**: qué tan separados quedan los grupos (más alto = mejor).
- **DBSCAN** separa un poco mejor, pero deja **23 % de perfiles sin grupo** (“ruido”).
- **K-Means** agrupa a **todos** con una separación buena (0.57). Por eso se eligió.
""")
    if med is None:
        st.info("Para ver los nombres y la ficha de cada segmento, copia la carpeta `resultados_no_supervisado` "
                "del zip junto a `app.py`.", icon="📁")
    else:
        st.caption("Los nombres se generan automáticamente según la variable en la que cada segmento más se distingue.")

        # ---------- Explorador de un segmento ----------
        st.markdown('<div class="sec-title">Explora un segmento</div>', unsafe_allow_html=True)
        opciones_seg = tam.sort_values("perfiles", ascending=False).index.tolist()
        c_sel = st.selectbox("Segmento", opciones_seg, format_func=lambda c: nombres[c], key="seg_sel")
        rel = rango_relativo(med)
        fila_rel = rel.loc[c_sel]
        destaca = (fila_rel - rel.drop(index=c_sel).mean()).sort_values(ascending=False)
        top = [ETIQUETAS.get(v, v).lower() for v in destaca.index[:2]]
        bajo = ETIQUETAS.get(destaca.index[-1], destaca.index[-1]).lower()
        f = st.columns(3)
        f[0].markdown(card("Perfiles", f"{int(tam.loc[c_sel, 'perfiles']):,}", f"{tam.loc[c_sel, 'porcentaje']:.1f} % del total"), unsafe_allow_html=True)
        f[1].markdown(card_txt("Destaca por", top[0].capitalize(), f"y {top[1]}"), unsafe_allow_html=True)
        f[2].markdown(card_txt("Más bajo en", bajo.capitalize(), "comparado con los otros segmentos"), unsafe_allow_html=True)

        e1, e2 = st.columns([1.3, 1])
        with e1:
            orden_v = fila_rel.sort_values().index
            fig = go.Figure(go.Bar(
                x=fila_rel[orden_v], y=[ETIQUETAS.get(v, v) for v in orden_v], orientation="h",
                marker=dict(color=fila_rel[orden_v], colorscale=[[0, "#3a4a56"], [1, TEAL]], cmin=0, cmax=1),
                customdata=med.loc[c_sel, orden_v].values,
                text=[f"S/ {x:,.0f}" if v != "CANTIDAD_TOTAL" else f"{x:,.0f} plazas"
                      for v, x in zip(orden_v, med.loc[c_sel, orden_v])],
                textposition="outside", cliponaxis=False,
                hovertemplate="%{y}<br>Mediana: %{customdata:,.1f}<extra></extra>"))
            fig.update_layout(title="Posición frente a los otros segmentos (bajo → alto)",
                              xaxis=dict(range=[0, 1.35], tickvals=[0, .5, 1], ticktext=["más bajo", "medio", "más alto"]),
                              margin=dict(l=10, r=10))
            st.plotly_chart(estilo_fig(fig, 430), width="stretch")
            st.caption("El número junto a cada barra es la mediana del segmento.")
        with e2:
            for var, titulo in [("SECTOR", "Sectores principales"), ("DESC_REGIMEN_LABORAL", "Regímenes principales")]:
                comp = leer_csv_ns(f"composicion_{var}.csv", index_col=0)
                if comp is not None and c_sel in comp.index:
                    top5 = comp.loc[c_sel].sort_values(ascending=False).head(5)[::-1]
                    fig = go.Figure(go.Bar(x=top5.values, y=[str(i)[:32] for i in top5.index], orientation="h",
                                           marker_color=AZUL, text=[f"{x:.0f}%" for x in top5.values],
                                           textposition="outside", cliponaxis=False))
                    fig.update_layout(title=titulo, xaxis=dict(visible=False, range=[0, top5.max() * 1.3]),
                                      margin=dict(l=10, r=10, t=40))
                    st.plotly_chart(estilo_fig(fig, 215), width="stretch")

        with st.expander("Ver el mapa completo de los 8 segmentos"):
            fig = px.imshow(rel.T.values, x=[nombres[c] for c in rel.index],
                            y=[ETIQUETAS.get(v, v) for v in rel.columns], aspect="auto",
                            color_continuous_scale=[[0, "#1f2a33"], [1, TEAL]], zmin=0, zmax=1)
            fig.update_traces(customdata=med.T.values,
                              hovertemplate="%{y}<br>%{x}<br>Mediana: %{customdata:,.1f}<extra></extra>")
            fig.update_layout(coloraxis_colorbar=dict(title="", tickvals=[0, 1], ticktext=["bajo", "alto"]),
                              xaxis=dict(tickangle=-30))
            st.plotly_chart(estilo_fig(fig, 480), width="stretch")
            st.caption("Color relativo por fila: oscuro = el segmento con menor valor en esa variable; verde = el mayor.")

# ===========================================================================
# 6. METODOLOGÍA
# ===========================================================================
with tab_met:
    st.markdown('<div class="sec-title" style="margin-top:.4rem">El proceso en una línea</div>', unsafe_allow_html=True)
    flujo = [("📥", "3.65 M registros", "AIRHSP ene-24 a ago-26"), ("🧹", "Limpieza y EDA", "nulos, montos por plaza, fugas"),
             ("🧩", "Panel por perfil", "1.68 M perfil-mes"), ("📅", "Train · Validación · Test", "separados por fechas"),
             ("🤖", "6 modelos + 2 reglas", "desbalance, ajuste, umbrales"), ("🏆", "LightGBM ajustado", "AP test 0.783")]
    st.markdown('<div class="ex" style="flex-wrap:nowrap">' + '<div class="arrow" style="font-size:1.1rem">→</div>'.join(
        f'<div class="box" style="min-width:0;padding:.8rem .8rem"><div style="font-size:1.4rem">{i}</div>'
        f'<div class="v" style="font-size:.95rem">{t}</div><div style="font-size:.78rem;opacity:.7;margin-top:.2rem">{d}</div></div>'
        for i, t, d in flujo) + '</div>', unsafe_allow_html=True)

    # ---------- Línea de tiempo ----------
    st.markdown('<div class="sec-title">¿Cómo se separaron los datos?</div>'
                '<div class="sec-sub">Por fechas, no al azar: el modelo aprende con el pasado y se evalúa en meses '
                'que nunca vio, como ocurriría en la realidad.</div>', unsafe_allow_html=True)
    tramos = pd.DataFrame([
        ("Entrenamiento", "2024-01-01", "2025-09-01", TEAL, "Ene-24 → ago-25 · 1.07 M filas"),
        ("Embargo", "2025-09-01", "2025-10-01", "#5d6f7e", "Set-25 excluido: su resultado caería en validación"),
        ("Validación", "2025-10-01", "2026-02-01", AZUL, "Oct-25 → ene-26 · aquí se eligió el modelo y el umbral"),
        ("Test", "2026-02-01", "2026-08-01", NARANJA, "Feb-26 → jul-26 · 336,748 filas · prueba final"),
        ("Solo resultado", "2026-08-01", "2026-09-01", "#8fa3b3", "Ago-26: solo aporta el estado futuro de julio"),
    ], columns=["Tramo", "Inicio", "Fin", "Color", "Detalle"])
    fig = go.Figure()
    for _, r in tramos.iterrows():
        ini, fin = pd.Timestamp(r.Inicio), pd.Timestamp(r.Fin)
        fig.add_bar(x=[(fin - ini).days * 86400000], base=[ini], y=["Datos"], orientation="h",
                    marker_color=r.Color, name=r.Tramo, hovertemplate=f"<b>{r.Tramo}</b><br>{r.Detalle}<extra></extra>",
                    text=r.Tramo if r.Tramo in ("Entrenamiento", "Validación", "Test") else "",
                    textposition="inside", insidetextanchor="middle", textfont=dict(color="white", size=14))
    ticks = pd.date_range("2024-01-01", "2026-09-01", freq="6MS")
    meses_cortos = ["ene", "feb", "mar", "abr", "may", "jun", "jul", "ago", "set", "oct", "nov", "dic"]
    fig.update_layout(barmode="overlay", showlegend=False, yaxis=dict(visible=False), margin=dict(t=10, b=10),
                      xaxis=dict(type="date", tickvals=ticks, ticktext=[f"{meses_cortos[d.month - 1]} {d.year}" for d in ticks]))
    st.plotly_chart(estilo_fig(fig, 150), width="stretch")
    st.markdown('<div class="chips">' + "".join(
        f'<span class="chip" style="background:{c}33;border-color:{c}"><b style="color:{c}">■</b> {t}: {d}</span>'
        for t, c, d in zip(tramos.Tramo, tramos.Color, tramos.Detalle)) + '</div>', unsafe_allow_html=True)

    # ---------- Decisiones ----------
    st.markdown('<div class="sec-title">Decisiones clave</div>', unsafe_allow_html=True)
    dec = [
        ("🎯", "¿Qué se predice?", "El estado predominante (OCUPADO o VACANTE) de cada perfil en el mes siguiente."),
        ("🧩", "¿Qué es un perfil?", "Un grupo de plazas con las mismas 10 características. No es una persona ni una plaza individual."),
        ("🚫", "¿Qué se dejó fuera?", "El régimen pensionario: su ausencia coincidía casi siempre con VACANTE y “delataba” la respuesta."),
        ("📅", "¿Cómo se validó?", "Por fechas y con un mes de separación (embargo) entre entrenamiento y validación."),
        ("⚙️", "¿Cómo se prepararon los datos?", "Codificación y escalado aprendidos solo con datos de entrenamiento, dentro del pipeline."),
        ("📏", "¿Con qué se midió?", "Average Precision: las vacantes son solo ~3 %, así que el accuracy (99 %) engaña."),
        ("⚖️", "¿Contra qué se comparó?", "La regla “el próximo mes estará igual que hoy” (persistencia), que es difícil de superar."),
        ("💸", "¿Qué costos se supusieron?", "Perder una vacante = 100; falsa alarma = 5. Son supuestos, no gastos reales."),
    ]
    st.markdown('<div class="grid g4">' + "".join(
        f'<div class="icard"><div class="ico">{i}</div><div class="t">{t}</div><div class="d">{d}</div></div>'
        for i, t, d in dec) + "</div>", unsafe_allow_html=True)

    # ---------- Variables ----------
    st.markdown('<div class="sec-title">¿Qué información recibe el modelo? (25 variables)</div>', unsafe_allow_html=True)
    bloques = [
        ("📌", "Estado y tamaño", ["ESTADO_REGISTRO", "CANTIDAD_TOTAL"], "Cómo está hoy y cuántas plazas agrupa."),
        ("📆", "Tiempo", ["MES"], "Mes del año (codificado como seno y coseno para que dic. y ene. queden cerca)."),
        ("🏛️", "Perfil institucional", ["NIVEL", "SECTOR", "PLIEGO", "UNIDAD_EJECUTORA", "TIPO_ESTABLECIMIENTO",
                                         "DESC_TIPO_REGISTRO", "DESC_SUB_TIPO_REGISTRO", "DESC_REGIMEN_LABORAL",
                                         "DESC_GRUPO_OCASIONAL", "DESC_CARGO_ESTRUCUTURAL", "DESC_CONDICION_LABORAL"],
         "Quién es el grupo: entidad, régimen, cargo…"),
        ("💰", "Montos por plaza", MONETARIAS, "Cuánto cuesta cada plaza (total del grupo ÷ cantidad)."),
    ]
    st.markdown('<div class="grid g4">' + "".join(
        f'<div class="icard"><div class="ico">{i}</div><div class="t">{t} · {len(vs) if vs != ["MES"] else 2}</div>'
        f'<div class="d">{d}</div><div class="chips">' + "".join(f'<span class="chip">{ETIQUETAS.get(v, v)}</span>' for v in vs)
        + '</div></div>' for i, t, vs, d in bloques) + "</div>", unsafe_allow_html=True)
    with st.expander("Ver nombres técnicos de las 25 variables"):
        grupos = pd.DataFrame({
            "Grupo": ["Montos por plaza"] * 10 + ["Tamaño", "Tiempo", "Tiempo"] + ["Perfil institucional"] * 3
                     + ["Perfil institucional"] * 8 + ["Estado"],
            "Variable": FEATURES, "Descripción": [ETIQUETAS[f] for f in FEATURES]})
        st.dataframe(grupos, hide_index=True, width="stretch", height=330)

    # ---------- Limitaciones ----------
    st.markdown('<div class="sec-title">Limitaciones y qué significan</div>', unsafe_allow_html=True)
    lims = [
        ("El test ya se revisó durante el proyecto.", "Para confirmar el modelo de forma independiente harán falta meses nuevos de datos."),
        ("Agosto 2026 tiene una fila VACANTE enorme (81 % de las vacantes del mes).", "Podría ser un registro atípico; se analizó con escenarios de sensibilidad, sin borrarlo."),
        ("El rendimiento cambia según el mes (AP 0.65 en diciembre vs 0.90 en octubre).", "En algunos meses las alertas son menos confiables; conviene vigilarlo."),
        ("Las explicaciones del modelo no son causas.", "Que una variable pese mucho no significa que provoque la vacante."),
        ("El perfil es un grupo, no una plaza.", "El pronóstico describe al grupo; no dice qué plaza individual quedará vacante."),
    ]
    st.markdown("".join(f'<div class="lim"><b>⚠️ {a}</b><span>→ {b}</span></div>' for a, b in lims), unsafe_allow_html=True)
    st.caption("Proyecto académico · UTEC · Aprendizaje automático aplicado · 2026")