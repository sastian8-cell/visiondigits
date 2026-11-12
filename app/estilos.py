"""estilos.py — Hoja de estilos: tipografía redondeada, colores alegres y tarjetas grandes para niños."""

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Fredoka:wght@400;500;600;700&display=swap');

/* Fuente infantil para textos; NO se aplica a <span> para no romper los íconos de Streamlit */
.stApp p, .stApp label, .stApp li, .stApp h1, .stApp h2, .stApp h3, .stApp h4,
.stApp input, .stApp textarea, .stApp button p, .stApp div[data-testid="stMetricValue"] div {
    font-family: 'Fredoka', 'Segoe UI', sans-serif !important; }

.vd-titulo { font-size: 2.6rem; font-weight: 700; line-height: 1.1;
             background: linear-gradient(90deg, #6366F1, #EC4899, #F59E0B);
             -webkit-background-clip: text; -webkit-text-fill-color: transparent; }
.vd-subtitulo { color: #64748B; font-size: 1.05rem; margin-bottom: .6rem; }

.stTabs [data-baseweb="tab"] { font-size: 1.05rem; font-weight: 600; padding: .5rem 1rem; }

.vd-banner { border-radius: 22px; padding: 1rem 1.4rem; margin-bottom: .8rem;
             background: linear-gradient(135deg, #EEF2FF 0%, #FCE7F3 100%); border: 3px dashed #A5B4FC; }
.vd-banner h2 { margin: 0; color: #3730A3; font-size: 1.8rem; }
.vd-banner p  { margin: .2rem 0 0 0; color: #475569; font-size: 1.05rem; }
.vd-objetivo { display: inline-block; min-width: 3.2rem; text-align: center; padding: 0 .5rem;
               border-radius: 14px; background: #6366F1; color: white !important; }

.vd-tarjeta { border-radius: 22px; padding: 1rem 1.2rem; background: #FFFFFF;
              border: 2px solid #E2E8F0; box-shadow: 0 6px 18px rgba(99,102,241,.08); }
.vd-numero { font-size: 6.5rem; font-weight: 700; line-height: 1; color: #4F46E5; text-align: center; }
.vd-nombre { text-align: center; font-size: 1.3rem; color: #64748B; margin-top: -.2rem; }

.vd-retro { border-radius: 20px; padding: .9rem 1.2rem; margin: .6rem 0; font-size: 1.1rem; }
.vd-retro h3 { margin: 0 0 .25rem 0; font-size: 1.45rem; }
.vd-exito    { background: #ECFDF5; border: 3px solid #10B981; color: #065F46; }
.vd-bien     { background: #EFF6FF; border: 3px solid #3B82F6; color: #1E3A8A; }
.vd-cerca    { background: #FFFBEB; border: 3px solid #F59E0B; color: #78350F; }
.vd-intentar { background: #FFF1F2; border: 3px solid #FB7185; color: #881337; }

.vd-estrellas { font-size: 1.5rem; text-align: center; }
.vd-muestra-pie { text-align: center; font-size: .85rem; color: #64748B; }
.vd-mejor { text-align: center; font-size: .85rem; color: #059669; font-weight: 700; }
.vd-nota { font-size: .85rem; color: #94A3B8; }

div[data-testid="stMetricValue"] { font-size: 1.6rem; }
.stButton > button { border-radius: 14px; font-weight: 600; }
</style>
"""
