"""
Rappi Operations Intelligence — Chatbot + Insights Automáticos
Streamlit Application
"""
import streamlit as st
import pandas as pd
from pathlib import Path
from dotenv import load_dotenv
import os

load_dotenv()

from src.data_loader import load_data, METRIC_DICTIONARY, COUNTRY_NAMES
from src.llm_engine import create_model, query
from src.insights import generate_all_insights, format_executive_report

# ─────────────────────────────────────────────
# Configuración de página
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Rappi Operations Intelligence",
    page_icon="🟠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
# CSS — Apple / Claude Design System
# ─────────────────────────────────────────────
st.markdown("""
<style>
/* ── Fuente SF Pro (Apple system font stack) ── */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

*, *::before, *::after {
    font-family: -apple-system, "SF Pro Display", "SF Pro Text", BlinkMacSystemFont,
                 "Inter", "Helvetica Neue", Arial, sans-serif !important;
    -webkit-font-smoothing: antialiased;
    -moz-osx-font-smoothing: grayscale;
    letter-spacing: -0.01em;
}

/* ── App base ── */
.stApp {
    background: #f5f5f7 !important;
}

/* ── Sidebar glass ── */
[data-testid="stSidebar"] {
    background: rgba(255,255,255,0.85) !important;
    backdrop-filter: blur(20px) saturate(180%) !important;
    -webkit-backdrop-filter: blur(20px) saturate(180%) !important;
    border-right: 1px solid rgba(0,0,0,0.06) !important;
    box-shadow: 2px 0 24px rgba(0,0,0,0.04) !important;
}
[data-testid="stSidebar"] * {
    color: #1d1d1f !important;
}

/* ── Main content area ── */
.main .block-container {
    max-width: 860px !important;
    padding: 2rem 2rem 4rem !important;
    animation: fadeInUp 0.4s cubic-bezier(0.16,1,0.3,1) both;
}

/* ── Animations ── */
@keyframes fadeInUp {
    from { opacity: 0; transform: translateY(16px); }
    to   { opacity: 1; transform: translateY(0);    }
}
@keyframes fadeIn {
    from { opacity: 0; }
    to   { opacity: 1; }
}
@keyframes slideInLeft {
    from { opacity: 0; transform: translateX(-12px); }
    to   { opacity: 1; transform: translateX(0);     }
}
@keyframes slideInRight {
    from { opacity: 0; transform: translateX(12px); }
    to   { opacity: 1; transform: translateX(0);    }
}
@keyframes pulse {
    0%, 100% { opacity: 1;   }
    50%       { opacity: 0.5; }
}

/* ── Chat messages ── */
[data-testid="stChatMessage"] {
    animation: fadeInUp 0.35s cubic-bezier(0.16,1,0.3,1) both;
    background: rgba(255,255,255,0.9) !important;
    border: 1px solid rgba(0,0,0,0.06) !important;
    border-radius: 16px !important;
    padding: 1rem 1.25rem !important;
    margin-bottom: 0.75rem !important;
    backdrop-filter: blur(10px) !important;
    box-shadow: 0 1px 3px rgba(0,0,0,0.04), 0 4px 12px rgba(0,0,0,0.03) !important;
    transition: box-shadow 0.2s ease, transform 0.2s ease !important;
}
[data-testid="stChatMessage"]:hover {
    box-shadow: 0 4px 16px rgba(0,0,0,0.08) !important;
    transform: translateY(-1px) !important;
}
/* User message accent */
[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) {
    border-left: 3px solid #FF5A00 !important;
    background: rgba(255,90,0,0.03) !important;
}

/* ── Chat input ── */
[data-testid="stChatInput"] {
    border-radius: 14px !important;
    border: 1.5px solid rgba(0,0,0,0.1) !important;
    background: rgba(255,255,255,0.95) !important;
    backdrop-filter: blur(20px) !important;
    box-shadow: 0 2px 16px rgba(0,0,0,0.06) !important;
    transition: border-color 0.2s ease, box-shadow 0.2s ease !important;
}
[data-testid="stChatInput"]:focus-within {
    border-color: #FF5A00 !important;
    box-shadow: 0 0 0 3px rgba(255,90,0,0.1), 0 2px 16px rgba(0,0,0,0.06) !important;
}

/* ── Buttons ── */
.stButton > button {
    border-radius: 10px !important;
    border: 1px solid rgba(0,0,0,0.1) !important;
    background: rgba(255,255,255,0.9) !important;
    color: #1d1d1f !important;
    font-weight: 500 !important;
    font-size: 0.875rem !important;
    padding: 0.5rem 1rem !important;
    transition: all 0.18s cubic-bezier(0.16,1,0.3,1) !important;
    box-shadow: 0 1px 4px rgba(0,0,0,0.06) !important;
    letter-spacing: -0.01em !important;
}
.stButton > button:hover {
    background: #FF5A00 !important;
    color: #ffffff !important;
    border-color: #FF5A00 !important;
    box-shadow: 0 4px 14px rgba(255,90,0,0.3) !important;
    transform: translateY(-1px) !important;
}
.stButton > button:active {
    transform: scale(0.97) !important;
}

/* ── Download button ── */
[data-testid="stDownloadButton"] > button {
    border-radius: 10px !important;
    background: rgba(0,0,0,0.04) !important;
    border: 1px solid rgba(0,0,0,0.08) !important;
    color: #1d1d1f !important;
    font-size: 0.8rem !important;
    font-weight: 500 !important;
    transition: all 0.18s ease !important;
}
[data-testid="stDownloadButton"] > button:hover {
    background: #1d1d1f !important;
    color: white !important;
    transform: translateY(-1px) !important;
}

/* ── Expanders ── */
[data-testid="stExpander"] {
    border: 1px solid rgba(0,0,0,0.06) !important;
    border-radius: 12px !important;
    background: rgba(255,255,255,0.7) !important;
    backdrop-filter: blur(10px) !important;
    overflow: hidden !important;
    transition: box-shadow 0.2s ease !important;
}
[data-testid="stExpander"]:hover {
    box-shadow: 0 2px 12px rgba(0,0,0,0.06) !important;
}
[data-testid="stExpander"] summary {
    font-weight: 500 !important;
    font-size: 0.875rem !important;
    color: #6e6e73 !important;
    padding: 0.75rem 1rem !important;
}

/* ── Metrics / KPI cards ── */
[data-testid="stMetric"] {
    background: rgba(255,255,255,0.9) !important;
    border: 1px solid rgba(0,0,0,0.06) !important;
    border-radius: 14px !important;
    padding: 1rem 1.25rem !important;
    box-shadow: 0 1px 4px rgba(0,0,0,0.04) !important;
    transition: transform 0.2s ease, box-shadow 0.2s ease !important;
    animation: fadeInUp 0.4s cubic-bezier(0.16,1,0.3,1) both;
}
[data-testid="stMetric"]:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 6px 20px rgba(0,0,0,0.08) !important;
}
[data-testid="stMetricLabel"] { font-size: 0.8rem !important; color: #6e6e73 !important; font-weight: 500 !important; }
[data-testid="stMetricValue"] { font-size: 1.75rem !important; font-weight: 700 !important; color: #1d1d1f !important; letter-spacing: -0.03em !important; }

/* ── Tabs ── */
[data-testid="stTabs"] [role="tablist"] {
    background: rgba(0,0,0,0.04) !important;
    border-radius: 10px !important;
    padding: 3px !important;
    gap: 2px !important;
    border-bottom: none !important;
}
[data-testid="stTabs"] [role="tab"] {
    border-radius: 8px !important;
    font-weight: 500 !important;
    font-size: 0.85rem !important;
    color: #6e6e73 !important;
    transition: all 0.18s ease !important;
    padding: 0.4rem 0.9rem !important;
    border: none !important;
}
[data-testid="stTabs"] [role="tab"][aria-selected="true"] {
    background: rgba(255,255,255,0.95) !important;
    color: #1d1d1f !important;
    font-weight: 600 !important;
    box-shadow: 0 1px 6px rgba(0,0,0,0.1) !important;
}

/* ── DataFrames ── */
[data-testid="stDataFrame"] {
    border-radius: 12px !important;
    overflow: hidden !important;
    border: 1px solid rgba(0,0,0,0.06) !important;
    animation: fadeIn 0.3s ease both;
    box-shadow: 0 1px 4px rgba(0,0,0,0.04) !important;
}

/* ── Plotly charts ── */
[data-testid="stPlotlyChart"] {
    border-radius: 14px !important;
    overflow: hidden !important;
    animation: fadeInUp 0.4s cubic-bezier(0.16,1,0.3,1) both;
    border: 1px solid rgba(0,0,0,0.06) !important;
    box-shadow: 0 2px 12px rgba(0,0,0,0.05) !important;
}

/* ── Selectbox / Inputs ── */
[data-testid="stSelectbox"] > div > div,
[data-testid="stTextInput"] > div > div > input {
    border-radius: 10px !important;
    border: 1.5px solid rgba(0,0,0,0.1) !important;
    background: rgba(255,255,255,0.9) !important;
    transition: border-color 0.18s ease, box-shadow 0.18s ease !important;
}
[data-testid="stTextInput"] > div > div > input:focus {
    border-color: #FF5A00 !important;
    box-shadow: 0 0 0 3px rgba(255,90,0,0.1) !important;
}

/* ── Headers & typography ── */
h1, h2, h3 {
    font-weight: 700 !important;
    letter-spacing: -0.03em !important;
    color: #1d1d1f !important;
}
h1 { font-size: 1.9rem !important; }
h2 { font-size: 1.5rem !important; }
h3 { font-size: 1.15rem !important; }
p, li, span { color: #3d3d3f; font-size: 0.9rem; }

/* ── Alerts / Info boxes ── */
[data-testid="stAlert"] {
    border-radius: 12px !important;
    border: none !important;
    backdrop-filter: blur(10px) !important;
    animation: fadeIn 0.3s ease both;
}

/* ── Spinner ── */
[data-testid="stSpinner"] {
    animation: pulse 1.5s ease-in-out infinite !important;
}

/* ── Radio buttons (nav) ── */
[data-testid="stRadio"] label {
    border-radius: 8px !important;
    padding: 0.4rem 0.75rem !important;
    transition: background 0.15s ease !important;
    font-weight: 500 !important;
    font-size: 0.875rem !important;
}
[data-testid="stRadio"] label:hover {
    background: rgba(255,90,0,0.08) !important;
}

/* ── Insight cards ── */
.insight-critical {
    border-left: 3px solid #FF3B30;
    padding: 14px 16px;
    margin: 10px 0;
    background: rgba(255,59,48,0.04);
    border-radius: 0 12px 12px 0;
    animation: slideInLeft 0.3s cubic-bezier(0.16,1,0.3,1) both;
    transition: transform 0.2s ease, box-shadow 0.2s ease;
}
.insight-warning {
    border-left: 3px solid #FF9500;
    padding: 14px 16px;
    margin: 10px 0;
    background: rgba(255,149,0,0.04);
    border-radius: 0 12px 12px 0;
    animation: slideInLeft 0.3s cubic-bezier(0.16,1,0.3,1) both;
    transition: transform 0.2s ease, box-shadow 0.2s ease;
}
.insight-info {
    border-left: 3px solid #007AFF;
    padding: 14px 16px;
    margin: 10px 0;
    background: rgba(0,122,255,0.04);
    border-radius: 0 12px 12px 0;
    animation: slideInLeft 0.3s cubic-bezier(0.16,1,0.3,1) both;
    transition: transform 0.2s ease, box-shadow 0.2s ease;
}
.insight-critical:hover,
.insight-warning:hover,
.insight-info:hover {
    transform: translateX(4px);
    box-shadow: 0 2px 12px rgba(0,0,0,0.07);
}

/* ── Example question buttons ── */
.stButton > button[kind="secondary"] {
    text-align: left !important;
    justify-content: flex-start !important;
}

/* ── Code blocks ── */
[data-testid="stCode"] {
    border-radius: 12px !important;
    overflow: hidden !important;
    font-size: 0.8rem !important;
}

/* ── Dividers ── */
hr {
    border: none !important;
    border-top: 1px solid rgba(0,0,0,0.07) !important;
    margin: 1rem 0 !important;
}

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 6px; height: 6px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: rgba(0,0,0,0.15); border-radius: 99px; }
::-webkit-scrollbar-thumb:hover { background: rgba(0,0,0,0.25); }

/* ── Success / Error badges ── */
[data-testid="stAlert"][data-baseweb="notification"] {
    border-radius: 12px !important;
}
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# Carga de datos (cached)
# ─────────────────────────────────────────────
@st.cache_data
def get_data():
    return load_data()

@st.cache_data
def get_insights(_df_metrics, _df_orders):
    insights = generate_all_insights(_df_metrics, _df_orders)
    report = format_executive_report(insights)
    return insights, report


# ─────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────
with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/6/69/Rappi_logo.svg/1200px-Rappi_logo.svg.png", width=120)
    st.title("Rappi Operations Intelligence")
    st.caption("Sistema de Análisis Inteligente para Operaciones")

    st.divider()

    # API Key input
    api_key = st.text_input(
        "🔑 Groq API Key",
        type="password",
        value=os.getenv("GROQ_API_KEY", ""),
        help="Obtén tu API key en console.groq.com",
    )

    st.divider()

    # Navegación
    page = st.radio(
        "Navegación",
        ["💬 Chatbot de Datos", "📊 Insights Automáticos", "📋 Explorador de Datos"],
        index=0,
    )

    st.divider()

    # Info de datos
    st.markdown("### 📁 Datos cargados")
    try:
        df_metrics, df_orders = get_data()
        st.success(f"✅ Métricas: {df_metrics.shape[0]:,} filas")
        st.success(f"✅ Órdenes: {df_orders.shape[0]:,} filas")
        st.caption(f"Países: {', '.join(sorted(df_metrics['COUNTRY'].unique()))}")
        st.caption(f"Métricas: {df_metrics['METRIC'].nunique()} tipos")
    except Exception as e:
        st.error(f"Error cargando datos: {e}")
        st.stop()

    st.divider()
    st.caption("Desarrollado por Pedro Esteban")
    st.caption("AI Engineer — Prueba Técnica Rappi 2025")


# ─────────────────────────────────────────────
# Página: Chatbot
# ─────────────────────────────────────────────
if page == "💬 Chatbot de Datos":

    st.header("💬 Chatbot de Datos Operacionales")
    st.caption("Pregunta en lenguaje natural sobre las métricas de Rappi. "
               "El bot genera y ejecuta código pandas automáticamente.")

    # Validar API key
    if not api_key:
        st.warning("⚠️ Ingresa tu Groq API Key en la barra lateral para comenzar.")
        st.info("1. Ve a [console.groq.com](https://console.groq.com)\n"
                "2. Login con Google → 'API Keys' → 'Create API Key'\n"
                "3. Copia la key y pégala en la barra lateral")
        st.stop()

    # Inicializar estado
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "conversation_history" not in st.session_state:
        st.session_state.conversation_history = []

    # Preguntas de ejemplo
    examples = [
        "¿Cuáles son las 5 zonas con mayor Lead Penetration esta semana?",
        "Compara Perfect Orders entre zonas Wealthy y Non Wealthy en México",
        "Muestra la evolución de Gross Profit UE en Chapinero últimas 8 semanas",
        "¿Cuál es el promedio de Lead Penetration por país?",
        "¿Qué zonas tienen alto Lead Penetration pero bajo Perfect Orders?",
        "¿Cuáles zonas crecen más en órdenes en las últimas 5 semanas?",
    ]

    if not st.session_state.messages:
        st.markdown("#### 💡 Preguntas de ejemplo")
        cols = st.columns(2)
        for i, example in enumerate(examples):
            col = cols[i % 2]
            if col.button(f"📌 {example}", key=f"example_{i}", width="stretch"):
                st.session_state.pending_question = example
                st.rerun()

    # Mostrar historial
    for msg_idx, msg in enumerate(st.session_state.messages):
        with st.chat_message(msg["role"], avatar="🧑‍💻" if msg["role"] == "user" else "🟠"):
            st.markdown(msg["content"])
            if "chart" in msg and msg["chart"] is not None:
                try:
                    st.plotly_chart(msg["chart"], use_container_width=True, key=f"chart_hist_{msg_idx}")
                except Exception:
                    st.caption("⚠️ No se pudo renderizar el gráfico del historial.")
            if "dataframe" in msg and msg["dataframe"] is not None:
                with st.expander("📋 Ver datos"):
                    st.dataframe(msg["dataframe"], width="stretch")
                    st.download_button(
                        "⬇️ Descargar CSV",
                        data=msg["dataframe"].to_csv(index=False),
                        file_name=f"rappi_resultado_{msg_idx + 1}.csv",
                        mime="text/csv",
                        key=f"csv_hist_{msg_idx}",
                    )
            if "code" in msg and msg["code"]:
                with st.expander("🔧 Ver código generado"):
                    st.code(msg["code"], language="python")
            if "suggestions" in msg and msg["suggestions"]:
                st.markdown("**💡 Sugerencias:**")
                for sug in msg["suggestions"]:
                    if st.button(f"→ {sug}", key=f"sug_hist_{msg_idx}_{hash(sug)}"):
                        st.session_state.pending_question = sug
                        st.rerun()

    # Manejar pregunta pendiente (de botones)
    pending = st.session_state.pop("pending_question", None)

    # Input del usuario
    user_input = st.chat_input("Pregunta sobre las operaciones de Rappi...")
    question = pending or user_input

    if question:
        # Mostrar mensaje del usuario
        st.session_state.messages.append({"role": "user", "content": question})
        with st.chat_message("user", avatar="🧑‍💻"):
            st.markdown(question)

        # Procesar con LLM
        with st.chat_message("assistant", avatar="🟠"):
            with st.spinner("🔍 Analizando datos..."):
                try:
                    model = create_model(api_key)
                    response = query(
                        model=model,
                        question=question,
                        df_metrics=df_metrics,
                        df_orders=df_orders,
                        conversation_history=st.session_state.conversation_history,
                    )

                    # --- Mostrar explicación (limpiar JSON residual) ---
                    explanation = response.get("explanation", "")
                    import re as _re
                    explanation = _re.sub(r'```(?:json)?\s*\{.*?\}\s*```', '', explanation, flags=_re.DOTALL).strip()
                    explanation = _re.sub(r'\{[^{}]*"thinking"[^{}]*\}', '', explanation, flags=_re.DOTALL).strip()

                    if not explanation:
                        explanation = "Análisis completado. Revisa los datos a continuación."

                    st.markdown(explanation)

                    # --- Mostrar error si existe ---
                    if response.get("error"):
                        st.error(f"⚠️ Error en ejecución del código: {response['error']}")

                    # --- Mostrar resultado (tabla) ---
                    result = response.get("result")
                    result_df = None

                    if isinstance(result, pd.DataFrame) and not result.empty:
                        result_df = result
                        st.dataframe(result, width="stretch")
                    elif isinstance(result, pd.Series):
                        result_df = result.to_frame()
                        st.dataframe(result_df, width="stretch")
                    elif result is not None:
                        st.info(f"Resultado: {result}")

                    if result_df is not None:
                        st.download_button(
                            "⬇️ Descargar CSV",
                            data=result_df.to_csv(index=False),
                            file_name="rappi_resultado.csv",
                            mime="text/csv",
                            key=f"csv_new_{len(st.session_state.messages)}",
                        )

                    # --- Mostrar output de print ---
                    output = response.get("output", "")
                    if output and result_df is None:
                        st.text(output)

                    # --- Mostrar gráfico ---
                    chart = response.get("chart")
                    if chart is not None:
                        try:
                            st.plotly_chart(chart, use_container_width=True, key=f"chart_new_{len(st.session_state.messages)}")
                        except Exception as chart_err:
                            st.warning(f"⚠️ Error mostrando gráfico: {chart_err}")
                    elif result_df is not None and len(result_df) > 0:
                        # Fallback: generar gráfico automático si el LLM no generó uno
                        try:
                            numeric_cols = result_df.select_dtypes(include="number").columns.tolist()
                            if numeric_cols and len(result_df.columns) >= 2:
                                import plotly.express as px
                                non_numeric = [c for c in result_df.columns if c not in numeric_cols]
                                x_col = non_numeric[0] if non_numeric else result_df.columns[0]
                                y_col = numeric_cols[0]
                                auto_fig = px.bar(result_df.head(20), x=x_col, y=y_col, title="Visualización automática")
                                auto_fig.update_layout(template="plotly_white", height=400)
                                st.plotly_chart(auto_fig, use_container_width=True, key=f"autochart_new_{len(st.session_state.messages)}")
                                chart = auto_fig  # para guardar en historial
                        except Exception:
                            pass

                    # --- Mostrar código generado ---
                    code = response.get("code", "")
                    if code:
                        with st.expander("🔧 Ver código generado"):
                            st.code(code, language="python")

                    # --- Debug: si no hay resultado ni error, mostrar respuesta cruda ---
                    if result is None and not response.get("error") and not code:
                        raw = response.get("raw_response", "")
                        if raw:
                            with st.expander("🐛 Debug: respuesta cruda del LLM"):
                                st.text(raw[:2000])

                    # --- Sugerencias ---
                    suggestions = response.get("suggestions", [])
                    if isinstance(suggestions, list) and suggestions:
                        st.markdown("**💡 Preguntas relacionadas:**")
                        for sug in suggestions[:3]:
                            if isinstance(sug, str) and sug:
                                if st.button(f"→ {sug}", key=f"sug_new_{len(st.session_state.messages)}_{hash(sug)}"):
                                    st.session_state.pending_question = sug
                                    st.rerun()

                    # Guardar en historial
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": explanation,
                        "chart": chart,
                        "dataframe": result_df,
                        "code": code,
                        "suggestions": suggestions,
                    })

                    # Actualizar historial conversacional para Groq
                    st.session_state.conversation_history.append(
                        {"role": "user", "content": question}
                    )
                    st.session_state.conversation_history.append(
                        {"role": "assistant", "content": explanation}
                    )

                except Exception as e:
                    error_msg = f"Error procesando tu pregunta: {str(e)}"
                    st.error(error_msg)
                    st.session_state.messages.append({
                        "role": "assistant", "content": error_msg
                    })


# ─────────────────────────────────────────────
# Página: Insights Automáticos
# ─────────────────────────────────────────────
elif page == "📊 Insights Automáticos":

    st.header("📊 Sistema de Insights Automáticos")
    st.caption("Análisis automatizado que identifica anomalías, tendencias y oportunidades.")

    with st.spinner("🔍 Generando insights..."):
        insights, report = get_insights(df_metrics, df_orders)

    # KPIs de resumen
    critical = [i for i in insights if i.severity == "critical"]
    warnings = [i for i in insights if i.severity == "warning"]
    info_ins = [i for i in insights if i.severity == "info"]

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Insights", len(insights))
    col2.metric("🔴 Críticos", len(critical))
    col3.metric("🟡 Advertencias", len(warnings))
    col4.metric("🟢 Informativos", len(info_ins))

    st.divider()

    # Tabs por categoría
    tab_report, tab_anomaly, tab_trend, tab_bench, tab_corr, tab_opp = st.tabs([
        "📝 Reporte Ejecutivo",
        "🚨 Anomalías",
        "📉 Tendencias",
        "📊 Benchmarking",
        "🔗 Correlaciones",
        "💡 Oportunidades",
    ])

    with tab_report:
        st.markdown(report)
        st.download_button(
            "📥 Descargar Reporte (Markdown)",
            data=report,
            file_name="rappi_insights_report.md",
            mime="text/markdown",
        )

    def render_insights(insight_list):
        if not insight_list:
            st.info("No se detectaron insights en esta categoría.")
            return
        for ins in insight_list:
            css_class = f"insight-{ins.severity}"
            emoji = "🔴" if ins.severity == "critical" else "🟡" if ins.severity == "warning" else "🟢"
            st.markdown(f"""
            <div class="{css_class}">
                <strong>{emoji} {ins.title}</strong><br>
                <strong>Hallazgo:</strong> {ins.finding}<br>
                <strong>Impacto:</strong> {ins.impact}<br>
                <strong>Recomendación:</strong> {ins.recommendation}
            </div>
            """, unsafe_allow_html=True)

    with tab_anomaly:
        render_insights([i for i in insights if i.category == "anomaly"])

    with tab_trend:
        render_insights([i for i in insights if i.category == "trend"])

    with tab_bench:
        render_insights([i for i in insights if i.category == "benchmark"])

    with tab_corr:
        render_insights([i for i in insights if i.category == "correlation"])

    with tab_opp:
        render_insights([i for i in insights if i.category == "opportunity"])


# ─────────────────────────────────────────────
# Página: Explorador de Datos
# ─────────────────────────────────────────────
elif page == "📋 Explorador de Datos":

    st.header("📋 Explorador de Datos")

    tab_metrics, tab_orders, tab_dict = st.tabs([
        "📈 Métricas Input", "📦 Órdenes", "📖 Diccionario"
    ])

    with tab_metrics:
        st.subheader("Dataset de Métricas Operacionales")

        col1, col2, col3 = st.columns(3)
        countries = ["Todos"] + sorted(df_metrics["COUNTRY"].unique().tolist())
        sel_country = col1.selectbox("País", countries, key="exp_country")
        metrics = ["Todos"] + sorted(df_metrics["METRIC"].unique().tolist())
        sel_metric = col2.selectbox("Métrica", metrics, key="exp_metric")
        zone_types = ["Todos"] + sorted(df_metrics["ZONE_TYPE"].unique().tolist())
        sel_type = col3.selectbox("Tipo de zona", zone_types, key="exp_type")

        filtered = df_metrics.copy()
        if sel_country != "Todos":
            filtered = filtered[filtered["COUNTRY"] == sel_country]
        if sel_metric != "Todos":
            filtered = filtered[filtered["METRIC"] == sel_metric]
        if sel_type != "Todos":
            filtered = filtered[filtered["ZONE_TYPE"] == sel_type]

        st.dataframe(filtered, width="stretch", height=400)
        st.caption(f"Mostrando {len(filtered):,} de {len(df_metrics):,} filas")

    with tab_orders:
        st.subheader("Dataset de Órdenes")

        col1, col2 = st.columns(2)
        countries_o = ["Todos"] + sorted(df_orders["COUNTRY"].unique().tolist())
        sel_country_o = col1.selectbox("País", countries_o, key="exp_country_o")

        filtered_o = df_orders.copy()
        if sel_country_o != "Todos":
            filtered_o = filtered_o[filtered_o["COUNTRY"] == sel_country_o]

        st.dataframe(filtered_o, width="stretch", height=400)
        st.caption(f"Mostrando {len(filtered_o):,} de {len(df_orders):,} filas")

    with tab_dict:
        st.subheader("Diccionario de Métricas")
        for metric, desc in METRIC_DICTIONARY.items():
            st.markdown(f"**{metric}**")
            st.caption(desc)
            st.divider()