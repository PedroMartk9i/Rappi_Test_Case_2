# 🟠 Rappi Operations Intelligence

Sistema de Análisis Inteligente para Operaciones Rappi — Chatbot conversacional de datos + Insights automáticos.

## 🎯 Descripción

Esta solución aborda el caso técnico de AI Engineer para Rappi, proporcionando:

1. **Chatbot Conversacional de Datos (70%):** Un bot que permite a usuarios no técnicos hacer preguntas en lenguaje natural sobre las métricas operacionales de Rappi y recibir respuestas precisas con visualizaciones.

2. **Sistema de Insights Automáticos (30%):** Un motor que analiza automáticamente los datos y genera un reporte ejecutivo con anomalías, tendencias, benchmarking, correlaciones y oportunidades.

## 🏗️ Arquitectura

```
┌─────────────────────────────────────────────┐
│              Streamlit UI                    │
│  ┌──────────────┐  ┌──────────────────────┐ │
│  │   Chatbot    │  │  Insights Dashboard  │ │
│  │  (70% peso)  │  │    (30% peso)        │ │
│  └──────┬───────┘  └──────────┬───────────┘ │
├─────────┼─────────────────────┼─────────────┤
│         ▼                     ▼             │
│  ┌──────────────┐  ┌──────────────────────┐ │
│  │  LLM Engine  │  │  Insights Engine     │ │
│  │  Groq API    │  │  (Determinístico)    │ │
│  │  NL → Pandas │  │  Anomalías/Trends    │ │
│  └──────┬───────┘  └──────────┬───────────┘ │
├─────────┼─────────────────────┼─────────────┤
│         ▼                     ▼             │
│  ┌──────────────────────────────────────┐   │
│  │         Pandas DataFrames            │   │
│  │   df_metrics (12,573 filas)          │   │
│  │   df_orders  (1,242 filas)           │   │
│  └──────────────────────────────────────┘   │
└─────────────────────────────────────────────┘
```

### Decisiones de Diseño

| Decisión | Justificación |
|----------|---------------|
| **Groq + Llama 3.3 70B** | Gratuito, inferencia ultra-rápida (~0.3s/query), excelente generación de código Python |
| **Pandas en memoria** | Dataset pequeño (~14K filas), no requiere DB. Ejecución instantánea |
| **Streamlit** | Prototipado rápido, widgets nativos para chat y charts, deploy sencillo |
| **Plotly** | Gráficos interactivos, integración nativa con Streamlit |
| **Insights determinísticos** | Anomalías y tendencias con reglas claras, reproducibles, sin costo LLM |

### Costo Estimado

- **Groq (Llama 3.3 70B):** Gratuito (hasta 30 requests/min, 14,400/día)
- **Hosting (opcional):** Streamlit Cloud gratuito
- **Costo total por sesión de 10 preguntas:** ~$0.00

## 🚀 Setup Rápido

### Prerrequisitos
- Python 3.10+
- API Key de Groq ([obtener aquí](https://console.groq.com))

### Instalación

```bash
# Clonar repositorio
git clone https://github.com/tu-usuario/rappi-analytics.git
cd rappi-analytics

# Crear entorno virtual
python -m venv venv
source venv/bin/activate  # Linux/Mac
# o: venv\Scripts\activate  # Windows

# Instalar dependencias
pip install -r requirements.txt

# Configurar API Key (opción 1: archivo .env)
cp .env.example .env
# Editar .env con tu GROQ_API_KEY

# Ejecutar
streamlit run app.py
```

La aplicación se abrirá en `http://localhost:8501`.

También puedes ingresar la API Key directamente en la interfaz sin necesidad del archivo `.env`.

## 💬 Capacidades del Chatbot

### Tipos de queries soportados

| Tipo | Ejemplo |
|------|---------|
| **Filtrado** | "¿Cuáles son las 5 zonas con mayor Lead Penetration esta semana?" |
| **Comparación** | "Compara Perfect Orders entre zonas Wealthy y Non Wealthy en México" |
| **Tendencia temporal** | "Muestra la evolución de Gross Profit UE en Chapinero últimas 8 semanas" |
| **Agregación** | "¿Cuál es el promedio de Lead Penetration por país?" |
| **Multivariable** | "¿Qué zonas tienen alto Lead Penetration pero bajo Perfect Orders?" |
| **Inferencia** | "¿Cuáles zonas crecen más en órdenes y qué podría explicarlo?" |

### Características
- ✅ Generación y ejecución automática de código pandas
- ✅ Visualizaciones Plotly dinámicas (barras, líneas, scatter, heatmap)
- ✅ Memoria conversacional (contexto de diálogo)
- ✅ Sugerencias proactivas de análisis
- ✅ Manejo de contexto de negocio ("zonas problemáticas" → métricas deterioradas)
- ✅ Retry automático si el código generado falla
- ✅ Transparencia: código generado visible para verificación

## 📊 Sistema de Insights Automáticos

### Categorías de detección

1. **🚨 Anomalías:** Cambios >10% WoW (deterioro o mejora)
2. **📉 Tendencias:** Métricas en deterioro 3+ semanas consecutivas
3. **📊 Benchmarking:** Zonas >1.5σ debajo del promedio de su grupo (país + tipo)
4. **🔗 Correlaciones:** Relaciones significativas entre métricas (|r| > 0.5)
5. **💡 Oportunidades:** Crecimiento/caída de órdenes en 5 semanas

### Reporte Ejecutivo
- Top 5 hallazgos críticos con finding → impacto → recomendación
- Detalle por categoría
- Descargable en Markdown

## 📂 Estructura del Proyecto

```
rappi-analytics/
├── app.py                    # Aplicación principal Streamlit
├── src/
│   ├── __init__.py
│   ├── data_loader.py        # Carga de datos + schema + diccionario
│   ├── llm_engine.py         # Motor LLM (Gemini → Pandas)
│   └── insights.py           # Detección automática de insights
├── data/
│   └── rappi_data.xlsx       # Dataset proporcionado
├── requirements.txt
├── .env.example
├── .gitignore
└── README.md
```

## ⚠️ Limitaciones y Próximos Pasos

### Limitaciones actuales
- El LLM puede generar código incorrecto en queries muy complejas (~5% de los casos)
- Sin persistencia de conversaciones entre sesiones
- Insights basados en reglas estáticas (no ML adaptativo)

### Mejoras con más tiempo
- **Deployment:** Streamlit Cloud o Railway para acceso sin instalación
- **RAG:** Indexar documentación de métricas para mejor comprensión contextual
- **Fine-tuning:** Optimizar prompts con ejemplos de queries reales de SP&A
- **Alertas automáticas:** Sistema de notificación por email/Slack cuando se detectan anomalías críticas
- **Exportación:** Generación de reportes en PDF con gráficos embebidos
- **Testing:** Suite de tests con queries predefinidas para validar precisión

## 👤 Autor

**Pedro Esteban**
Prueba Técnica — AI Engineer, Rappi 2025
