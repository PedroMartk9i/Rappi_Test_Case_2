"""
Motor LLM: convierte preguntas en lenguaje natural a código pandas
y genera visualizaciones automáticas usando Groq (Llama 3.3 70B).
"""
import re
import json
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from groq import Groq
from src.data_loader import get_schema_description


SYSTEM_PROMPT = """Eres un analista de datos experto de Rappi. Tu trabajo es responder preguntas
sobre las operaciones de Rappi analizando datos con pandas.

{schema}

## Instrucciones

Cuando el usuario haga una pregunta sobre datos:
1. Genera código Python/pandas para responderla.
2. El código debe usar los DataFrames `df_metrics` y/o `df_orders` que ya están cargados.
3. El resultado final DEBE guardarse en una variable llamada `result`.
4. Si `result` es un DataFrame, debe tener columnas descriptivas y legibles.

## Formato de respuesta

IMPORTANTE: Responde ÚNICAMENTE con el bloque JSON. NO escribas NADA antes ni después.

```json
{{
  "thinking": "Tu razonamiento breve",
  "code": "código pandas. NUNCA uses import. pd y np ya existen. Guarda resultado en variable result.",
  "explanation": "Explicación clara y completa EN ESPAÑOL para usuario no técnico. NUNCA uses referencias a variables Python como result['col'] o result.values[0] — eso NO se evalúa y se muestra como texto roto. En su lugar escribe texto genérico describiendo qué calcula el código (ej: 'La zona con mayor valor se muestra en la tabla de resultados'). NO menciones nombres de columnas técnicas como L0W_ROLL.",
  "chart": {{
    "type": "bar|line|scatter|heatmap|none",
    "title": "Título del gráfico",
    "x": "columna eje X",
    "y": "columna eje Y",
    "color": null,
    "orientation": "v|h"
  }},
  "suggestions": ["Pregunta 1", "Pregunta 2", "Pregunta 3"]
}}
```

## Reglas CRÍTICAS — lee TODAS
- SOLO responde con el JSON. NADA de texto fuera del JSON.
- NUNCA uses import. pd (pandas) y np (numpy) ya están disponibles.
- Solo UNA variable `result`. NUNCA result2, result3.
- Si necesitas combinar análisis, usa pd.concat() en un solo DataFrame.
- En la explicación, describe qué hace el análisis de forma genérica. NUNCA escribas código Python ni referencias a variables entre llaves. Los datos concretos se muestran en la tabla.
- NO menciones nombres técnicos de columnas (L0W_ROLL, L1W_ROLL). Tradúcelos (ej: "esta semana", "semana anterior").
- Si la pregunta NO es sobre datos, usa code="" y da explicación directa.
- Para "zonas problemáticas" = métricas con deterioro >10% WoW.
- Siempre en español.
- Para "última semana" usa L0W_ROLL/L0W. Para "semana anterior" L1W_ROLL/L1W.
- Si no aplica gráfico, usa "type": "none".
"""


def create_model(api_key: str) -> Groq:
    """Crea el cliente Groq."""
    return Groq(api_key=api_key)


def build_messages(
    conversation_history: list[dict],
    user_question: str,
    schema: str,
) -> list[dict]:
    """Construye el historial de mensajes para Groq (formato OpenAI)."""
    system = SYSTEM_PROMPT.format(schema=schema)
    messages = [{"role": "system", "content": system}]

    for msg in conversation_history[-10:]:
        messages.append(msg)

    messages.append({"role": "user", "content": user_question})
    return messages


def parse_llm_response(raw_text: str) -> dict:
    """Extrae el JSON de la respuesta del LLM."""
    # Intentar extraer JSON del bloque de código
    json_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", raw_text, re.DOTALL)
    if json_match:
        try:
            return json.loads(json_match.group(1))
        except json.JSONDecodeError:
            pass

    # Intentar encontrar JSON balanceado
    depth = 0
    start = None
    for i, ch in enumerate(raw_text):
        if ch == '{':
            if depth == 0:
                start = i
            depth += 1
        elif ch == '}':
            depth -= 1
            if depth == 0 and start is not None:
                try:
                    return json.loads(raw_text[start:i+1])
                except json.JSONDecodeError:
                    start = None

    # Fallback
    return {
        "thinking": "",
        "code": "",
        "explanation": raw_text,
        "chart": {"type": "none"},
        "suggestions": [],
    }


def clean_code(code: str) -> str:
    """Limpia el código generado por el LLM."""
    lines = []
    for line in code.split("\n"):
        stripped = line.strip()
        # Eliminar imports
        if stripped.startswith("import ") or stripped.startswith("from "):
            continue
        lines.append(line)
    return "\n".join(lines)


def execute_code(code: str, df_metrics: pd.DataFrame, df_orders: pd.DataFrame) -> tuple:
    """
    Ejecuta el código pandas generado por el LLM.
    Retorna (result, output_text, error).
    """
    if not code or not code.strip():
        return None, "", None

    import io
    import sys
    import numpy as np

    code = clean_code(code)

    old_stdout = sys.stdout
    sys.stdout = buffer = io.StringIO()

    # Entorno de ejecución con acceso completo a builtins de Python
    exec_globals = {
        "__builtins__": __builtins__,
        "df_metrics": df_metrics.copy(),
        "df_orders": df_orders.copy(),
        "pd": pd,
        "np": np,
    }

    try:
        exec(code, exec_globals)
        output = buffer.getvalue()
        result = exec_globals.get("result", None)
        return result, output, None
    except Exception as e:
        return None, "", f"{type(e).__name__}: {str(e)}"
    finally:
        sys.stdout = old_stdout


def clean_explanation(explanation: str, result) -> str:
    """Reemplaza referencias a variables Python en la explicación con valores reales."""
    if not explanation or not isinstance(explanation, str):
        return explanation or ""

    # Patrón: {result['COL'].values[0]} o {result['COL'].iloc[0]} etc.
    def replace_ref(match):
        expr = match.group(1)
        try:
            value = eval(expr, {"result": result, "pd": pd, "__builtins__": {}})
            if isinstance(value, float):
                return f"{value:.4f}".rstrip('0').rstrip('.')
            return str(value)
        except Exception:
            return ""

    # Reemplazar {result...} patterns
    cleaned = re.sub(r"\{(result\b[^}]*)\}", replace_ref, explanation)
    # Reemplazar {variable} patterns residuales (sin result)
    cleaned = re.sub(r"\{[a-zA-Z_][a-zA-Z0-9_]*(?:\[[^\]]*\])*(?:\.[a-zA-Z_]+(?:\([^)]*\))?)*\}", "", cleaned)
    # Limpiar espacios dobles
    cleaned = re.sub(r"  +", " ", cleaned).strip()
    return cleaned


def create_chart(result: pd.DataFrame, chart_config: dict) -> go.Figure | None:
    """Crea un gráfico Plotly basado en la configuración del LLM."""
    if not isinstance(result, pd.DataFrame) or result.empty:
        return None

    chart_type = chart_config.get("type", "none")
    if chart_type == "none" or not chart_type:
        return None

    title = chart_config.get("title", "")
    x = chart_config.get("x")
    y = chart_config.get("y")
    color = chart_config.get("color")
    orientation = chart_config.get("orientation", "v")

    if x and x not in result.columns:
        x = result.columns[0]
    if y and y not in result.columns:
        y = result.columns[-1] if len(result.columns) > 1 else result.columns[0]
    if color and color not in result.columns:
        color = None

    try:
        if chart_type == "bar":
            fig = px.bar(result, x=x, y=y, color=color, title=title,
                         orientation=orientation)
        elif chart_type == "line":
            fig = px.line(result, x=x, y=y, color=color, title=title)
        elif chart_type == "scatter":
            fig = px.scatter(result, x=x, y=y, color=color, title=title)
        elif chart_type == "heatmap":
            numeric_cols = result.select_dtypes(include="number").columns.tolist()
            if len(numeric_cols) >= 1:
                fig = px.imshow(result[numeric_cols], title=title,
                                aspect="auto", color_continuous_scale="RdYlGn")
            else:
                return None
        else:
            return None

        fig.update_layout(
            template="plotly_white",
            height=450,
            margin=dict(l=40, r=40, t=60, b=40),
            font=dict(size=12),
        )
        return fig
    except Exception:
        return None


def query(
    model: Groq,
    question: str,
    df_metrics: pd.DataFrame,
    df_orders: pd.DataFrame,
    conversation_history: list[dict],
) -> dict:
    """Pipeline completo: pregunta → LLM → código → ejecución → resultado."""
    schema = get_schema_description()
    messages = build_messages(conversation_history, question, schema)

    try:
        response = model.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=messages,
            temperature=0.1,
            max_tokens=4096,
        )
        raw_text = response.choices[0].message.content
    except Exception as e:
        return {
            "explanation": f"Error al conectar con Groq: {str(e)}",
            "result": None,
            "chart": None,
            "suggestions": [],
            "error": str(e),
            "code": "",
        }

    # Parsear
    try:
        parsed = parse_llm_response(raw_text)
    except Exception:
        parsed = {
            "thinking": "", "code": "", "explanation": raw_text,
            "chart": {"type": "none"}, "suggestions": [],
        }

    code = parsed.get("code", "")
    explanation = parsed.get("explanation", "")
    suggestions = parsed.get("suggestions", [])
    chart_config = parsed.get("chart", {"type": "none"})

    # Ejecutar código
    result = None
    output = ""
    error = None

    if code:
        result, output, error = execute_code(code, df_metrics, df_orders)

        # Retry si falla
        if error:
            retry_messages = messages + [
                {"role": "assistant", "content": raw_text},
                {"role": "user", "content": (
                    f"ERROR: {error}\n\n"
                    f"Código que falló:\n```python\n{code}\n```\n\n"
                    "RECUERDA: NO uses import. pd y np ya existen. "
                    "Solo UNA variable result. Responde SOLO con JSON."
                )},
            ]
            try:
                retry_resp = model.chat.completions.create(
                    model="llama-3.3-70b-versatile",
                    messages=retry_messages,
                    temperature=0.1,
                    max_tokens=4096,
                )
                retry_parsed = parse_llm_response(retry_resp.choices[0].message.content)
                retry_code = retry_parsed.get("code", "")
                if retry_code:
                    result, output, error = execute_code(retry_code, df_metrics, df_orders)
                    if not error:
                        code = retry_code
                        explanation = retry_parsed.get("explanation", explanation)
                        chart_config = retry_parsed.get("chart", chart_config)
                        suggestions = retry_parsed.get("suggestions", suggestions)
            except Exception:
                pass

    # Limpiar referencias a variables en la explicación
    explanation = clean_explanation(explanation, result)

    # Crear gráfico
    chart = None
    if isinstance(result, pd.DataFrame) and not result.empty:
        chart = create_chart(result, chart_config)

    return {
        "explanation": explanation,
        "result": result,
        "output": output,
        "chart": chart,
        "suggestions": suggestions,
        "error": error,
        "code": code,
    }