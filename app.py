import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go
import plotly.express as px
from io import BytesIO
import time
import re
from datetime import datetime

# ==========================================================
# COMPATIBILIDAD SCIKIT-LEARN
# ==========================================================
try:
    import sklearn.compose._column_transformer as ct

    if not hasattr(ct, "_RemainderColsList"):
        class _RemainderColsList(list):
            pass

        ct._RemainderColsList = _RemainderColsList

except Exception:
    pass


# ==========================================================
# CONFIGURACIÓN STREAMLIT
# ==========================================================
st.set_page_config(
    page_title="Sistema Predictivo Laboratorio Clínico (XGBoost)",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ==========================================================
# CSS PERSONALIZADO
# ==========================================================
st.markdown(
    """
    <style>
    .main {
        background-color: #0e1117;
    }
    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    .hero-card {
        background: linear-gradient(135deg, #0f2b3d 0%, #1a4a6f 45%, #2c6e9e 100%);
        padding: 35px;
        border-radius: 24px;
        text-align: center;
        margin-bottom: 30px;
        border: 1px solid rgba(255,255,255,0.09);
        box-shadow: 0 8px 28px rgba(0,0,0,0.35);
    }
    .hero-title {
        font-size: 42px;
        font-weight: 900;
        color: #ff6f91;
        margin-bottom: 10px;
        letter-spacing: 1px;
    }
    .hero-subtitle {
        font-size: 17px;
        color: #d8d8d8;
    }
    .section-title {
        font-size: 26px;
        font-weight: 800;
        color: #ffffff;
        margin-top: 25px;
        margin-bottom: 15px;
        border-left: 5px solid #ff6f91;
        padding-left: 15px;
    }
    .result-card {
        background: rgba(255,255,255,0.04);
        border: 1px solid rgba(255,255,255,0.10);
        border-radius: 20px;
        padding: 26px;
        text-align: center;
        box-shadow: 0 5px 20px rgba(0,0,0,0.28);
        min-height: 245px;
    }
    .prediction-label {
        font-size: 13px;
        color: #b9b9b9;
        letter-spacing: 2px;
        text-transform: uppercase;
    }
    .prediction-value {
        font-size: 42px;
        color: #4c83ff;
        font-weight: 900;
        margin-top: 10px;
    }
    .prediction-sub {
        font-size: 16px;
        color: #d0d0d0;
        margin-top: 10px;
    }
    .info-box-green {
        background: rgba(21, 128, 61, 0.20);
        border-left: 5px solid #22c55e;
        padding: 18px;
        border-radius: 12px;
        color: #d1fae5;
        margin-top: 15px;
        line-height: 1.6;
    }
    .info-box-yellow {
        background: rgba(202, 138, 4, 0.22);
        border-left: 5px solid #facc15;
        padding: 18px;
        border-radius: 12px;
        color: #fef9c3;
        margin-top: 15px;
        line-height: 1.6;
    }
    .info-box-red {
        background: rgba(185, 28, 28, 0.22);
        border-left: 5px solid #ef4444;
        padding: 18px;
        border-radius: 12px;
        color: #fee2e2;
        margin-top: 15px;
        line-height: 1.6;
    }
    .info-box-blue {
        background: rgba(37, 99, 235, 0.18);
        border-left: 5px solid #3b82f6;
        padding: 18px;
        border-radius: 12px;
        color: #dbeafe;
        margin-top: 15px;
        line-height: 1.6;
    }
    div[data-testid="stMetricValue"] {
        font-size: 30px;
    }
    div[data-testid="stMetricLabel"] {
        font-size: 14px;
    }
    </style>
    """,
    unsafe_allow_html=True
)


# ==========================================================
# ENCABEZADO
# ==========================================================
st.markdown(
    """
    <div class="hero-card">
        <div class="hero-title">🏥 Sistema Predictivo Laboratorio Clínico</div>
        <div class="hero-subtitle">
            Predicción de facturación con XGBoost | Análisis de rentabilidad | Dashboard interactivo
        </div>
    </div>
    """,
    unsafe_allow_html=True
)


# ==========================================================
# CARGAR MODELO (XGBoost desde PKL)
# ==========================================================
@st.cache_resource
def cargar_modelo():
    return joblib.load("modelo_predictor_total.pkl")

try:
    modelo = cargar_modelo()
    st.success("✅ Modelo XGBoost cargado correctamente.", icon="✅")
except FileNotFoundError:
    st.error(
        """
        ❌ No se encontró el archivo del modelo.
        
        Verifica que el archivo **modelo_predictor_total.pkl** esté en la misma carpeta que este script.
        """
    )
    st.stop()
except Exception as e:
    st.error("❌ Ocurrió un error al cargar el modelo.")
    st.exception(e)
    st.stop()


# ==========================================================
# COLUMNAS DEL MODELO
# ==========================================================
COLUMNAS_MODELO = [
    "Subtotal",
    "Precio",
    "mes",
    "año",
    "dia_semana",
    "Genero_num",
    "TipoFacturacion_num",
    "diferencia_precio",
    "ratio_subtotal_precio"
]

TARGET = "Total"


# ==========================================================
# FUNCIONES AUXILIARES
# ==========================================================
def preparar_datos(datos):
    datos = datos.copy()
    for col in COLUMNAS_MODELO:
        if col not in datos.columns:
            datos[col] = 0
    datos = datos[COLUMNAS_MODELO]
    for col in datos.columns:
        datos[col] = pd.to_numeric(datos[col], errors="coerce").fillna(0)
    return datos


def parse_multi_table_csv_corregido(uploaded_file):
    content = uploaded_file.getvalue().decode('utf-8')
    lines = content.split('\n')
    
    header_patterns = {
        '"IDCliente","Nombre","FechaNacimiento","Genero","Telefono"': "clientes",
        '"IDDetalleFactura","IDFactura","IDDetalleOrden","Subtotal","Precio","NombreParametro","Idparametro","IdtipoExamen"': "detalle_factura",
        '"IDDetalleOrden","IDOrden","IDTipoExamen","IDMuestra"': "detalle_orden",
        '"IDFactura","IDCliente","FechaFactura","Total","IDMedico","TipoFacturacion"': "facturas",
        '"IDMedico","Nombre","Especialidad","Password","Telefono","Rol","Correo","ResetToken","ResetTokenExpires","ContrasenaTemporal"': "medicos",
        '"id","Muestra"': "muestras",
        '"IDOrden","IDCliente","IDMedico","FechaOrden","Estado","fechaEntrega","NumeroMuestra"': "ordenes",
        '"IDParametro","IDTipoExamen","NombreParametro","Precio"': "parametros",
        '"IDResultado","IDDetalleOrden","IDParametro","Resultado","FechaResultado","NombreParametro"': "resultados",
        '"IDTipoExamen","NombreExamen","Precio"': "tipos_examen"
    }
    
    tables = {}
    i = 0
    
    while i < len(lines):
        line = lines[i].strip()
        
        for header_pattern, table_name in header_patterns.items():
            if line.startswith(header_pattern) or line == header_pattern:
                headers = [h.strip('"') for h in line.split(',')]
                data_lines = []
                i += 1
                
                while i < len(lines):
                    next_line = lines[i].strip()
                    if not next_line:
                        i += 1
                        continue
                    
                    is_new_header = False
                    for hp in header_patterns.keys():
                        if next_line.startswith(hp):
                            is_new_header = True
                            break
                    if is_new_header:
                        break
                    
                    values = []
                    in_quotes = False
                    current = []
                    for char in next_line:
                        if char == '"' and not in_quotes:
                            in_quotes = True
                        elif char == '"' and in_quotes:
                            in_quotes = False
                        elif char == ',' and not in_quotes:
                            values.append(''.join(current))
                            current = []
                        else:
                            current.append(char)
                    values.append(''.join(current))
                    
                    if len(values) == len(headers):
                        data_lines.append(values)
                    i += 1
                
                if data_lines:
                    tables[table_name] = pd.DataFrame(data_lines, columns=headers)
                break
        else:
            i += 1
    
    return tables


def preparar_datos_para_modelo_corregido(df_facturas, df_detalle_factura, df_clientes=None):
    """
    Prepara los datos combinando facturas y detalle_factura para el modelo.
    """
    if df_facturas is None or df_detalle_factura is None:
        return None
    
    # Hacer copias para no modificar originales
    facturas = df_facturas.copy()
    detalle = df_detalle_factura.copy()
    
    # Convertir a numérico
    facturas['Total'] = pd.to_numeric(facturas['Total'], errors='coerce').fillna(0)
    detalle['Subtotal'] = pd.to_numeric(detalle['Subtotal'], errors='coerce').fillna(0)
    detalle['Precio'] = pd.to_numeric(detalle['Precio'], errors='coerce').fillna(0)
    
    # Extraer fecha
    if 'FechaFactura' in facturas.columns:
        facturas['FechaFactura'] = pd.to_datetime(facturas['FechaFactura'], errors='coerce')
        facturas['mes'] = facturas['FechaFactura'].dt.month.fillna(1).astype(int)
        facturas['año'] = facturas['FechaFactura'].dt.year.fillna(2024).astype(int)
        facturas['dia_semana'] = facturas['FechaFactura'].dt.dayofweek.fillna(0).astype(int)
    else:
        facturas['mes'] = 1
        facturas['año'] = 2024
        facturas['dia_semana'] = 0
    
    # Mapear TipoFacturacion a número
    tipo_map = {'PARTICULAR': 1, 'EXAMEN': 2, 'PARAMETRO': 3}
    facturas['TipoFacturacion_num'] = facturas['TipoFacturacion'].map(tipo_map).fillna(1).astype(int)
    
    # Si tenemos clientes, mapear género
    if df_clientes is not None and 'IDCliente' in df_clientes.columns and 'Genero' in df_clientes.columns:
        clientes = df_clientes.copy()
        genero_map = {'FEMENINO': 0, 'MASCULINO': 1}
        clientes['Genero_num'] = clientes['Genero'].map(genero_map).fillna(0).astype(int)
        facturas = facturas.merge(clientes[['IDCliente', 'Genero_num']], on='IDCliente', how='left')
        facturas['Genero_num'] = facturas['Genero_num'].fillna(0).astype(int)
    else:
        facturas['Genero_num'] = 0
    
    # Agrupar detalle por factura para evitar duplicados
    detalle_agrupado = detalle.groupby('IDFactura').agg({
        'Subtotal': 'sum',
        'Precio': 'mean'
    }).reset_index()
    detalle_agrupado.columns = ['IDFactura', 'Subtotal', 'Precio']
    
    # Combinar
    resultado = facturas.merge(detalle_agrupado, on='IDFactura', how='left')
    
    # Rellenar valores nulos
    resultado['Subtotal'] = resultado['Subtotal'].fillna(0)
    resultado['Precio'] = resultado['Precio'].fillna(0)
    
    # Calcular features derivadas
    resultado['diferencia_precio'] = resultado['Subtotal'] - resultado['Precio']
    resultado['ratio_subtotal_precio'] = resultado['Subtotal'] / resultado['Precio'].replace(0, 1)
    resultado['ratio_subtotal_precio'] = resultado['ratio_subtotal_precio'].fillna(0)
    
    return resultado


def crear_fila_producto(subtotal, precio, mes, año, dia_semana, genero_num, tipo_facturacion_num, diferencia_precio, ratio_subtotal_precio):
    datos = pd.DataFrame({
        "Subtotal": [subtotal],
        "Precio": [precio],
        "mes": [mes],
        "año": [año],
        "dia_semana": [dia_semana],
        "Genero_num": [genero_num],
        "TipoFacturacion_num": [tipo_facturacion_num],
        "diferencia_precio": [diferencia_precio],
        "ratio_subtotal_precio": [ratio_subtotal_precio]
    })
    return datos


def calcular_indicadores(fila):
    subtotal = float(fila["Subtotal"]) if "Subtotal" in fila else 0
    costo_total = subtotal * 0.65
    ganancia = subtotal - costo_total
    margen = (ganancia / subtotal) * 100 if subtotal > 0 else 0
    return subtotal, costo_total, subtotal, ganancia, margen


def generar_estado_comercial(ganancia, margen):
    if ganancia > 0 and margen >= 30:
        return "Excelente", "Alta"
    elif ganancia > 0 and margen >= 15:
        return "Rentable", "Media-Alta"
    elif ganancia > 0 and margen < 15:
        return "Rentabilidad baja", "Media"
    elif ganancia == 0:
        return "Punto de equilibrio", "Media-Baja"
    else:
        return "Pérdida", "Baja"


def generar_explicacion(subtotal, ganancia, margen, resultado):
    if ganancia > 0 and margen >= 30:
        return f"""
        <div class="info-box-green">
        🎯 El modelo XGBoost estima un total de factura de <b>{resultado:.2f}</b>. 
        Con un subtotal de <b>{subtotal:.2f}</b> y un margen estimado de <b>{margen:.2f}%</b>, 
        la operación se considera <b>Excelente</b> y altamente rentable.
        </div>
        """
    elif ganancia > 0 and margen >= 15:
        return f"""
        <div class="info-box-green">
        ✅ El modelo XGBoost estima un total de factura de <b>{resultado:.2f}</b>. 
        La operación es <b>Rentable</b> con un margen del <b>{margen:.2f}%</b>.
        </div>
        """
    elif ganancia > 0:
        return f"""
        <div class="info-box-yellow">
        ⚠️ El modelo XGBoost estima un total de factura de <b>{resultado:.2f}</b>. 
        Aunque hay ganancia, el margen es bajo (<b>{margen:.2f}%</b>). 
        Se recomienda revisar la estructura de precios.
        </div>
        """
    elif ganancia == 0:
        return f"""
        <div class="info-box-yellow">
        📊 El modelo estima un total de factura de <b>{resultado:.2f}</b>. 
        La operación está en <b>punto de equilibrio</b>. Revise precios o costos.
        </div>
        """
    else:
        return f"""
        <div class="info-box-red">
        ❌ El modelo estima un total de factura de <b>{resultado:.2f}</b>. 
        La operación genera <b>pérdida</b>. Se requiere revisión inmediata.
        </div>
        """


def grafico_gauge(valor, titulo):
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number+delta",
            value=valor,
            delta={"reference": 50},
            title={"text": titulo},
            gauge={
                "axis": {"range": [0, 100]},
                "bar": {"color": "#22c55e"},
                "steps": [
                    {"range": [0, 30], "color": "#7f1d1d"},
                    {"range": [30, 60], "color": "#854d0e"},
                    {"range": [60, 100], "color": "#14532d"},
                ],
                "threshold": {
                    "line": {"color": "#ffffff", "width": 4},
                    "thickness": 0.75,
                    "value": valor,
                },
            },
        )
    )
    fig.update_layout(height=320, paper_bgcolor="rgba(0,0,0,0)", font={"color": "white"})
    return fig


def convertir_excel(df):
    output = BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name="Resultados")
    return output.getvalue()


def obtener_importancias_modelo(modelo):
    try:
        if hasattr(modelo, "get_booster") and hasattr(modelo.get_booster(), "get_score"):
            score = modelo.get_booster().get_score(importance_type='weight')
            if score:
                importancias = pd.DataFrame(list(score.items()), columns=['Variable', 'Importancia'])
                importancias['Importancia'] = importancias['Importancia'] / importancias['Importancia'].sum()
                return importancias.sort_values('Importancia', ascending=False)
        
        if hasattr(modelo, "feature_importances_"):
            importancias = modelo.feature_importances_
            nombres = COLUMNAS_MODELO
            if len(importancias) != len(nombres):
                nombres = [f"Variable_{i+1}" for i in range(len(importancias))]
            return pd.DataFrame({"Variable": nombres, "Importancia": importancias}).sort_values("Importancia", ascending=False)
    except Exception:
        pass
    return None


def grafico_boxplot_predicciones(df, columna="Prediccion_Modelo"):
    """Boxplot de predicciones por tipo de facturación"""
    if columna not in df.columns or 'TipoFacturacion' not in df.columns:
        return None
    
    fig = px.box(df, x='TipoFacturacion', y=columna, 
                 title="📦 Distribución de Predicciones por Tipo de Facturación",
                 color='TipoFacturacion',
                 color_discrete_sequence=px.colors.qualitative.Set2)
    fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font_color="white")
    return fig


def grafico_scatter_correlacion(df):
    """Gráfico de correlación entre Subtotal y Predicción (SIN trendline para evitar statsmodels)"""
    if 'Subtotal' not in df.columns or 'Prediccion_Modelo' not in df.columns:
        return None
    
    fig = px.scatter(df, x='Subtotal', y='Prediccion_Modelo', 
                     title="🔍 Correlación: Subtotal vs Predicción",
                     color_discrete_sequence=['#4c83ff'],
                     opacity=0.6)
    fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font_color="white")
    return fig


def grafico_violin_rentabilidad(df):
    """Gráfico de violín para rentabilidad por estado comercial"""
    if 'Estado_Comercial' not in df.columns or 'Ganancia_estimada' not in df.columns:
        return None
    
    fig = px.violin(df, x='Estado_Comercial', y='Ganancia_estimada', 
                    title="🎻 Distribución de Ganancias por Estado Comercial",
                    box=True, points="all",
                    color='Estado_Comercial',
                    color_discrete_sequence=px.colors.qualitative.Set1)
    fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font_color="white")
    return fig


def grafico_torta_estados(df):
    """Gráfico circular de distribución de estados comerciales"""
    if 'Estado_Comercial' not in df.columns:
        return None
    
    conteo = df['Estado_Comercial'].value_counts().reset_index()
    conteo.columns = ['Estado', 'Cantidad']
    
    fig = px.pie(conteo, values='Cantidad', names='Estado', 
                 title="🥧 Distribución de Estados Comerciales",
                 hole=0.4,
                 color_discrete_sequence=px.colors.qualitative.Set3)
    fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font_color="white")
    return fig


def grafico_barras_top_facturas(df, top_n=10):
    """Top N facturas por predicción"""
    if 'Prediccion_Modelo' not in df.columns or 'IDFactura' not in df.columns:
        return None
    
    top = df.nlargest(top_n, 'Prediccion_Modelo')[['IDFactura', 'Prediccion_Modelo', 'Estado_Comercial']]
    
    fig = px.bar(top, x='IDFactura', y='Prediccion_Modelo', 
                 title=f"🏆 Top {top_n} Facturas por Predicción",
                 color='Estado_Comercial',
                 text='Prediccion_Modelo',
                 color_discrete_sequence=px.colors.qualitative.Safe)
    fig.update_traces(texttemplate='$%{{text:.2f}}', textposition='outside')
    fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font_color="white",
                     xaxis_tickangle=-45)
    return fig


# ==========================================================
# PANEL DE CONTROL
# ==========================================================
st.sidebar.title("⚙️ Panel de Control")
modo = st.sidebar.radio(
    "Modo de trabajo",
    ["🔮 Predicción individual", "🎲 What If individual", "📁 Cargar CSV", "📊 Análisis del modelo"]
)
st.sidebar.divider()
st.sidebar.markdown(
    """
    ### ℹ️ Información
    **Modelo:** XGBoost Regressor  
    **Variables de entrada:** 9  
    **Objetivo:** Predicción de Total de factura
    """
)


# ==========================================================
# MODO 1: PREDICCIÓN INDIVIDUAL (CORREGIDO)
# ==========================================================
if modo == "🔮 Predicción individual":
    st.markdown('<div class="section-title">📝 Parámetros de Entrada</div>', unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    with col1:
        subtotal = st.number_input("💰 Subtotal (suma de precios)", min_value=0.0, value=150.0, step=10.0,
                                   help="Suma de los precios de los exámenes sin impuestos")
    with col2:
        precio = st.number_input("🏷️ Precio promedio por examen", min_value=0.0, value=75.0, step=5.0,
                                help="Precio promedio de cada examen en la factura")
    with col3:
        costo_porcentaje = st.slider("💰 Porcentaje de costo sobre subtotal", min_value=0, max_value=100, value=65, step=5,
                                     help="¿Qué porcentaje del subtotal representa el costo real?")
    
    col4, col5, col6 = st.columns(3)
    with col4:
        mes = st.selectbox("📅 Mes", options=list(range(1,13)), index=datetime.now().month-1)
    with col5:
        año = st.number_input("📆 Año", min_value=2020, value=2024, step=1)
    with col6:
        dia_semana = st.selectbox("📌 Día semana (0=Lunes,6=Domingo)", options=list(range(7)), index=0)
    
    col7, col8 = st.columns(2)
    with col7:
        genero_num = st.selectbox("👤 Género", options=[0,1], 
                                 format_func=lambda x: "Femenino (0)" if x==0 else "Masculino (1)")
    with col8:
        tipo_facturacion_num = st.selectbox("📄 Tipo Facturación", options=[1,2,3], 
                                            format_func=lambda x: {1:"PARTICULAR", 2:"EXAMEN", 3:"PARAMETRO"}.get(x, str(x)))
    
    # Mostrar información adicional
    st.info(f"💡 **Cálculo automático:** Diferencia precio = Subtotal - Precio = ${subtotal - precio:.2f} | "
            f"Ratio = Subtotal/Precio = {(subtotal/precio if precio>0 else 0):.2f}")
    
    diferencia_precio = subtotal - precio
    ratio_subtotal_precio = subtotal / precio if precio != 0 else 0
    
    if st.button("🔍 Predecir Operación", use_container_width=True):
        # Preparar datos para el modelo
        datos = crear_fila_producto(subtotal, precio, mes, año, dia_semana, genero_num, 
                                    tipo_facturacion_num, diferencia_precio, ratio_subtotal_precio)
        datos_modelo = preparar_datos(datos)
        
        try:
            with st.spinner("🔄 Procesando predicción..."):
                time.sleep(0.5)
                prediccion = modelo.predict(datos_modelo)
                total_factura = max(0, float(prediccion[0]))  # ✅ Este es el TOTAL predicho
            
            # ✅ CORREGIDO: Calcular correctamente costo y ganancia
            costo_real = subtotal * (costo_porcentaje / 100)  # Costo basado en el subtotal
            ganancia = total_factura - costo_real  # Ganancia = Total facturado - Costo real
            margen = (ganancia / total_factura) * 100 if total_factura > 0 else 0
            
            # Evaluar rentabilidad con criterios realistas
            if ganancia > 0 and margen >= 40:
                estado_comercial = "Excelente"
                nivel_rentabilidad = "Alta"
                color_box = "green"
                emoji = "🏆"
            elif ganancia > 0 and margen >= 25:
                estado_comercial = "Rentable"
                nivel_rentabilidad = "Media-Alta"
                color_box = "green"
                emoji = "✅"
            elif ganancia > 0 and margen >= 10:
                estado_comercial = "Aceptable"
                nivel_rentabilidad = "Media"
                color_box = "yellow"
                emoji = "⚠️"
            elif ganancia > 0 and margen > 0:
                estado_comercial = "Margen bajo"
                nivel_rentabilidad = "Media-Baja"
                color_box = "yellow"
                emoji = "📊"
            elif ganancia == 0:
                estado_comercial = "Punto de equilibrio"
                nivel_rentabilidad = "Baja"
                color_box = "yellow"
                emoji = "⚖️"
            else:
                estado_comercial = "Pérdida"
                nivel_rentabilidad = "Muy Baja"
                color_box = "red"
                emoji = "❌"
            
            confianza_visual = max(0, min(100, margen + 20))
            
            st.divider()
            st.markdown('<div class="section-title">📊 Resultado de la Predicción</div>', unsafe_allow_html=True)
            
            col_res1, col_res2 = st.columns([1,1])
            with col_res1:
                st.markdown(
                    f"""
                    <div class="result-card">
                        <div class="prediction-label">💵 Total Factura Estimado (Modelo XGBoost)</div>
                        <div class="prediction-value">${total_factura:,.2f}</div>
                        <div class="prediction-sub">Clasificación: <b>{estado_comercial}</b></div>
                        <div class="prediction-sub">Nivel Rentabilidad: <b>{nivel_rentabilidad}</b></div>
                        <div class="prediction-sub">Margen: <b>{margen:.1f}%</b></div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
            with col_res2:
                st.plotly_chart(grafico_gauge(confianza_visual, f"📈 Margen de Rentabilidad\n{margen:.1f}%"), 
                               use_container_width=True)
            
            # Mostrar métricas en columnas
            col_m1, col_m2, col_m3, col_m4 = st.columns(4)
            with col_m1:
                st.metric("💰 Subtotal", f"${subtotal:,.2f}", help="Suma de precios de exámenes")
            with col_m2:
                st.metric("💵 Total Factura (Predicho)", f"${total_factura:,.2f}", 
                         delta=f"${total_factura - subtotal:.2f}", help="Incluye impuestos")
            with col_m3:
                st.metric("🏥 Costo Real", f"${costo_real:,.2f}", 
                         help=f"{costo_porcentaje}% del subtotal")
            with col_m4:
                st.metric("📈 Ganancia Neta", f"${ganancia:,.2f}", 
                         delta=f"{margen:.1f}% margen", help="Total - Costo")
            
            # Explicación dinámica
            if estado_comercial == "Excelente":
                st.markdown(f"""
                <div class="info-box-green">
                {emoji} <b>¡Excelente operación!</b> El modelo XGBoost estima un total de factura de <b>${total_factura:,.2f}</b>.<br>
                Con un costo real de <b>${costo_real:,.2f}</b> ({costo_porcentaje}% del subtotal) y una ganancia de <b>${ganancia:,.2f}</b>, 
                el margen de rentabilidad es del <b>{margen:.1f}%</b>, considerado <b>Excelente</b> para el sector.<br>
                🎯 <b>Recomendación:</b> Mantener esta estructura de precios.
                </div>
                """, unsafe_allow_html=True)
            elif estado_comercial == "Rentable":
                st.markdown(f"""
                <div class="info-box-green">
                {emoji} <b>Operación Rentable</b> - El modelo predice <b>${total_factura:,.2f}</b> de facturación.<br>
                La ganancia estimada es de <b>${ganancia:,.2f}</b> con un margen del <b>{margen:.1f}%</b>, 
                considerado <b>Rentable</b> para el laboratorio.<br>
                📊 <b>Recomendación:</b> El negocio es viable, buscar optimizar costos para mejorar margen.
                </div>
                """, unsafe_allow_html=True)
            elif estado_comercial == "Aceptable":
                st.markdown(f"""
                <div class="info-box-yellow">
                {emoji} <b>Rentabilidad Aceptable</b> - Factura estimada: <b>${total_factura:,.2f}</b><br>
                Margen actual: <b>{margen:.1f}%</b> - Ganancia: <b>${ganancia:,.2f}</b><br>
                ⚠️ <b>Recomendación:</b> Revisar estructura de precios o reducir costos operativos.
                </div>
                """, unsafe_allow_html=True)
            elif estado_comercial == "Margen bajo":
                st.markdown(f"""
                <div class="info-box-yellow">
                {emoji} <b>Margen de Beneficio Bajo</b> - El modelo predice <b>${total_factura:,.2f}</b><br>
                El margen actual es de solo <b>{margen:.1f}%</b> con ganancia de <b>${ganancia:,.2f}</b><br>
                🔧 <b>Recomendación:</b> Analizar costos, aumentar precios o mejorar eficiencia operativa.
                </div>
                """, unsafe_allow_html=True)
            elif estado_comercial == "Punto de equilibrio":
                st.markdown(f"""
                <div class="info-box-yellow">
                {emoji} <b>Punto de Equilibrio</b> - Factura estimada: <b>${total_factura:,.2f}</b><br>
                Los costos igualan a los ingresos, no hay ganancia.<br>
                🚨 <b>Recomendación:</b> Urgente revisar precios y reducir costos para generar rentabilidad.
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="info-box-red">
                {emoji} <b>OPERACIÓN CON PÉRDIDA</b> - El modelo predice <b>${total_factura:,.2f}</b><br>
                Los costos (<b>${costo_real:,.2f}</b>) superan los ingresos, generando pérdida de <b>${abs(ganancia):,.2f}</b><br>
                🚨 <b>ACCIÓN INMEDIATA:</b> Revisar urgentemente precios, descuentos y estructura de costos.
                </div>
                """, unsafe_allow_html=True)
            
            with st.expander("🔧 Ver datos enviados al modelo XGBoost"):
                st.dataframe(datos_modelo, use_container_width=True)
                st.caption("El modelo utiliza estas 9 features para predecir el Total de factura")
                
        except Exception as e:
            st.error("❌ Ocurrió un error al realizar la predicción.")
            st.exception(e)


# ==========================================================
# MODO 2: WHAT IF INDIVIDUAL
# ==========================================================
elif modo == "🎲 What If individual":
    st.markdown('<div class="section-title">🎲 What If Individual Dinámico</div>', unsafe_allow_html=True)
    st.markdown(
        """
        <div class="info-box-blue">
        🔬 Este módulo simula múltiples escenarios variando subtotal y precio unitario.
        Útil para encontrar la combinación óptima de precios y maximizar ganancias.
        </div>
        """,
        unsafe_allow_html=True
    )
    
    st.subheader("📋 Datos base")
    col1, col2 = st.columns(2)
    with col1:
        subtotal_base = st.number_input("Subtotal base", min_value=0.0, value=150.0, step=10.0)
    with col2:
        precio_base = st.number_input("Precio unitario base", min_value=0.0, value=75.0, step=5.0)
    
    col3, col4, col5 = st.columns(3)
    with col3:
        mes_base = st.selectbox("Mes", options=list(range(1,13)), index=0)
    with col4:
        año_base = st.number_input("Año", min_value=2020, value=2024, step=1)
    with col5:
        dia_semana_base = st.selectbox("Día semana", options=list(range(7)), index=0)
    
    col6, col7 = st.columns(2)
    with col6:
        genero_num_base = st.selectbox("Género", options=[0,1], format_func=lambda x: "Femenino (0)" if x==0 else "Masculino (1)")
    with col7:
        tipo_facturacion_num_base = st.selectbox("Tipo Facturación", options=[1,2,3])
    
    st.subheader("🎯 Rangos de simulación")
    col8, col9, col10 = st.columns(3)
    with col8:
        rango_subtotal_min = st.number_input("Subtotal mínimo", min_value=0, value=50, step=10)
    with col9:
        rango_subtotal_max = st.number_input("Subtotal máximo", min_value=0, value=300, step=10)
    with col10:
        paso_subtotal = st.number_input("Paso subtotal", min_value=1, value=25, step=5)
    
    col11, col12, col13 = st.columns(3)
    with col11:
        rango_precio_min = st.number_input("Precio mínimo", min_value=0, value=30, step=10)
    with col12:
        rango_precio_max = st.number_input("Precio máximo", min_value=0, value=150, step=10)
    with col13:
        paso_precio = st.number_input("Paso precio", min_value=1, value=20, step=5)
    
    if rango_subtotal_max < rango_subtotal_min:
        st.error("❌ El subtotal máximo no puede ser menor que el mínimo.")
        st.stop()
    if rango_precio_max < rango_precio_min:
        st.error("❌ El precio máximo no puede ser menor que el mínimo.")
        st.stop()
    
    total_escenarios = ((rango_subtotal_max - rango_subtotal_min)//paso_subtotal + 1) * \
                       ((rango_precio_max - rango_precio_min)//paso_precio + 1)
    st.info(f"📊 Escenarios a generar: **{total_escenarios}**")
    
    if total_escenarios > 200:
        st.warning("⚠️ Más de 200 escenarios. Puede tomar unos segundos.")
    
    if st.button("🚀 Ejecutar What If", use_container_width=True):
        if total_escenarios == 0:
            st.error("❌ No se generaron escenarios.")
            st.stop()
        
        escenarios = []
        subtotales = list(range(rango_subtotal_min, rango_subtotal_max + 1, paso_subtotal))
        precios = list(range(rango_precio_min, rango_precio_max + 1, paso_precio))
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        total = len(subtotales) * len(precios)
        count = 0
        
        for sub in subtotales:
            for prec in precios:
                count += 1
                progress_bar.progress(count / total)
                status_text.text(f"Procesando escenario {count} de {total}")
                
                diferencia_precio = sub - prec
                ratio_subtotal_precio = sub / prec if prec != 0 else 0
                
                fila = crear_fila_producto(sub, prec, mes_base, año_base, dia_semana_base,
                                          genero_num_base, tipo_facturacion_num_base,
                                          diferencia_precio, ratio_subtotal_precio)
                
                try:
                    pred = max(0, float(modelo.predict(preparar_datos(fila))[0]))
                except:
                    pred = 0
                
                costo_total = sub * 0.65
                ganancia = sub - costo_total
                margen = (ganancia / sub) * 100 if sub > 0 else 0
                estado, _ = generar_estado_comercial(ganancia, margen)
                
                escenarios.append({
                    "Subtotal": sub,
                    "Precio": prec,
                    "diferencia_precio": diferencia_precio,
                    "ratio": f"{ratio_subtotal_precio:.2f}",
                    "Prediccion": pred,
                    "Ganancia": ganancia,
                    "Margen_%": round(margen, 2),
                    "Estado": estado
                })
        
        progress_bar.progress(1.0)
        status_text.success(f"✅ Simulación completada. {len(escenarios)} escenarios generados.")
        
        df_escenarios = pd.DataFrame(escenarios)
        
        st.divider()
        st.markdown('<div class="section-title">📈 Resultados del What If</div>', unsafe_allow_html=True)
        
        mejor = df_escenarios.loc[df_escenarios['Ganancia'].idxmax()]
        
        col_m1, col_m2, col_m3, col_m4 = st.columns(4)
        with col_m1:
            st.metric("🎯 Mejor ganancia", f"${mejor['Ganancia']:,.2f}")
        with col_m2:
            st.metric("📊 Ganancia promedio", f"${df_escenarios['Ganancia'].mean():,.2f}")
        with col_m3:
            st.metric("💵 Predicción promedio", f"${df_escenarios['Prediccion'].mean():,.2f}")
        with col_m4:
            st.metric("📈 Margen promedio", f"{df_escenarios['Margen_%'].mean():.1f}%")
        
        st.markdown(
            f"""
            <div class="info-box-green">
            🏆 <b>Mejor escenario:</b> Subtotal ${mejor['Subtotal']:,.0f} | Precio ${mejor['Precio']:,.0f} | 
            Ganancia ${mejor['Ganancia']:,.2f} | Margen {mejor['Margen_%']:.1f}%
            </div>
            """,
            unsafe_allow_html=True
        )
        
        # Heatmap
        pivot_ganancia = df_escenarios.pivot(index='Subtotal', columns='Precio', values='Ganancia')
        fig_heatmap = px.imshow(pivot_ganancia, labels=dict(x="Precio", y="Subtotal", color="Ganancia"),
                                title="🔥 Mapa de calor: Ganancia por combinación", color_continuous_scale="RdYlGn", aspect="auto")
        fig_heatmap.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font_color="white")
        st.plotly_chart(fig_heatmap, use_container_width=True)
        
        # Gráfico 3D
        fig_3d = px.scatter_3d(df_escenarios, x='Subtotal', y='Precio', z='Ganancia', color='Margen_%',
                               title="📊 Superficie de ganancias 3D", labels={'Margen_%': 'Margen (%)'},
                               color_continuous_scale="Viridis")
        fig_3d.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font_color="white")
        st.plotly_chart(fig_3d, use_container_width=True)
        
        st.dataframe(df_escenarios, use_container_width=True)
        
        csv_data = df_escenarios.to_csv(index=False).encode('utf-8')
        st.download_button("📥 Descargar CSV", csv_data, "what_if_resultados.csv", mime="text/csv", use_container_width=True)


# ==========================================================
# MODO 3: CARGAR CSV (CORREGIDO - SIN statsmodels)
# ==========================================================
elif modo == "📁 Cargar CSV":
    st.markdown('<div class="section-title">📁 Cargar Archivo CSV (Múltiples Tablas)</div>', unsafe_allow_html=True)
    st.markdown(
        """
        <div class="info-box-blue">
        📌 Este módulo procesa archivos CSV que contienen múltiples tablas 
        separadas por filas en blanco (como tu archivo laboratorioclinico_clean.csv).
        </div>
        """,
        unsafe_allow_html=True
    )
    
    archivo_csv = st.file_uploader("📂 Seleccione el archivo CSV", type=["csv"])
    
    if archivo_csv is not None:
        try:
            with st.spinner("🔄 Procesando archivo multi-tabla..."):
                tablas = parse_multi_table_csv_corregido(archivo_csv)
            
            st.success(f"✅ Archivo procesado correctamente. Se encontraron {len(tablas)} tablas.")
            
            # Mostrar tablas encontradas
            st.subheader("📋 Tablas encontradas")
            for nombre_tabla, df_tabla in tablas.items():
                with st.expander(f"📊 {nombre_tabla.upper()} - {len(df_tabla)} filas"):
                    st.dataframe(df_tabla.head(10), use_container_width=True)
            
            # Verificar tablas necesarias
            if 'facturas' in tablas and 'detalle_factura' in tablas:
                st.subheader("🔮 Predicción con datos reales del laboratorio")
                
                df_combinado = preparar_datos_para_modelo_corregido(
                    tablas['facturas'], 
                    tablas['detalle_factura'],
                    tablas.get('clientes')
                )
                
                if df_combinado is not None and len(df_combinado) > 0:
                    # Preparar para modelo
                    df_para_modelo = preparar_datos(df_combinado)
                    
                    with st.spinner("🔄 Generando predicciones..."):
                        predicciones = []
                        for idx, row in df_para_modelo.iterrows():
                            try:
                                pred = float(modelo.predict(pd.DataFrame([row]))[0])
                                predicciones.append(max(0, pred))
                            except:
                                predicciones.append(0)
                    
                    df_combinado['Prediccion_Modelo'] = predicciones
                    
                    # Calcular indicadores
                    df_combinado['Ganancia_estimada'] = df_combinado['Subtotal'] * 0.35
                    df_combinado['Margen_estimado'] = (df_combinado['Ganancia_estimada'] / df_combinado['Subtotal'].replace(0, 1)) * 100
                    
                    estados = df_combinado.apply(lambda row: generar_estado_comercial(row["Ganancia_estimada"], row["Margen_estimado"])[0], axis=1)
                    df_combinado['Estado_Comercial'] = estados
                    
                    # Agregar TipoFacturacion original para gráficos
                    if 'TipoFacturacion' in tablas['facturas'].columns:
                        tipo_original = tablas['facturas'][['IDFactura', 'TipoFacturacion']].copy()
                        df_combinado = df_combinado.merge(tipo_original, on='IDFactura', how='left')
                    
                    st.divider()
                    st.markdown('<div class="section-title">📊 Dashboard de Resultados</div>', unsafe_allow_html=True)
                    
                    # Métricas principales
                    col_m1, col_m2, col_m3, col_m4 = st.columns(4)
                    with col_m1:
                        st.metric("📋 Facturas", len(df_combinado))
                    with col_m2:
                        st.metric("💰 Total facturado", f"${df_combinado['Total'].sum():,.2f}")
                    with col_m3:
                        st.metric("📈 Total predicción", f"${df_combinado['Prediccion_Modelo'].sum():,.2f}")
                    with col_m4:
                        st.metric("💵 Ganancia total", f"${df_combinado['Ganancia_estimada'].sum():,.2f}")
                    
                    # FILA 1: Gráficos principales
                    col_g1, col_g2 = st.columns(2)
                    with col_g1:
                        # Histograma de predicciones
                        fig_hist = px.histogram(df_combinado, x="Prediccion_Modelo", nbins=30,
                                               title="📊 Distribución de Predicciones",
                                               color_discrete_sequence=['#4c83ff'])
                        fig_hist.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font_color="white")
                        st.plotly_chart(fig_hist, use_container_width=True)
                    
                    with col_g2:
                        # Boxplot por tipo de facturación
                        fig_box = grafico_boxplot_predicciones(df_combinado)
                        if fig_box:
                            st.plotly_chart(fig_box, use_container_width=True)
                    
                    # FILA 2: Gráficos de correlación y violín
                    col_g3, col_g4 = st.columns(2)
                    with col_g3:
                        fig_scatter = grafico_scatter_correlacion(df_combinado)
                        if fig_scatter:
                            st.plotly_chart(fig_scatter, use_container_width=True)
                    
                    with col_g4:
                        fig_violin = grafico_violin_rentabilidad(df_combinado)
                        if fig_violin:
                            st.plotly_chart(fig_violin, use_container_width=True)
                    
                    # FILA 3: Torta y top facturas
                    col_g5, col_g6 = st.columns(2)
                    with col_g5:
                        fig_pie = grafico_torta_estados(df_combinado)
                        if fig_pie:
                            st.plotly_chart(fig_pie, use_container_width=True)
                    
                    with col_g6:
                        fig_top = grafico_barras_top_facturas(df_combinado, top_n=10)
                        if fig_top:
                            st.plotly_chart(fig_top, use_container_width=True)
                    
                    # Tendencia mensual
                    if 'mes' in df_combinado.columns:
                        tendencia = df_combinado.groupby('mes')['Prediccion_Modelo'].agg(['mean', 'sum']).reset_index()
                        fig_tend = go.Figure()
                        fig_tend.add_trace(go.Scatter(x=tendencia['mes'], y=tendencia['mean'], mode='lines+markers',
                                                      name='Promedio', line=dict(color='#4c83ff', width=3)))
                        fig_tend.add_trace(go.Bar(x=tendencia['mes'], y=tendencia['sum'], name='Total', yaxis='y2',
                                                  marker_color='rgba(76, 131, 255, 0.3)'))
                        fig_tend.update_layout(title="📈 Tendencia Mensual de Facturación",
                                              xaxis=dict(title="Mes", tick0=1, dtick=1),
                                              yaxis=dict(title="Promedio"), 
                                              yaxis2=dict(title="Total", overlaying='y', side='right'),
                                              paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font_color="white")
                        st.plotly_chart(fig_tend, use_container_width=True)
                    
                    # Tabla de resultados
                    st.subheader("📋 Tabla detallada de resultados")
                    columnas_mostrar = ['IDFactura', 'Total', 'Subtotal', 'Precio', 'Prediccion_Modelo', 
                                       'Ganancia_estimada', 'Margen_estimado', 'Estado_Comercial', 'TipoFacturacion']
                    columnas_existentes = [col for col in columnas_mostrar if col in df_combinado.columns]
                    st.dataframe(df_combinado[columnas_existentes], use_container_width=True)
                    
                    # Descargas
                    csv_resultados = df_combinado.to_csv(index=False).encode('utf-8')
                    excel_resultados = convertir_excel(df_combinado)
                    
                    col_d1, col_d2 = st.columns(2)
                    with col_d1:
                        st.download_button("📥 Descargar CSV", csv_resultados, "resultados_prediccion.csv", 
                                          mime="text/csv", use_container_width=True)
                    with col_d2:
                        st.download_button("📥 Descargar Excel", excel_resultados, "resultados_prediccion.xlsx", 
                                          use_container_width=True)
                else:
                    st.warning("⚠️ No se pudieron combinar los datos de facturas y detalle_factura.")
            else:
                st.warning("⚠️ El archivo no contiene las tablas necesarias para predicción ('facturas' y 'detalle_factura').")
                st.info("Las tablas encontradas son: " + ", ".join(tablas.keys()))
                
        except Exception as e:
            st.error("❌ Error al procesar el archivo.")
            st.exception(e)
            st.info("💡 Asegúrate de que el archivo tenga el formato correcto (múltiples tablas separadas por filas en blanco).")


# ==========================================================
# MODO 4: ANÁLISIS DEL MODELO
# ==========================================================
else:
    st.markdown('<div class="section-title">📊 Análisis General del Modelo</div>', unsafe_allow_html=True)
    
    st.markdown(
        """
        <div class="info-box-blue">
        🤖 <b>XGBoost Regressor</b> - Modelo de Gradient Boosting optimizado para regresión.
        Combina múltiples árboles de decisión de forma secuencial, corrigiendo errores 
        de los anteriores para lograr alta precisión predictiva.
        </div>
        """,
        unsafe_allow_html=True
    )
    
    col_info1, col_info2 = st.columns(2)
    with col_info1:
        st.subheader("📋 Información del modelo")
        st.write("**Tipo:**", str(type(modelo)))
        st.write("**Variables de entrada:**", len(COLUMNAS_MODELO))
        st.write("**Variable objetivo:**", TARGET)
        
    with col_info2:
        st.subheader("🎯 Características del problema")
        st.write("**Tipo:** Regresión supervisada")
        st.write("**Función de pérdida:** reg:squarederror")
        st.write("**Algoritmo:** XGBoost")
    
    st.subheader("📊 Importancia de variables")
    
    importancias = obtener_importancias_modelo(modelo)
    if importancias is not None:
        fig_imp = px.bar(importancias.head(10), x='Importancia', y='Variable', orientation='h',
                        title="Top 10 - Importancia de características", color='Importancia', color_continuous_scale='Viridis')
        fig_imp.update_layout(yaxis={'categoryorder': 'total ascending'}, paper_bgcolor="rgba(0,0,0,0)",
                              plot_bgcolor="rgba(0,0,0,0)", font_color="white", height=500)
        st.plotly_chart(fig_imp, use_container_width=True)
        
        st.markdown(f"""
        <div class="info-box-green">
        💡 <b>Variable más importante:</b> {importancias.iloc[0]['Variable']} - 
        Esta característica tiene el mayor impacto en la predicción del Total.
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("<div class='info-box-yellow'>⚠️ No se pudieron extraer las importancias del modelo.</div>", unsafe_allow_html=True)
    
    st.subheader("📋 Descripción de variables")
    
    df_descripcion = pd.DataFrame({
        "Variable": COLUMNAS_MODELO,
        "Tipo": ["Numérica"] * len(COLUMNAS_MODELO),
        "Descripción": [
            "💰 Valor total de la venta antes de impuestos o descuentos",
            "🏷️ Precio unitario del producto/servicio",
            "📅 Mes de la transacción (1-12)",
            "📆 Año de la transacción",
            "📌 Día de la semana (0=Lunes a 6=Domingo)",
            "👤 Género del cliente codificado (0=Femenino, 1=Masculino)",
            "📄 Tipo de facturación (1=Contado, 2=Crédito, 3=Mixto)",
            "📊 Diferencia entre subtotal y precio (Subtotal - Precio)",
            "📈 Relación entre subtotal y precio (Subtotal / Precio)"
        ],
        "Rango sugerido": [">0", ">0", "1-12", "2020-2024", "0-6", "0 o 1", "1, 2 o 3", "Puede ser negativo", ">0"]
    })
    st.dataframe(df_descripcion, use_container_width=True)
    
    st.subheader("🎯 Recomendaciones de uso")
    st.markdown(
        """
        <div class="info-box-green">
        ✅ <b>Mejores prácticas:</b>
        <ul>
            <li>Usar datos de facturación reales para mayor precisión</li>
            <li>Incluir todas las variables requeridas para el modelo</li>
            <li>Validar que los rangos de valores sean consistentes</li>
            <li>Actualizar el modelo periódicamente con nuevos datos</li>
        </ul>
        </div>
        """,
        unsafe_allow_html=True
    )