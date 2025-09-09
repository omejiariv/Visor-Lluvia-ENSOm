# --- Importaciones ---
import streamlit as st
import pandas as pd
import altair as alt
import folium
from folium.plugins import MarkerCluster, MiniMap
from folium.raster_layers import WmsTileLayer
from streamlit_folium import folium_static
import plotly.express as px
import plotly.graph_objects as go
import geopandas as gpd
import zipfile
import tempfile
import os
import io
import numpy as np
from pykrige.ok import OrdinaryKriging
import locale
import base64
from scipy import stats
import statsmodels.api as sm

# --- Funciones de Carga y Procesamiento ---
def parse_spanish_dates(date_series):
    """Convierte abreviaturas de meses en español a inglés."""
    months_es_to_en = {
        'ene': 'Jan', 'abr': 'Apr', 'ago': 'Aug', 'dic': 'Dec'
    }
    for es, en in months_es_to_en.items():
        date_series = date_series.str.replace(es, en, regex=False, case=False)
    return date_series

@st.cache_data
def load_data(file_path, sep=';', date_cols=None, lower_case=True):
    """Carga y decodifica un archivo CSV de manera robusta."""
    if file_path is None:
        return None
    try:
        content = file_path.getvalue()
        if not content.strip():
            st.error("El archivo parece estar vacío.")
            return None
    except Exception as e:
        st.error(f"Error al leer el archivo: {e}")
        return None

    encodings_to_try = ['utf-8', 'latin1', 'cp1252', 'iso-8859-1']
    for encoding in encodings_to_try:
        try:
            df = pd.read_csv(io.BytesIO(content), sep=sep, encoding=encoding, parse_dates=date_cols)
            df.columns = df.columns.str.strip().str.replace(';', '')
            if lower_case:
                df.columns = df.columns.str.lower()
            return df
        except Exception:
            continue
    st.error("No se pudo decodificar el archivo.")
    return None

@st.cache_data
def load_shapefile(file_path):
    """Procesa y carga un shapefile desde un archivo .zip."""
    try:
        with tempfile.TemporaryDirectory() as temp_dir:
            with zipfile.ZipFile(file_path, 'r') as zip_ref:
                zip_ref.extractall(temp_dir)
            shp_files = [f for f in os.listdir(temp_dir) if f.endswith('.shp')]
            if not shp_files:
                st.error("No se encontró un archivo .shp en el archivo .zip.")
                return None
            shp_path = os.path.join(temp_dir, shp_files[0])
            gdf = gpd.read_file(shp_path)
            gdf.columns = gdf.columns.str.strip().str.lower()
            if gdf.crs is None:
                gdf.set_crs("EPSG:9377", inplace=True)
            return gdf.to_crs("EPSG:4326")
    except Exception as e:
        st.error(f"Error al procesar el shapefile: {e}")
        return None

@st.cache_data
def complete_series(_df):
    """Completa las series de tiempo de precipitación usando interpolación."""
    all_completed_dfs = []
    station_list = _df['nom_est'].unique()
    progress_bar = st.progress(0, text="Completando todas las series...")
    for i, station in enumerate(station_list):
        df_station = _df[_df['nom_est'] == station].copy()
        df_station['fecha_mes_año'] = pd.to_datetime(df_station['fecha_mes_año'], format='%b-%y', errors='coerce')
        df_station.set_index('fecha_mes_año', inplace=True)

        if not df_station.index.is_unique:
            df_station = df_station[~df_station.index.duplicated(keep='first')]

        original_data = df_station[['precipitation', 'origen']].copy()
        df_resampled = df_station.resample('MS').asfreq()
        df_resampled['precipitation'] = original_data['precipitation']
        df_resampled['origen'] = original_data['origen']
        df_resampled['anomalia_oni'] = df_station['anomalia_oni']
        df_resampled['origen'] = df_resampled['origen'].fillna('Completado')
        df_resampled['precipitation'] = df_resampled['precipitation'].interpolate(method='time')

        df_resampled['nom_est'] = station
        df_resampled['año'] = df_resampled.index.year
        df_resampled['mes'] = df_resampled.index.month
        df_resampled.reset_index(inplace=True)
        all_completed_dfs.append(df_resampled)
        progress_bar.progress((i + 1) / len(station_list), text=f"Completando series... Estación: {station}")
    progress_bar.empty()
    return pd.concat(all_completed_dfs, ignore_index=True)

@st.cache_data
def preprocess_data(uploaded_file_mapa, uploaded_file_precip, uploaded_zip_shapefile):
    """Preprocesa todos los archivos de entrada."""
    df_precip_anual = load_data(uploaded_file_mapa)
    df_precip_mensual_raw = load_data(uploaded_file_precip)
    gdf_municipios = load_shapefile(uploaded_zip_shapefile)

    if any(df is None for df in [df_precip_anual, df_precip_mensual_raw, gdf_municipios]):
        return None, None, None, None, None

    # Preprocesamiento de datos anuales y de estaciones
    lon_col = next((col for col in df_precip_anual.columns if 'longitud' in col.lower() or 'lon' in col.lower()), None)
    lat_col = next((col for col in df_precip_anual.columns if 'latitud' in col.lower() or 'lat' in col.lower()), None)
    if not all([lon_col, lat_col]):
        st.error("No se encontraron las columnas de longitud y/o latitud en el archivo de estaciones.")
        return None, None, None, None, None
    df_precip_anual[lon_col] = pd.to_numeric(df_precip_anual[lon_col].astype(str).str.replace(',', '.'), errors='coerce')
    df_precip_anual[lat_col] = pd.to_numeric(df_precip_anual[lat_col].astype(str).str.replace(',', '.'), errors='coerce')
    if 'alt_est' in df_precip_anual.columns:
        df_precip_anual['alt_est'] = pd.to_numeric(df_precip_anual['alt_est'].astype(str).str.replace(',', '.'), errors='coerce')
    df_precip_anual.dropna(subset=[lon_col, lat_col], inplace=True)
    gdf_temp = gpd.GeoDataFrame(df_precip_anual, geometry=gpd.points_from_xy(df_precip_anual[lon_col], df_precip_anual[lat_col]), crs="EPSG:9377")
    gdf_stations = gdf_temp.to_crs("EPSG:4326")
    gdf_stations['longitud_geo'] = gdf_stations.geometry.x
    gdf_stations['latitud_geo'] = gdf_stations.geometry.y

    # Preprocesamiento de datos mensuales y ENSO
    df_precip_mensual = df_precip_mensual_raw.copy()
    station_cols = [col for col in df_precip_mensual.columns if col.isdigit()]
    if not station_cols:
        st.error("No se encontraron columnas de estación (ej: '12345') en el archivo de precipitación mensual.")
        return None, None, None, None, None

    id_vars_base = ['id', 'fecha_mes_año', 'año', 'mes', 'enso_año', 'enso_mes']
    id_vars_enso = ['anomalia_oni', 'temp_sst', 'temp_media']
    id_vars = id_vars_base + id_vars_enso
    
    for col in id_vars_enso:
        if col in df_precip_mensual.columns:
            df_precip_mensual[col] = df_precip_mensual[col].astype(str).str.replace(',', '.')

    df_long = df_precip_mensual.melt(id_vars=[col for col in id_vars if col in df_precip_mensual.columns],
                                     value_vars=station_cols, var_name='id_estacion', value_name='precipitation')

    df_long['precipitation'] = pd.to_numeric(df_long['precipitation'].astype(str).str.replace(',', '.'), errors='coerce')
    for col in id_vars_enso:
        if col in df_long.columns:
            df_long[col] = pd.to_numeric(df_long[col], errors='coerce')
    
    df_long.dropna(subset=['precipitation'], inplace=True)
    df_long['fecha_mes_año'] = parse_spanish_dates(df_long['fecha_mes_año'])
    df_long['fecha_mes_año'] = pd.to_datetime(df_long['fecha_mes_año'], format='%b-%y', errors='coerce')
    
    df_long.dropna(subset=['fecha_mes_año'], inplace=True)
    df_long['origen'] = 'Original'

    gdf_stations['id_estacio'] = gdf_stations['id_estacio'].astype(str).str.strip()
    df_long['id_estacion'] = df_long['id_estacion'].astype(str).str.strip()
    station_mapping = gdf_stations.set_index('id_estacio')['nom_est'].to_dict()
    df_long['nom_est'] = df_long['id_estacion'].map(station_mapping)
    df_long.dropna(subset=['nom_est'], inplace=True)

    enso_cols = ['id', 'fecha_mes_año', 'anomalia_oni', 'temp_sst', 'temp_media']
    df_enso = df_precip_mensual[enso_cols].drop_duplicates().copy()
    for col in ['anomalia_oni', 'temp_sst', 'temp_media']:
        if col in df_enso.columns:
            df_enso[col] = pd.to_numeric(df_enso[col], errors='coerce')
    
    df_enso['fecha_mes_año'] = parse_spanish_dates(df_enso['fecha_mes_año'])
    df_enso['fecha_mes_año'] = pd.to_datetime(df_enso['fecha_mes_año'], format='%b-%y', errors='coerce')
    
    df_enso.dropna(subset=['fecha_mes_año'], inplace=True)

    return gdf_stations, df_precip_anual, gdf_municipios, df_long, df_enso

def create_enso_chart(enso_data):
    if enso_data.empty or 'anomalia_oni' not in enso_data.columns:
        return go.Figure()

    data = enso_data.copy().sort_values('fecha_mes_año')
    data.dropna(subset=['anomalia_oni'], inplace=True)

    conditions = [data['anomalia_oni'] >= 0.5, data['anomalia_oni'] <= -0.5]
    phases = ['El Niño', 'La Niña']
    colors = ['red', 'blue']
    data['phase'] = np.select(conditions, phases, default='Neutral')
    data['color'] = np.select(conditions, colors, default='grey')

    y_range = [data['anomalia_oni'].min() - 0.5, data['anomalia_oni'].max() + 0.5]

    fig = go.Figure()

    fig.add_trace(go.Bar(
        x=data['fecha_mes_año'],
        y=[y_range[1] - y_range[0]] * len(data),
        base=y_range[0],
        marker_color=data['color'],
        width=30*24*60*60*1000,
        opacity=0.3,
        hoverinfo='none',
        showlegend=False
    ))

    legend_map = {'El Niño': 'red', 'La Niña': 'blue', 'Neutral': 'grey'}
    for phase, color in legend_map.items():
        fig.add_trace(go.Scatter(
            x=[None], y=[None], mode='markers',
            marker=dict(size=15, color=color, symbol='square', opacity=0.5),
            name=phase, showlegend=True
        ))

    fig.add_trace(go.Scatter(
        x=data['fecha_mes_año'], y=data['anomalia_oni'],
        mode='lines',
        name='Anomalía ONI',
        line=dict(color='black', width=2),
        showlegend=True
    ))

    fig.add_hline(y=0.5, line_dash="dash", line_color="red")
    fig.add_hline(y=-0.5, line_dash="dash", line_color="blue")

    fig.update_layout(
        height=600,
        title="Fases del Fenómeno ENSO y Anomalía ONI",
        yaxis_title="Anomalía ONI (°C)",
        xaxis_title="Fecha",
        showlegend=True,
        legend_title_text='Fase',
        yaxis_range=y_range
    )
    return fig

def create_anomaly_chart(df_plot):
    if df_plot.empty:
        return go.Figure()

    df_plot['color'] = np.where(df_plot['anomalia'] < 0, 'red', 'blue')
    
    fig = go.Figure()

    # Añadir barras de anomalía
    fig.add_trace(go.Bar(
        x=df_plot['fecha_mes_año'],
        y=df_plot['anomalia'],
        marker_color=df_plot['color'],
        name='Anomalía de Precipitación'
    ))

    # Añadir sombreado para fases ENSO
    y_min, y_max = df_plot['anomalia'].min(), df_plot['anomalia'].max()
    
    df_plot_enso = df_plot.dropna(subset=['anomalia_oni'])
    
    nino_periods = df_plot_enso[df_plot_enso['anomalia_oni'] >= 0.5]
    for _, row in nino_periods.iterrows():
        fig.add_vrect(x0=row['fecha_mes_año'] - pd.DateOffset(days=15), x1=row['fecha_mes_año'] + pd.DateOffset(days=15), 
                      fillcolor="red", opacity=0.15, layer="below", line_width=0)

    nina_periods = df_plot_enso[df_plot_enso['anomalia_oni'] <= -0.5]
    for _, row in nina_periods.iterrows():
        fig.add_vrect(x0=row['fecha_mes_año'] - pd.DateOffset(days=15), x1=row['fecha_mes_año'] + pd.DateOffset(days=15), 
                      fillcolor="blue", opacity=0.15, layer="below", line_width=0)

    # Añadir elementos de leyenda para el sombreado
    fig.add_trace(go.Scatter(x=[None], y=[None], mode='markers', marker=dict(symbol='square', color='rgba(255, 0, 0, 0.3)'), name='Fase El Niño'))
    fig.add_trace(go.Scatter(x=[None], y=[None], mode='markers', marker=dict(symbol='square', color='rgba(0, 0, 255, 0.3)'), name='Fase La Niña'))
    
    fig.update_layout(
        height=600,
        title="Anomalías Mensuales de Precipitación y Fases ENSO",
        yaxis_title="Anomalía de Precipitación (mm)",
        xaxis_title="Fecha",
        showlegend=True
    )
    return fig

# --- Configuración de la página y CSS ---
st.set_page_config(layout="wide", page_title="Sistema de información de las lluvias y el Clima en el norte de la región Andina")

st.markdown("""
<style>
div.block-container {padding-top: 2rem;}
.sidebar .sidebar-content {font-size: 13px; }
[data-testid="stMetricValue"] {
    font-size: 1.8rem;
}
[data-testid="stMetricLabel"] {
    font-size: 1rem;
    padding-bottom: 5px;
}
button[data-baseweb="tab"] {
    font-size: 16px;
    font-weight: bold;
    color: #333;
}
</style>
""", unsafe_allow_html=True)

# --- Interfaz de Carga de Archivos ---
logo_path = "CuencaVerdeLogo_V1.JPG"
logo_gota_path = "CuencaVerdeGoticaLogo.JPG"

title_col1, title_col2 = st.columns([0.07, 0.93])
with title_col1:
    if os.path.exists(logo_gota_path):
        st.image(logo_gota_path, width=50)
with title_col2:
    st.markdown('<h1 style="font-size:28px; margin-top:1rem;">Sistema de información de las lluvias y el Clima en el norte de la región Andina</h1>', unsafe_allow_html=True)

st.sidebar.header("Panel de Control")

# --- Lógica de carga de datos persistente y recarga ---
with st.sidebar.expander("**Cargar Archivos**", expanded=True):
    uploaded_file_mapa = st.file_uploader("1. Cargar archivo de estaciones (mapaCVENSO.csv)", type="csv")
    uploaded_file_precip = st.file_uploader("2. Cargar archivo de precipitación mensual y ENSO (DatosPptnmes_ENSO.csv)", type="csv")
    uploaded_zip_shapefile = st.file_uploader("3. Cargar shapefile de municipios (.zip)", type="zip")
    
    if st.button("Recargar Datos"):
        st.session_state.data_loaded = False
        st.cache_data.clear()
        st.rerun()

# Si no hay datos en la sesión, forzar la carga inicial
if 'data_loaded' not in st.session_state or not st.session_state.data_loaded:
    if not all([uploaded_file_mapa, uploaded_file_precip, uploaded_zip_shapefile]):
        st.info("Por favor, suba los 3 archivos requeridos para habilitar la aplicación. Una vez cargados, la plataforma se iniciará automáticamente y no necesitará volver a cargarlos.")
        st.stop()
    else:
        with st.spinner("Procesando archivos y cargando datos... Esto puede tomar un momento."):
            st.session_state.gdf_stations, st.session_state.df_precip_anual, st.session_state.gdf_municipios, st.session_state.df_long, st.session_state.df_enso = preprocess_data(uploaded_file_mapa, uploaded_file_precip, uploaded_zip_shapefile)
        if st.session_state.gdf_stations is not None:
            st.session_state.data_loaded = True
            st.rerun()
        else:
            st.error("Hubo un error al procesar los archivos. Por favor, verifique que los archivos sean correctos y vuelva a intentarlo.")
            st.stop()
            
# Si los datos no se cargaron, detener la ejecución de la app
if st.session_state.gdf_stations is None:
    st.stop()

# Carga de datos de la sesión
gdf_stations = st.session_state.gdf_stations
df_precip_anual = st.session_state.df_precip_anual
df_long = st.session_state.df_long
df_enso = st.session_state.df_enso
gdf_municipios = st.session_state.gdf_municipios


# --- LÓGICA DE FILTRADO OPTIMIZADA Y DINÁMICA ---
@st.cache_data
def apply_filters_to_stations(df, min_data_perc, selected_altitudes, selected_regions, selected_municipios, selected_celdas):
    """Aplica los filtros geográficos y de datos para obtener la lista de estaciones disponibles."""
    stations_filtered = df.copy()
    if 'porc_datos' in stations_filtered.columns:
        stations_filtered['porc_datos'] = pd.to_numeric(stations_filtered['porc_datos'].astype(str).str.replace(',', '.'), errors='coerce').fillna(0)
        stations_filtered = stations_filtered[stations_filtered['porc_datos'] >= min_data_perc]
    
    if selected_altitudes:
        conditions = []
        for r in selected_altitudes:
            if r == '0-500': conditions.append((stations_filtered['alt_est'] >= 0) & (stations_filtered['alt_est'] <= 500))
            elif r == '500-1000': conditions.append((stations_filtered['alt_est'] > 500) & (stations_filtered['alt_est'] <= 1000))
            elif r == '1000-2000': conditions.append((stations_filtered['alt_est'] > 1000) & (stations_filtered['alt_est'] <= 2000))
            elif r == '2000-3000': conditions.append((stations_filtered['alt_est'] > 2000) & (stations_filtered['alt_est'] <= 3000))
            elif r == '>3000': conditions.append(stations_filtered['alt_est'] > 3000)
        if conditions:
            combined_condition = pd.concat(conditions, axis=1).any(axis=1)
            stations_filtered = stations_filtered[combined_condition]

    if selected_regions:
        stations_filtered = stations_filtered[stations_filtered['depto_region'].isin(selected_regions)]
    
    if selected_municipios:
        stations_filtered = stations_filtered[stations_filtered['municipio'].isin(selected_municipios)]
    
    if selected_celdas:
        stations_filtered = stations_filtered[stations_filtered['celda_xy'].isin(selected_celdas)]
    
    return stations_filtered

# --- Controles en la Barra Lateral ---
st.sidebar.header("Panel de Control")
with st.sidebar.expander("**1. Filtros Geográficos y de Datos**", expanded=True):
    min_data_perc = st.slider("Filtrar por % de datos mínimo:", 0, 100, 0)
    altitude_ranges = ['0-500', '500-1000', '1000-2000', '2000-3000', '>3000']
    selected_altitudes = st.multiselect('Filtrar por Altitud (m)', options=altitude_ranges)
    regions_list = sorted(gdf_stations['depto_region'].dropna().unique())
    selected_regions = st.multiselect('Filtrar por Depto/Región', options=regions_list)
    
    # Filtros dinámicos para municipios y celdas
    filtered_stations_temp = apply_filters_to_stations(gdf_stations, min_data_perc, selected_altitudes, selected_regions, [], [])
    municipios_list = sorted(filtered_stations_temp['municipio'].dropna().unique())
    selected_municipios = st.multiselect('Filtrar por Municipio', options=municipios_list)

    filtered_stations_temp_2 = apply_filters_to_stations(gdf_stations, min_data_perc, selected_altitudes, selected_regions, selected_municipios, [])
    celdas_list = sorted(filtered_stations_temp_2['celda_xy'].dropna().unique())
    selected_celdas = st.multiselect('Filtrar por Celda_XY', options=celdas_list)

with st.sidebar.expander("**2. Selección de Estaciones y Período**", expanded=True):
    # Lógica para hacer dinámico el filtro de estaciones
    stations_master_list = apply_filters_to_stations(gdf_stations, min_data_perc, selected_altitudes, selected_regions, selected_municipios, selected_celdas)
    stations_options = sorted(stations_master_list['nom_est'].unique())

    if 'select_all_stations_state' not in st.session_state:
        st.session_state.select_all_stations_state = False

    if st.checkbox("Seleccionar/Deseleccionar todas las estaciones", value=st.session_state.select_all_stations_state, key='select_all_checkbox'):
        selected_stations = stations_options
    else:
        selected_stations = st.multiselect(
            'Seleccionar Estaciones',
            options=stations_options,
            default=stations_options if st.session_state.select_all_stations_state else [],
            key='station_multiselect'
        )
    
    if set(selected_stations) == set(stations_options) and not st.session_state.select_all_stations_state:
        st.session_state.select_all_stations_state = True
    elif set(selected_stations) != set(stations_options) and st.session_state.select_all_stations_state:
        st.session_state.select_all_stations_state = False

    # Filtro dinámico para años
    if selected_stations:
        years_with_data_in_selection = sorted([int(col) for col in gdf_stations.loc[gdf_stations['nom_est'].isin(selected_stations)].columns if str(col).isdigit()])
    else:
        years_with_data_in_selection = sorted([int(col) for col in gdf_stations.columns if str(col).isdigit()])

    if not years_with_data_in_selection:
        st.error("No se encontraron años disponibles para la selección actual.")
        st.stop()
    
    year_range = st.slider(
        "Seleccionar Rango de Años", 
        min(years_with_data_in_selection), 
        max(years_with_data_in_selection), 
        (min(years_with_data_in_selection), max(years_with_data_in_selection))
    )
    
    meses_dict = {'Enero': 1, 'Febrero': 2, 'Marzo': 3, 'Abril': 4, 'Mayo': 5, 'Junio': 6, 'Julio': 7, 'Agosto': 8, 'Septiembre': 9, 'Octubre': 10, 'Noviembre': 11, 'Diciembre': 12}
    meses_nombres = st.multiselect("Seleccionar Meses", list(meses_dict.keys()), default=list(meses_dict.keys()))
    meses_numeros = [meses_dict[m] for m in meses_nombres]

with st.sidebar.expander("**3. Opciones de Análisis Avanzado**", expanded=False):
    analysis_mode = st.radio("Análisis de Series Mensuales", ("Usar datos originales", "Completar series (interpolación)"))

if 'analysis_mode' not in st.session_state or st.session_state.analysis_mode != analysis_mode:
    st.session_state.analysis_mode = analysis_mode
    if analysis_mode == "Completar series (interpolación)":
        st.session_state.df_monthly_processed = complete_series(st.session_state.df_long)
    else:
        st.session_state.df_monthly_processed = st.session_state.df_long.copy()

df_monthly_to_process = st.session_state.df_monthly_processed

# --- Lógica de filtrado de datos principal ---
with st.spinner("Filtrando datos..."):
    # 1. Filtrar las estaciones con base en los filtros de la barra lateral
    gdf_filtered = apply_filters_to_stations(
        gdf_stations,
        min_data_perc,
        selected_altitudes,
        selected_regions,
        selected_municipios,
        selected_celdas
    )

    # 2. Filtrar los datos anuales a partir de las estaciones seleccionadas
    df_anual_melted = gdf_filtered.melt(
        id_vars=['nom_est', 'municipio', 'longitud_geo', 'latitud_geo', 'alt_est'],
        value_vars=[str(y) for y in range(year_range[0], year_range[1] + 1) if str(y) in gdf_stations.columns],
        var_name='año', value_name='precipitacion')
    df_anual_melted.dropna(subset=['precipitacion'], inplace=True)


    # 3. Filtrar los datos mensuales a partir de las estaciones seleccionadas
    df_monthly_filtered = df_monthly_to_process[
        (df_monthly_to_process['nom_est'].isin(gdf_filtered['nom_est'].unique())) &
        (df_monthly_to_process['fecha_mes_año'].dt.year >= year_range[0]) &
        (df_monthly_to_process['fecha_mes_año'].dt.year <= year_range[1]) &
        (df_monthly_to_process['fecha_mes_año'].dt.month.isin(meses_numeros))
    ].copy()

    # 4. Actualizar las opciones de estaciones si se usa el multiselect manual
    stations_options = sorted(gdf_filtered['nom_est'].unique())
    if selected_stations:
        gdf_filtered = gdf_filtered[gdf_filtered['nom_est'].isin(selected_stations)]
        df_anual_melted = df_anual_melted[df_anual_melted['nom_est'].isin(selected_stations)]
        df_monthly_filtered = df_monthly_filtered[df_monthly_filtered['nom_est'].isin(selected_stations)]


# --- Pestañas Principales ---
tab_names = ["Distribución Espacial", "Gráficos", "Mapas Avanzados", "Tabla de Estaciones", "Análisis de Anomalías", "Estadísticas", "Análisis ENSO", "Tendencias y Pronósticos", "Descargas"]
mapa_tab, graficos_tab, mapas_avanzados_tab, tabla_estaciones_tab, anomalias_tab, estadisticas_tab, enso_tab, tendencias_tab, descargas_tab = st.tabs(tab_names)

with mapa_tab:
    st.header("Distribución espacial de las Estaciones de Lluvia (1970 - 2021)")
    if not selected_stations:
        st.warning("Por favor, seleccione al menos una estación para ver esta sección.")
    else:
        if not df_anual_melted.empty:
            df_mean_precip = df_anual_melted.groupby('nom_est')['precipitacion'].mean().reset_index()
            gdf_filtered = gdf_filtered.merge(df_mean_precip.rename(columns={'precipitacion': 'precip_media_anual'}), on='nom_est', how='left')
        else:
            gdf_filtered['precip_media_anual'] = np.nan
        gdf_filtered['precip_media_anual'] = gdf_filtered['precip_media_anual'].fillna(0)
        
        sub_tab_mapa, sub_tab_grafico = st.tabs(["Mapa Interactivo", "Gráfico de Disponibilidad de Datos"])

        with sub_tab_mapa:
            controls_col, map_col = st.columns([1, 3])
            
            with controls_col:
                st.subheader("Controles del Mapa")
                
                # --- Opciones de Mapas Base WMS/WMTS ---
                map_options = {
                    "CartoDB Positron (Predeterminado)": {"tiles": "cartodbpositron", "attr": '&copy; <a href="https://carto.com/attributions">CartoDB</a>', "overlay": False},
                    "OpenStreetMap": {"tiles": "OpenStreetMap", "attr": '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors', "overlay": False},
                    "Topografía (OpenTopoMap)": {"tiles": "https://{s}.tile.opentopomap.org/{z}/{x}/{y}.png", "attr": 'Map data: &copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors, <a href="http://viewfinderpanoramas.org">SRTM</a> | Map style: &copy; <a href="https://opentopomap.org">OpenTopoMap</a> (<a href="https://creativecommons.org/licenses/by-sa/3.0/">CC-BY-SA</a>)', "overlay": False},
                    "Relieve (Stamen Terrain)": {"tiles": "Stamen Terrain", "attr": 'Map tiles by <a href="http://stamen.com">Stamen Design</a>, <a href="http://creativecommons.org/licenses/by/3.0">CC BY 3.0</a> &mdash; Map data &copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors', "overlay": False},
                    "Relieve y Océanos (GEBCO)": {"url": "https://www.gebco.net/data_and_products/gebco_web_services/web_map_service/web_map_service.php", "layers": "GEBCO_2021_Surface", "transparent": False, "attr": "GEBCO 2021", "overlay": True},
                    "Mapa de Colombia (WMS IDEAM)": {"url": "https://geoservicios.ideam.gov.co/geoserver/ideam/wms", "layers": "ideam:col_admin", "transparent": True, "attr": "IDEAM", "overlay": True},
                    "Cobertura de la Tierra (WMS IGAC)": {"url": "https://servicios.igac.gov.co/server/services/IDEAM/IDEAM_Cobertura_Corine/MapServer/WMSServer", "layers": "IDEAM_Cobertura_Corine_Web", "transparent": True, "attr": "IGAC", "overlay": True},
                }
                
                base_maps = {k: v for k, v in map_options.items() if not v.get("overlay")}
                overlays = {k: v for k, v in map_options.items() if v.get("overlay")}
                
                selected_base_map_name = st.selectbox("Seleccionar Mapa Base", list(base_maps.keys()))
                
                selected_overlays = st.multiselect("Seleccionar Capas Adicionales", list(overlays.keys()))
                
                if not gdf_filtered.empty:
                    st.markdown("---")
                    m1, m2 = st.columns([1, 3])
                    with m1:
                        if os.path.exists(logo_gota_path):
                            st.image(logo_gota_path, width=50)
                    with m2:
                        st.metric("Estaciones en Vista", len(gdf_filtered))

                    st.markdown("---")
                    map_centering = st.radio("Opciones de centrado:", ("Automático", "Vistas Predefinidas"), key="map_centering_radio")

                    if 'map_view' not in st.session_state:
                        st.session_state.map_view = {"location": [4.57, -74.29], "zoom": 5}

                    if map_centering == "Vistas Predefinidas":
                        if st.button("Ver Colombia"):
                            st.session_state.map_view = {"location": [4.57, -74.29], "zoom": 5}
                        if st.button("Ver Antioquia"):
                            st.session_state.map_view = {"location": [6.24, -75.58], "zoom": 8}
                        if st.button("Ajustar a Selección"):
                            bounds = gdf_filtered.total_bounds
                            center_lat = (bounds[1] + bounds[3]) / 2
                            center_lon = (bounds[0] + bounds[2]) / 2
                            st.session_state.map_view = {"location": [center_lat, center_lon], "zoom": 9}
                
                # Resumen de filtros activos
                st.markdown("---")
                with st.expander("Resumen de Filtros Activos", expanded=True):
                    summary_text = f"**Período:** {year_range[0]} - {year_range[1]}\n\n"
                    summary_text += f"**% Mínimo de Datos:** {min_data_perc}%\n\n"
                    if selected_altitudes:
                        summary_text += f"**Altitud:** {', '.join(selected_altitudes)}\n\n"
                    if selected_regions:
                        summary_text += f"**Región:** {', '.join(selected_regions)}\n\n"
                    if selected_municipios:
                        summary_text += f"**Municipio:** {', '.join(selected_municipios)}\n\n"
                    if selected_celdas:
                        summary_text += f"**Celda XY:** {', '.join(selected_celdas)}\n\n"
                    st.info(summary_text)
                
            with map_col:
                if not gdf_filtered.empty:
                    base_map_config = base_maps[selected_base_map_name]
                    
                    m = folium.Map(
                        location=st.session_state.map_view["location"],
                        zoom_start=st.session_state.map_view["zoom"],
                        tiles=base_map_config.get("tiles", "OpenStreetMap"),
                        attr=base_map_config.get("attr", None)
                    )
                    
                    if map_centering == "Automático":
                        bounds = gdf_filtered.total_bounds
                        m.fit_bounds([[bounds[1], bounds[0]], [bounds[3], bounds[2]]])

                    # Añadir la capa de municipios
                    folium.GeoJson(gdf_municipios.to_json(), name='Municipios').add_to(m)

                    # Añadir las capas WMS seleccionadas
                    for layer_name in selected_overlays:
                        layer_config = overlays[layer_name]
                        folium.raster_layers.WmsTileLayer(
                            url=layer_config["url"],
                            layers=layer_config["layers"],
                            fmt='image/png',
                            transparent=layer_config.get("transparent", False),
                            overlay=True,
                            control=True,
                            name=layer_name
                        ).add_to(m)
                    
                    # Añadir la capa de estaciones DE ÚLTIMO para que se vean sobre todas las capas
                    marker_cluster = MarkerCluster(name='Estaciones').add_to(m)
                    for _, row in gdf_filtered.iterrows():
                        html = f"""
                        <b>Estación:</b> {row['nom_est']}<br>
                        <b>Municipio:</b> {row['municipio']}<br>
                        <b>Celda:</b> {row['celda_xy']}<br>
                        <b>% Datos Disponibles:</b> {row['porc_datos']:.0f}%<br>
                        <b>Ppt. Media Anual (mm):</b> {row['precip_media_anual']:.0f}
                        """
                        folium.Marker(
                            location=[row['latitud_geo'], row['longitud_geo']],
                            tooltip=html
                        ).add_to(marker_cluster)

                    # Añadir el control de capas para poder ver y ocultar las capas
                    folium.LayerControl().add_to(m)
                    
                    # Añadir Minimapa
                    minimap = MiniMap(toggle_display=True)
                    m.add_child(minimap)
                    
                    folium_static(m, height=700)
                else:
                    st.warning("No hay estaciones seleccionadas para mostrar en el mapa.")

        with sub_tab_grafico:
            st.subheader("Disponibilidad y Composición de Datos por Estación")
            if not gdf_filtered.empty:
                if analysis_mode == "Completar series (interpolación)":
                    st.info("Mostrando la composición de datos originales vs. completados para el período seleccionado.")
                    
                    data_composition = df_monthly_filtered.groupby(['nom_est', 'origen']).size().unstack(fill_value=0)
                    if 'Original' not in data_composition:
                        data_composition['Original'] = 0
                    if 'Completado' not in data_composition:
                        data_composition['Completado'] = 0
                    
                    data_composition['total'] = data_composition['Original'] + data_composition['Completado']
                    data_composition['% Original'] = (data_composition['Original'] / data_composition['total']) * 100
                    data_composition['% Completado'] = (data_composition['Completado'] / data_composition['total']) * 100

                    sort_order_comp = st.radio(
                        "Ordenar por:",
                        ["% Datos Originales (Mayor a Menor)", "% Datos Originales (Menor a Mayor)", "Alfabético"],
                        horizontal=True, key="sort_comp"
                    )
                    if "Mayor a Menor" in sort_order_comp:
                        data_composition = data_composition.sort_values("% Original", ascending=False)
                    elif "Menor a Mayor" in sort_order_comp:
                        data_composition = data_composition.sort_values("% Original", ascending=True)
                    else:
                        data_composition = data_composition.sort_index(ascending=True)

                    df_plot = data_composition.reset_index().melt(
                        id_vars='nom_est',
                        value_vars=['% Original', '% Completado'],
                        var_name='Tipo de Dato',
                        value_name='Porcentaje'
                    )

                    fig_comp = px.bar(df_plot,
                                      x='nom_est',
                                      y='Porcentaje',
                                      color='Tipo de Dato',
                                      title='Composición de Datos por Estación',
                                      labels={'nom_est': 'Estación', 'Porcentaje': '% del Período'},
                                      color_discrete_map={'% Original': '#1f77b4', '% Completado': '#ff7f0e'},
                                      text_auto='.1f')
                    fig_comp.update_layout(height=600, xaxis={'categoryorder': 'trace'})
                    st.plotly_chart(fig_comp, use_container_width=True)

                else:
                    st.info("Mostrando el porcentaje de disponibilidad de datos según el archivo de estaciones.")
                    sort_order_disp = st.radio(
                        "Ordenar estaciones por:",
                        ["% Datos (Mayor a Menor)", "% Datos (Menor a Mayor)", "Alfabético"],
                        horizontal=True, key="sort_disp"
                    )

                    df_chart = gdf_filtered.copy()
                    if "% Datos (Mayor a Menor)" in sort_order_disp:
                        df_chart = df_chart.sort_values("porc_datos", ascending=False)
                    elif "% Datos (Menor a Mayor)" in sort_order_disp:
                        df_chart = df_chart.sort_values("porc_datos", ascending=True)
                    else:
                        df_chart = df_chart.sort_values("nom_est", ascending=True)
                    
                    fig_disp = px.bar(df_chart, 
                                      x='nom_est', 
                                      y='porc_datos', 
                                      title='Porcentaje de Disponibilidad de Datos Históricos',
                                      labels={'nom_est': 'Estación', 'porc_datos': '% de Datos Disponibles'},
                                      color='porc_datos', 
                                      color_continuous_scale=px.colors.sequential.Viridis)
                    fig_disp.update_layout(
                        height=600,
                        xaxis={'categoryorder':'trace'}
                    )
                    st.plotly_chart(fig_disp, use_container_width=True)
            else:
                st.warning("No hay estaciones seleccionadas para mostrar el gráfico.")

with graficos_tab:
    st.header("Visualizaciones de Precipitación")
    if not selected_stations:
        st.warning("Por favor, seleccione al menos una estación para ver esta sección.")
    else:
        sub_tab_anual, sub_tab_mensual, sub_tab_comparacion, sub_tab_distribucion, sub_tab_acumulada, sub_tab_altitud = st.tabs(["Análisis Anual", "Análisis Mensual", "Comparación Rápida", "Distribución", "Acumulada", "Relación Altitud"])

        with sub_tab_anual:
            anual_graf_tab, anual_analisis_tab = st.tabs(["Gráfico de Serie Anual", "Análisis Multianual"])
            
            with anual_graf_tab:
                if not df_anual_melted.empty:
                    st.subheader("Precipitación Anual (mm)")
                    chart_anual = alt.Chart(df_anual_melted).mark_line(point=True).encode(
                        x=alt.X('año:O', title='Año'),
                        y=alt.Y('precipitacion:Q', title='Precipitación (mm)'),
                        color='nom_est:N',
                        tooltip=[alt.Tooltip('nom_est'), alt.Tooltip('año'), alt.Tooltip('precipitacion:Q', format='.0f')]
                    ).properties(height=600).interactive()
                    st.altair_chart(chart_anual, use_container_width=True)

            with anual_analisis_tab:
                if not df_anual_melted.empty:
                    st.subheader("Precipitación Media Multianual")
                    st.caption(f"Período de análisis: {year_range[0]} - {year_range[1]}")

                    chart_type_annual = st.radio("Seleccionar tipo de gráfico:", ("Gráfico de Barras (Promedio)", "Gráfico de Cajas (Distribución)"), key="avg_chart_type_annual", horizontal=True)

                    if chart_type_annual == "Gráfico de Barras (Promedio)":
                        df_summary = df_anual_melted.groupby('nom_est', as_index=False)['precipitacion'].mean().round(0)
                        sort_order = st.radio("Ordenar estaciones por:", ["Promedio (Mayor a Menor)", "Promedio (Menor a Mayor)", "Alfabético"], horizontal=True, key="sort_annual_avg")
                        if "Mayor a Menor" in sort_order:
                            df_summary = df_summary.sort_values("precipitacion", ascending=False)
                        elif "Menor a Mayor" in sort_order:
                            df_summary = df_summary.sort_values("precipitacion", ascending=True)
                        else:
                            df_summary = df_summary.sort_values("nom_est", ascending=True)

                        fig_avg = px.bar(df_summary, x='nom_est', y='precipitacion', title='Promedio de Precipitación Anual', labels={'nom_est': 'Estación', 'precipitacion': 'Precipitación Media Anual (mm)'}, color='precipitacion', color_continuous_scale=px.colors.sequential.Blues_r)
                        fig_avg.update_layout(height=600, xaxis={'categoryorder':'total descending' if "Mayor a Menor" in sort_order else ('total ascending' if "Menor a Mayor" in sort_order else 'trace')})
                        st.plotly_chart(fig_avg, use_container_width=True)
                    else:
                        fig_box = px.box(df_anual_melted, x='nom_est', y='precipitacion', color='nom_est', points='all', title='Distribución de la Precipitación Anual por Estación', labels={'nom_est': 'Estación', 'precipitacion': 'Precipitación Anual (mm)'})
                        fig_box.update_layout(height=600)
                        st.plotly_chart(fig_box, use_container_width=True)

        with sub_tab_mensual:
            mensual_graf_tab, mensual_enso_tab, mensual_datos_tab = st.tabs(["Gráfico de Serie Mensual", "Análisis ENSO en el Período", "Tabla de Datos"])

            with mensual_graf_tab:
                if not df_monthly_filtered.empty:
                    control_col1, control_col2 = st.columns(2)
                    chart_type = control_col1.radio("Tipo de Gráfico:", ["Líneas y Puntos", "Nube de Puntos", "Gráfico de Cajas (Distribución Mensual)"], key="monthly_chart_type")
                    color_by = control_col2.radio("Colorear por:", ["Estación", "Mes"], key="monthly_color_by", disabled=(chart_type == "Gráfico de Cajas (Distribución Mensual)"))

                    if chart_type != "Gráfico de Cajas (Distribución Mensual)":
                        base_chart = alt.Chart(df_monthly_filtered).encode(x=alt.X('fecha_mes_año:T', title='Fecha'), y=alt.Y('precipitation:Q', title='Precipitación (mm)'), tooltip=[alt.Tooltip('fecha_mes_año', format='%Y-%m'), alt.Tooltip('precipitation', format='.0f'), 'nom_est', 'origen', alt.Tooltip('mes:N', title="Mes")])
                        if color_by == "Estación":
                            color_encoding = alt.Color('nom_est:N', legend=alt.Legend(title="Estaciones"))
                        else:
                            color_encoding = alt.Color('month(fecha_mes_año):N', legend=alt.Legend(title="Meses"), scale=alt.Scale(scheme='tableau20'))
                        
                        if chart_type == "Líneas y Puntos":
                            line_chart = base_chart.mark_line(opacity=0.4, color='lightgray').encode(detail='nom_est:N')
                            point_chart = base_chart.mark_point(filled=True, size=60).encode(color=color_encoding)
                            final_chart = (line_chart + point_chart)
                        else:
                            point_chart = base_chart.mark_point(filled=True, size=60).encode(color=color_encoding)
                            final_chart = point_chart
                        st.altair_chart(final_chart.properties(height=600).interactive(), use_container_width=True)
                    else:
                        st.subheader("Distribución de la Precipitación Mensual")
                        fig_box_monthly = px.box(df_monthly_filtered, x='mes', y='precipitation', color='nom_est', title='Distribución de la Precipitación por Mes', labels={'mes': 'Mes', 'precipitation': 'Precipitación Mensual (mm)', 'nom_est': 'Estación'})
                        fig_box_monthly.update_layout(height=600)
                        st.plotly_chart(fig_box_monthly, use_container_width=True)
            
            with mensual_enso_tab:
                enso_filtered = df_enso[(df_enso['fecha_mes_año'].dt.year >= year_range[0]) & (df_enso['fecha_mes_año'].dt.year <= year_range[1]) & (df_enso['fecha_mes_año'].dt.month.isin(meses_numeros))]
                fig_enso_mensual = create_enso_chart(enso_filtered)
                st.plotly_chart(fig_enso_mensual, use_container_width=True, key="enso_chart_mensual")

            with mensual_datos_tab:
                st.subheader("Datos de Precipitación Mensual Detallados")
                if not df_monthly_filtered.empty:
                    df_values = df_monthly_filtered.pivot_table(index='fecha_mes_año', columns='nom_est', values='precipitation').round(0)
                    st.dataframe(df_values)
            
        with sub_tab_comparacion:
            st.subheader("Comparación de Precipitación entre Estaciones")
            if len(selected_stations) < 2:
                st.info("Seleccione al menos dos estaciones para comparar.")
            else:
                # Gráfico de Líneas para Precipitación Mensual Promedio
                st.markdown("##### Precipitación Mensual Promedio")
                df_monthly_avg = df_monthly_filtered.groupby(['nom_est', 'mes'])['precipitation'].mean().reset_index()
                
                fig_avg_monthly = px.line(df_monthly_avg, 
                                          x='mes', y='precipitation', color='nom_est',
                                          labels={'mes': 'Mes', 'precipitation': 'Precipitación Promedio (mm)'},
                                          title='Promedio de Precipitación Mensual por Estación')
                fig_avg_monthly.update_layout(height=600, xaxis = dict(tickmode='array', tickvals=list(meses_dict.values()), ticktext=list(meses_dict.keys())))
                st.plotly_chart(fig_avg_monthly, use_container_width=True)

                # Gráfico de Cajas para Precipitación Anual
                st.markdown("##### Distribución de Precipitación Anual")
                fig_box_annual = px.box(df_anual_melted, x='nom_est', y='precipitacion', color='nom_est',
                                         title='Distribución de la Precipitación Anual por Estación',
                                         labels={'nom_est': 'Estación', 'precipitacion': 'Precipitación Anual (mm)'})
                st.plotly_chart(fig_box_annual, use_container_width=True)

        with sub_tab_distribucion:
            st.subheader("Distribución de la Precipitación")
            
            distribucion_tipo = st.radio("Seleccionar tipo de distribución:", ("Anual", "Mensual"), horizontal=True)

            if distribucion_tipo == "Anual":
                if not df_anual_melted.empty:
                    fig_hist_anual = px.histogram(df_anual_melted, x='precipitacion', color='nom_est',
                                                   title='Distribución Anual de Precipitación',
                                                   labels={'precipitacion': 'Precipitación Anual (mm)', 'count': 'Frecuencia'})
                    fig_hist_anual.update_layout(height=600)
                    st.plotly_chart(fig_hist_anual, use_container_width=True)
                else:
                    st.info("No hay datos anuales para mostrar la distribución.")
            else: # Mensual
                if not df_monthly_filtered.empty:
                    fig_hist_mensual = px.histogram(df_monthly_filtered, x='precipitation', color='nom_est',
                                                     title='Distribución Mensual de Precipitación',
                                                     labels={'precipitation': 'Precipitación Mensual (mm)', 'count': 'Frecuencia'})
                    fig_hist_mensual.update_layout(height=600)
                    st.plotly_chart(fig_hist_mensual, use_container_width=True)
                else:
                    st.info("No hay datos mensuales para mostrar la distribución.")

        with sub_tab_acumulada:
            st.subheader("Precipitación Acumulada Anual")
            if not df_anual_melted.empty:
                df_acumulada = df_anual_melted.groupby(['año', 'nom_est'])['precipitacion'].sum().reset_index()
                fig_acumulada = px.bar(df_acumulada, x='año', y='precipitacion', color='nom_est',
                                       title='Precipitación Acumulada por Año',
                                       labels={'año': 'Año', 'precipitacion': 'Precipitación Acumulada (mm)'})
                fig_acumulada.update_layout(barmode='group', height=600)
                st.plotly_chart(fig_acumulada, use_container_width=True)
            else:
                st.info("No hay datos para calcular la precipitación acumulada.")

        with sub_tab_altitud:
            st.subheader("Relación entre Altitud y Precipitación")
            if not df_anual_melted.empty:
                df_relacion = df_anual_melted.groupby('nom_est')['precipitacion'].mean().reset_index()
                df_relacion = df_relacion.merge(gdf_stations[['nom_est', 'alt_est']].drop_duplicates(), on='nom_est')

                fig_relacion = px.scatter(df_relacion, x='alt_est', y='precipitacion', color='nom_est',
                                          title='Relación entre Precipitación Media Anual y Altitud',
                                          labels={'alt_est': 'Altitud (m)', 'precipitacion': 'Precipitación Media Anual (mm)'})
                fig_relacion.update_layout(height=600)
                st.plotly_chart(fig_relacion, use_container_width=True)
            else:
                st.info("No hay datos para analizar la relación entre altitud y precipitación.")


with mapas_avanzados_tab:
    st.header("Mapas Avanzados")
    gif_tab, temporal_tab, compare_tab, kriging_tab = st.tabs(["Animación GIF (Antioquia)", "Visualización Temporal", "Comparación de Mapas", "Interpolación Kriging"])

    with gif_tab:
        st.subheader("Distribución Espacio-Temporal de la Lluvia en Antioquia")
        gif_path = "PPAM.gif"
        if os.path.exists(gif_path):
            img_col1, img_col2 = st.columns([1, 1])
            with img_col1:
                if 'gif_rerun_count' not in st.session_state:
                    st.session_state.gif_rerun_count = 0

                gif_placeholder = st.empty()
                
                with open(gif_path, "rb") as file:
                    contents = file.read()
                    data_url = base64.b64encode(contents).decode("utf-8")
                    file.close()
                gif_placeholder.markdown(
                    f'<img src="data:image/gif;base64,{data_url}" alt="Animación PPAM" style="width:100%;">',
                    unsafe_allow_html=True,
                )
                st.caption("Precipitación Promedio Anual Multianual en Antioquia")

                if st.button("Reiniciar Animación", key="restart_gif"):
                    st.session_state.gif_rerun_count += 1
                    with open(gif_path, "rb") as file:
                        contents = file.read()
                        data_url = base64.b64encode(contents).decode("utf-8")
                        file.close()
                    gif_placeholder.markdown(
                        f'<img src="data:image/gif;base64,{data_url}" alt="Animación PPAM {st.session_state.gif_rerun_count}" style="width:100%;">',
                        unsafe_allow_html=True,
                    )
        else:
            st.warning("No se encontró el archivo GIF 'PPAM.gif'. Asegúrate de que esté en el directorio principal de la aplicación.")

    with temporal_tab:
        if not selected_stations:
            st.warning("Por favor, seleccione al menos una estación para ver esta sección.")
        else:
            exp_tab, race_tab, anim_tab = st.tabs(["Explorador Interactivo", "Gráfico de Carrera", "Mapa Animado"])

            with exp_tab:
                st.subheader("Explorador Anual de Precipitación")
                if not df_anual_melted.empty:
                    all_years_int = sorted([int(y) for y in df_anual_melted['año'].unique()])
                    if all_years_int:
                        selected_year = st.slider('Seleccione un Año para Explorar', min_value=min(all_years_int), max_value=max(all_years_int), value=min(all_years_int))
                        controls_col, map_col = st.columns([1, 3])
                        with controls_col:
                            st.markdown("##### Opciones de Visualización")
                            map_view_option = st.radio("Seleccionar vista geográfica:", ["Zona de Selección", "Antioquia", "Colombia"], horizontal=True, key="map_view_interactive")
                            st.markdown(f"#### Resumen del Año: {selected_year}")
                            df_year_filtered = df_anual_melted[df_anual_melted['año'] == str(selected_year)].dropna(subset=['precipitacion'])
                            logo_col, info_col = st.columns([1, 4])
                            with logo_col:
                                if os.path.exists(logo_gota_path):
                                    st.image(logo_gota_path, width=40)
                            with info_col:
                                st.metric(f"Estaciones con Datos en {selected_year}", f"{len(df_year_filtered)} de {len(selected_stations)}")
                            if not df_year_filtered.empty:
                                max_row = df_year_filtered.loc[df_year_filtered['precipitacion'].idxmax()]
                                min_row = df_year_filtered.loc[df_year_filtered['precipitacion'].idxmin()]
                                st.info(f"""
                                **Ppt. Máxima ({selected_year}):**
                                {max_row['nom_est']} ({max_row['precipitacion']:.0f} mm)
                                
                                **Ppt. Mínima ({selected_year}):**
                                {min_row['nom_est']} ({min_row['precipitacion']:.0f} mm)
                                """)
                            else:
                                st.warning(f"No hay datos de precipitación para el año {selected_year}.")
                        with map_col:
                            min_precip_range, max_precip_range = df_anual_melted['precipitacion'].min(), df_anual_melted['precipitacion'].max()
                            fig_interactive_map = px.scatter_geo(
                                df_year_filtered, lat='latitud_geo', lon='longitud_geo',
                                color='precipitacion', size='precipitacion',
                                hover_name='nom_est', title=f'Precipitación Anual por Estación - Año {selected_year}',
                                hover_data={'municipio': True, 'precipitacion': ':.0f'},
                                color_continuous_scale=px.colors.sequential.YlGnBu, range_color=[min_precip_range, max_precip_range]
                            )
                            bounds = {}
                            if map_view_option == "Zona de Selección":
                                bounds_values = gdf_stations.loc[gdf_stations['nom_est'].isin(selected_stations)].total_bounds
                                bounds = {'lon': [bounds_values[0]-0.2, bounds_values[2]+0.2], 'lat': [bounds_values[1]-0.2, bounds_values[3]+0.2]}
                            elif map_view_option == "Antioquia":
                                bounds = {'lon': [-77, -74.5], 'lat': [5.5, 8.5]}
                            elif map_view_option == "Colombia":
                                bounds = {'lon': [-79, -67], 'lat': [-4.5, 12.5]}
                            fig_interactive_map.update_geos(lataxis_range=bounds.get('lat'), lonaxis_range=bounds.get('lon'), visible=True, showcoastlines=True, coastlinewidth=0.5, showland=True, landcolor="rgb(243, 243, 243)", showocean=True, oceancolor="rgb(220, 235, 255)", showcountries=True, countrywidth=0.5)
                            fig_interactive_map.update_layout(height=700)
                            st.plotly_chart(fig_interactive_map, use_container_width=True)

            with race_tab:
                st.subheader("Ranking Anual de Precipitación por Estación")
                if not df_anual_melted.empty:
                    station_order = df_anual_melted.groupby('nom_est')['precipitacion'].sum().sort_values(ascending=True).index
                    
                    fig_racing = px.bar(
                        df_anual_melted,
                        x="precipitacion",
                        y="nom_est",
                        animation_frame="año",
                        orientation='h',
                        text="precipitacion",
                        labels={'precipitacion': 'Precipitación Anual (mm)', 'nom_est': 'Estación'},
                        title="Evolución de Precipitación Anual por Estación",
                        category_orders={'nom_est': station_order}
                    )
                    fig_racing.update_traces(texttemplate='%{x:.0f}', textposition='outside')
                    fig_racing.update_layout(
                        xaxis_range=[0, df_anual_melted['precipitacion'].max() * 1.15],
                        height=max(600, len(selected_stations) * 35),
                        title_font_size=20,
                        font_size=12
                    )
                    fig_racing.layout.sliders[0]['currentvalue']['font']['size'] = 24
                    fig_racing.layout.sliders[0]['currentvalue']['prefix'] = '<b>Año: </b>'
                    fig_racing.layout.updatemenus[0].buttons[0].args[1]['frame']['duration'] = 800
                    fig_racing.layout.updatemenus[0].buttons[0].args[1]['transition']['duration'] = 500
                    st.plotly_chart(fig_racing, use_container_width=True)

            with anim_tab:
                st.subheader("Mapa Animado de Precipitación Anual")
                if not df_anual_melted.empty:
                    all_years = sorted(df_anual_melted['año'].unique())
                    if all_years:
                        all_selected_stations_info = gdf_stations.loc[gdf_stations['nom_est'].isin(selected_stations)][['nom_est', 'latitud_geo', 'longitud_geo', 'alt_est']].drop_duplicates()
                        full_grid = pd.MultiIndex.from_product([all_selected_stations_info['nom_est'], all_years], names=['nom_est', 'año']).to_frame(index=False)
                        full_grid = pd.merge(full_grid, all_selected_stations_info, on='nom_est')
                        df_anim_complete = pd.merge(full_grid, df_anual_melted[['nom_est', 'año', 'precipitacion']], on=['nom_est', 'año'], how='left')
                        df_anim_complete['texto_tooltip'] = df_anim_complete.apply(lambda row: f"<b>Estación:</b> {row['nom_est']}<br><b>Precipitación:</b> {row['precipitacion']:.0f} mm" if pd.notna(row['precipitacion']) else f"<b>Estación:</b> {row['nom_est']}<br><b>Precipitación:</b> Sin datos", axis=1)
                        df_anim_complete['precipitacion_plot'] = df_anim_complete['precipitacion'].fillna(0)
                        min_precip_anim, max_precip_anim = df_anual_melted['precipitacion'].min(), df_anual_melted['precipitacion'].max()
                        fig_mapa_animado = px.scatter_geo(df_anim_complete, lat='latitud_geo', lon='longitud_geo', color='precipitacion_plot', size='precipitacion_plot', hover_name='nom_est', hover_data={'latitud_geo': False, 'longitud_geo': False, 'precipitacion_plot': False, 'texto_tooltip': True}, animation_frame='año', projection='natural earth', title='Precipitación Anual por Estación', color_continuous_scale=px.colors.sequential.YlGnBu, range_color=[min_precip_anim, max_precip_anim])
                        fig_mapa_animado.update_traces(hovertemplate='%{customdata[0]}')
                        fig_mapa_animado.update_geos(fitbounds="locations", visible=True, showcoastlines=True, coastlinewidth=0.5, showland=True, landcolor="rgb(243, 243, 243)", showocean=True, oceancolor="rgb(220, 235, 255)", showcountries=True, countrywidth=0.5)
                        fig_mapa_animado.update_layout(height=700, sliders=[dict(currentvalue=dict(font=dict(size=24, color="#707070"), prefix='<b>Año: </b>', visible=True))])
                        st.plotly_chart(fig_mapa_animado, use_container_width=True)
    
    with compare_tab:
        st.subheader("Comparación de Mapas Anuales")
        if not selected_stations:
            st.warning("Por favor, seleccione al menos una estación para ver esta sección.")
        elif not df_anual_melted.empty and len(df_anual_melted['año'].unique()) > 0:
            map_col1, map_col2 = st.columns(2)
            min_year, max_year = int(df_anual_melted['año'].min()), int(df_anual_melted['año'].max())

            year1 = map_col1.slider("Seleccione el año para el Mapa 1", min_year, max_year, max_year)
            year2 = map_col2.slider("Seleccione el año para el Mapa 2", min_year, max_year, max_year - 1 if max_year > min_year else max_year)
            
            data_year1 = df_anual_melted[df_anual_melted['año'].astype(int) == year1]
            data_year2 = df_anual_melted[df_anual_melted['año'].astype(int) == year2]
            
            logo_col1, metric_col1 = map_col1.columns([1,4])
            logo_col2, metric_col2 = map_col2.columns([1,4])
            
            with logo_col1:
                if os.path.exists(logo_gota_path):
                    st.image(logo_gota_path, width=30)
            with metric_col1:
                st.metric(f"Estaciones con datos en {year1}", f"{len(data_year1)} de {len(selected_stations)}")
            
            with logo_col2:
                if os.path.exists(logo_gota_path):
                    st.image(logo_gota_path, width=30)
            with metric_col2:
                st.metric(f"Estaciones con datos en {year2}", f"{len(data_year2)} de {len(selected_stations)}")

            min_precip_comp, max_precip_comp = int(df_anual_melted['precipitacion'].min()), int(df_anual_melted['precipitacion'].max())
            color_range_comp = st.slider("Rango de Escala de Color (mm)", min_precip_comp, max_precip_comp, (min_precip_comp, max_precip_comp), key="color_comp")

            fig1 = px.scatter_geo(data_year1, lat='latitud_geo', lon='longitud_geo', color='precipitacion', size='precipitacion', hover_name='nom_est', hover_data={'municipio': True, 'precipitacion': ':.0f'}, color_continuous_scale='YlGnBu', range_color=color_range_comp, projection='natural earth', title=f"Precipitación en {year1}")
            fig1.update_geos(fitbounds="locations", visible=True)
            map_col1.plotly_chart(fig1, use_container_width=True)
            
            fig2 = px.scatter_geo(data_year2, lat='latitud_geo', lon='longitud_geo', color='precipitacion', size='precipitacion', hover_name='nom_est', hover_data={'municipio': True, 'precipitacion': ':.0f'}, color_continuous_scale='YlGnBu', range_color=color_range_comp, projection='natural earth', title=f"Precipitación en {year2}")
            fig2.update_geos(fitbounds="locations", visible=True)
            map_col2.plotly_chart(fig2, use_container_width=True)
        else:
            st.warning("No hay años disponibles para la comparación.")

    with kriging_tab:
        st.subheader("Interpolación Kriging para un Año Específico")
        if not selected_stations:
            st.warning("Por favor, seleccione al menos una estación para ver esta sección.")
        elif not df_anual_melted.empty and len(df_anual_melted['año'].unique()) > 0:
            min_year, max_year = int(df_anual_melted['año'].min()), int(df_anual_melted['año'].max())
            year_kriging = st.slider("Seleccione el año para la interpolación", min_year, max_year, max_year, key="year_kriging")
            data_year_kriging = df_anual_melted[df_anual_melted['año'].astype(int) == year_kriging]
            
            logo_col_k, metric_col_k = st.columns([1,8])
            with logo_col_k:
                if os.path.exists(logo_gota_path):
                    st.image(logo_gota_path, width=40)
            with metric_col_k:
                st.metric(f"Estaciones con datos en {year_kriging}", f"{len(data_year_kriging)} de {len(selected_stations)}")

            if len(data_year_kriging) < 3:
                st.warning(f"Se necesitan al menos 3 estaciones con datos en el año {year_kriging} para generar el mapa Kriging.")
            else:
                with st.spinner("Generando mapa Kriging..."):
                    data_year_kriging['tooltip'] = data_year_kriging.apply(
                        lambda row: f"<b>{row['nom_est']}</b><br>Municipio: {row['municipio']}<br>Ppt: {row['precipitacion']:.0f} mm",
                        axis=1
                    )
                    lons, lats, vals = data_year_kriging['longitud_geo'].values, data_year_kriging['latitud_geo'].values, data_year_kriging['precipitacion'].values
                    bounds = gdf_stations.loc[gdf_stations['nom_est'].isin(selected_stations)].total_bounds
                    lon_range = [bounds[0] - 0.1, bounds[2] + 0.1]
                    lat_range = [bounds[1] - 0.1, bounds[3] + 0.1]
                    grid_lon, grid_lat = np.linspace(lon_range[0], lon_range[1], 100), np.linspace(lat_range[0], lat_range[1], 100)
                    OK = OrdinaryKriging(lons, lats, vals, variogram_model='linear', verbose=False, enable_plotting=False)
                    z, ss = OK.execute('grid', grid_lon, grid_lat)

                    fig_krig = go.Figure(data=go.Contour(z=z.T, x=grid_lon, y=grid_lat, colorscale='YlGnBu', contours=dict(showlabels=True, labelfont=dict(size=12, color='white'))))
                    fig_krig.add_trace(go.Scatter(
                        x=lons, y=lats, mode='markers', 
                        marker=dict(color='red', size=5, symbol='circle'), 
                        name='Estaciones',
                        text=data_year_kriging['tooltip'],
                        hoverinfo='text'
                    ))
                    fig_krig.update_layout(height=700, title=f"Superficie de Precipitación Interpolada (Kriging) - Año {year_kriging}", xaxis_title="Longitud", yaxis_title="Latitud")
                    st.plotly_chart(fig_krig, use_container_width=True)
        else:
            st.warning("No hay datos para realizar la interpolación.")

with tabla_estaciones_tab:
    st.header("Información Detallada de las Estaciones")
    if not selected_stations:
        st.warning("Por favor, seleccione al menos una estación para ver esta sección.")
    elif not df_anual_melted.empty:
        df_info_table = gdf_filtered[['nom_est', 'alt_est', 'municipio', 'depto_region', 'porc_datos']].copy()
        df_mean_precip = df_anual_melted.groupby('nom_est')['precipitacion'].mean().round(0).reset_index()
        df_mean_precip.rename(columns={'precipitacion': 'Precipitación media anual (mm)'}, inplace=True)
        df_info_table = df_info_table.merge(df_mean_precip, on='nom_est', how='left')
        st.dataframe(df_info_table)
    else:
        st.info("No hay datos de precipitación anual para mostrar en la selección actual.")

with anomalias_tab:
    st.header("Análisis de Anomalías de Precipitación")
    if not selected_stations:
        st.warning("Por favor, seleccione al menos una estación para ver esta sección.")
    else:
        df_long_filtered_stations = df_long[df_long['nom_est'].isin(selected_stations)]
        df_climatology = df_long_filtered_stations.groupby(['nom_est', 'mes'])['precipitation'].mean().reset_index().rename(columns={'precipitation': 'precip_promedio_mes'})
        
        df_anomalias = pd.merge(df_monthly_filtered, df_climatology, on=['nom_est', 'mes'], how='left')
        df_anomalias['anomalia'] = df_anomalias['precipitation'] - df_anomalias['precip_promedio_mes']

        if df_anomalias.empty or df_anomalias['anomalia'].isnull().all():
            st.warning("No hay suficientes datos históricos para las estaciones y el período seleccionado para calcular y mostrar las anomalías.")
        else:
            anom_graf_tab, anom_mapa_tab, anom_fase_tab, anom_extremos_tab = st.tabs(["Gráfico de Anomalías", "Mapa de Anomalías Anuales", "Anomalías por Fase ENSO", "Tabla de Eventos Extremos"])

            with anom_graf_tab:
                avg_monthly_anom = df_anomalias.groupby(['fecha_mes_año', 'mes'])['anomalia'].mean().reset_index()
                df_plot = pd.merge(avg_monthly_anom, df_enso[['fecha_mes_año', 'anomalia_oni']], on='fecha_mes_año', how='left')
                fig = create_anomaly_chart(df_plot)
                st.plotly_chart(fig, use_container_width=True)

            with anom_mapa_tab:
                st.subheader("Mapa Interactivo de Anomalías Anuales")
                df_anomalias_anual = df_anomalias.groupby(['nom_est', 'año'])['anomalia'].sum().reset_index()
                df_anomalias_anual = pd.merge(df_anomalias_anual, gdf_stations.loc[gdf_stations['nom_est'].isin(selected_stations)][['nom_est', 'latitud_geo', 'longitud_geo']], on='nom_est')

                years_with_anomalies = sorted(df_anomalias_anual['año'].unique().astype(int))
                if years_with_anomalies:
                    year_to_map = st.slider("Seleccione un año para visualizar en el mapa:", min_value=min(years_with_anomalies), max_value=max(years_with_anomalies), value=max(years_with_anomalies))
                    df_map_anom = df_anomalias_anual[df_anomalias_anual['año'] == str(year_to_map)]

                    max_abs_anom = df_anomalias_anual['anomalia'].abs().max()

                    fig_anom_map = px.scatter_geo(
                        df_map_anom,
                        lat='latitud_geo', lon='longitud_geo',
                        color='anomalia',
                        size=df_map_anom['anomalia'].abs(),
                        hover_name='nom_est',
                        hover_data={'anomalia': ':.0f'},
                        color_continuous_scale='RdBu',
                        range_color=[-max_abs_anom, max_abs_anom],
                        title=f"Anomalía de Precipitación Anual para el año {year_to_map}"
                    )
                    fig_anom_map.update_geos(fitbounds="locations", visible=True)
                    st.plotly_chart(fig_anom_map, use_container_width=True)

            with anom_fase_tab:
                df_anomalias_enso = df_anomalias.dropna(subset=['anomalia_oni']).copy()
                conditions = [df_anomalias_enso['anomalia_oni'] >= 0.5, df_anomalias_enso['anomalia_oni'] <= -0.5]
                phases = ['El Niño', 'La Niña']
                df_anomalias_enso['enso_fase'] = np.select(conditions, phases, default='Neutral')
                
                fig_box = px.box(df_anomalias_enso, x='enso_fase', y='anomalia', color='enso_fase',
                                 title="Distribución de Anomalías de Precipitación por Fase ENSO",
                                 labels={'anomalia': 'Anomalía de Precipitación (mm)', 'enso_fase': 'Fase ENSO'},
                                 points='all')
                st.plotly_chart(fig_box, use_container_width=True)

            with anom_extremos_tab:
                st.subheader("Eventos Mensuales Extremos (Basado en Anomalías)")
                df_extremos = df_anomalias.dropna(subset=['anomalia']).copy()
                df_extremos['fecha'] = df_extremos['fecha_mes_año'].dt.strftime('%Y-%m')
                
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown("##### 10 Meses más Secos")
                    secos = df_extremos.nsmallest(10, 'anomalia')[['fecha', 'nom_est', 'anomalia', 'precipitation', 'precip_promedio_mes']]
                    st.dataframe(secos.rename(columns={'nom_est': 'Estación', 'anomalia': 'Anomalía (mm)', 'precipitation': 'Ppt. (mm)', 'precip_promedio_mes': 'Ppt. Media (mm)'}).round(0), use_container_width=True)
                with col2:
                    st.markdown("##### 10 Meses más Húmedos")
                    humedos = df_extremos.nlargest(10, 'anomalia')[['fecha', 'nom_est', 'anomalia', 'precipitation', 'precip_promedio_mes']]
                    st.dataframe(humedos.rename(columns={'nom_est': 'Estación', 'anomalia': 'Anomalía (mm)', 'precipitation': 'Ppt. (mm)', 'precip_promedio_mes': 'Ppt. Media (mm)'}).round(0), use_container_width=True)

with estadisticas_tab:
    st.header("Estadísticas de Precipitación")
    if not selected_stations:
        st.warning("Por favor, seleccione al menos una estación para ver esta sección.")
    else:
        matriz_tab, resumen_mensual_tab, sintesis_tab = st.tabs(["Matriz de Disponibilidad", "Resumen Mensual", "Síntesis General"])

        with matriz_tab:
            st.subheader("Matriz de Disponibilidad de Datos Anual")
            original_data_counts = df_long[df_long['nom_est'].isin(selected_stations)]
            original_data_counts = original_data_counts.groupby(['nom_est', 'año']).size().reset_index(name='count')
            original_data_counts['porc_original'] = (original_data_counts['count'] / 12) * 100
            heatmap_original_df = original_data_counts.pivot(index='nom_est', columns='año', values='porc_original')

            heatmap_df = heatmap_original_df
            color_scale = "Greens"
            title_text = "Disponibilidad Promedio de Datos Originales"

            if analysis_mode == "Completar series (interpolación)":
                view_mode = st.radio("Seleccione la vista de la matriz:", ("Porcentaje de Datos Originales", "Porcentaje de Datos Completados"), horizontal=True)

                if view_mode == "Porcentaje de Datos Completados":
                    completed_data = df_monthly_to_process[(df_monthly_to_process['nom_est'].isin(selected_stations)) & (df_monthly_to_process['origen'] == 'Completado')]
                    if not completed_data.empty:
                        completed_counts = completed_data.groupby(['nom_est', 'año']).size().reset_index(name='count')
                        completed_counts['porc_completado'] = (completed_counts['count'] / 12) * 100
                        heatmap_df = completed_counts.pivot(index='nom_est', columns='año', values='porc_completado')
                        color_scale = "Reds"
                        title_text = "Disponibilidad Promedio de Datos Completados"
                    else:
                        heatmap_df = pd.DataFrame()

            if not heatmap_df.empty:
                avg_availability = heatmap_df.stack().mean()
                
                logo_col, metric_col = st.columns([1, 5])
                with logo_col:
                    if os.path.exists(logo_gota_path):
                        st.image(logo_gota_path, width=50)
                with metric_col:
                    st.metric(label=title_text, value=f"{avg_availability:.1f}%")

                styled_df = heatmap_df.style.background_gradient(cmap=color_scale, axis=None, vmin=0, vmax=100).format("{:.0f}%", na_rep="-").set_table_styles([
                    {'selector': 'th', 'props': [('background-color', '#333'), ('color', 'white'), ('font-size', '14px')]},
                    {'selector': 'td', 'props': [('text-align', 'center')]}
                ])
                st.dataframe(styled_df, use_container_width=True)
            else:
                st.info("No hay datos para mostrar en la matriz con la selección actual.")

        with resumen_mensual_tab:
            st.subheader("Resumen de Estadísticas Mensuales por Estación")
            summary_data = []
            for station_name, group in df_monthly_filtered.groupby('nom_est'):
                max_row = group.loc[group['precipitation'].idxmax()]
                min_row = group.loc[group['precipitation'].idxmin()]
                summary_data.append({
                    "Estación": station_name,
                    "Ppt. Máxima Mensual (mm)": max_row['precipitation'],
                    "Fecha Máxima": max_row['fecha_mes_año'].strftime('%Y-%m'),
                    "Ppt. Mínima Mensual (mm)": min_row['precipitation'],
                    "Fecha Mínima": min_row['fecha_mes_año'].strftime('%Y-%m'),
                    "Promedio Mensual (mm)": group['precipitation'].mean()
                })
            summary_df = pd.DataFrame(summary_data)
            st.dataframe(summary_df.round(0), use_container_width=True)

        with sintesis_tab:
            st.subheader("Síntesis General de Precipitación")
            if not df_monthly_filtered.empty and not df_anual_melted.empty:
                max_annual_row = df_anual_melted.loc[df_anual_melted['precipitacion'].idxmax()]
                
                # Obtener la fila con la precipitación mensual máxima
                max_monthly_row = df_monthly_filtered.loc[df_monthly_filtered['precipitation'].idxmax()]
                
                # Crear la columna 'nom_mes' para mostrar el nombre del mes
                meses_map = {1: 'Enero', 2: 'Febrero', 3: 'Marzo', 4: 'Abril', 5: 'Mayo', 6: 'Junio', 7: 'Julio', 8: 'Agosto', 9: 'Septiembre', 10: 'Octubre', 11: 'Noviembre', 12: 'Diciembre'}
                max_monthly_row['nom_mes'] = meses_map.get(max_monthly_row['mes'])

                col1, col2 = st.columns(2)
                with col1:
                    st.metric(
                        "Máxima Ppt. Anual Registrada",
                        f"{max_annual_row['precipitacion']:.0f} mm",
                        f"{max_annual_row['nom_est']} (Año {max_annual_row['año']})"
                    )
                with col2:
                    st.metric(
                        "Máxima Ppt. Mensual Registrada",
                        f"{max_monthly_row['precipitation']:.0f} mm",
                        f"{max_monthly_row['nom_est']} ({max_monthly_row['nom_mes']} {max_monthly_row['fecha_mes_año'].year})"
                    )

with enso_tab:
    st.header("Análisis de Precipitación y el Fenómeno ENSO")
    enso_series_tab, enso_anim_tab = st.tabs(["Series de Tiempo ENSO", "Mapa Animado ENSO"])

    with enso_series_tab:
        if df_enso.empty:
            st.warning("No se encontraron datos del fenómeno ENSO en el archivo de precipitación cargado.")
        else:
            enso_vars_available = {
                'anomalia_oni': 'Anomalía ONI',
                'temp_sst': 'Temp. Superficial del Mar (SST)',
                'temp_media': 'Temp. Media'
            }
            available_tabs = [name for var, name in enso_vars_available.items() if var in df_enso.columns]
            
            if not available_tabs:
                st.warning("No hay variables ENSO disponibles en el archivo de datos para visualizar.")
            else:
                enso_variable_tabs = st.tabs(available_tabs)
                for i, var_name in enumerate(available_tabs):
                    with enso_variable_tabs[i]:
                        var_code = [code for code, name in enso_vars_available.items() if name == var_name][0]
                        df_enso_filtered = df_enso[(df_enso['fecha_mes_año'].dt.year >= year_range[0]) & (df_enso['fecha_mes_año'].dt.year <= year_range[1]) & (df_enso['fecha_mes_año'].dt.month.isin(meses_numeros))]
                        if not df_enso_filtered.empty and var_code in df_enso_filtered.columns and not df_enso_filtered[var_code].isnull().all():
                            fig_enso_series = px.line(df_enso_filtered, x='fecha_mes_año', y=var_code, title=f"Serie de Tiempo para {var_name}")
                            st.plotly_chart(fig_enso_series, use_container_width=True)
                        else:
                            st.warning(f"No hay datos disponibles para '{var_code}' en el período seleccionado.")

    with enso_anim_tab:
        st.subheader("Evolución Mensual del Fenómeno ENSO")
        if not df_enso.empty and not gdf_stations.empty:
            controls_col, map_col = st.columns([1, 3])
            enso_anim_data = df_enso[['fecha_mes_año', 'anomalia_oni']].copy()
            enso_anim_data.dropna(subset=['anomalia_oni'], inplace=True)
            conditions = [enso_anim_data['anomalia_oni'] >= 0.5, enso_anim_data['anomalia_oni'] <= -0.5]
            phases = ['El Niño', 'La Niña']
            enso_anim_data['fase'] = np.select(conditions, phases, default='Neutral')
            enso_anim_data_filtered = enso_anim_data[(enso_anim_data['fecha_mes_año'].dt.year >= year_range[0]) & (enso_anim_data['fecha_mes_año'].dt.year <= year_range[1])]
            
            with controls_col:
                st.markdown("##### Estadísticas ENSO")
                logo_col, info_col = st.columns([1, 4])
                with logo_col:
                    if os.path.exists(logo_gota_path):
                        st.image(logo_gota_path, width=40)
                with info_col:
                    st.metric("Total Meses Analizados", len(enso_anim_data_filtered))
                
                phase_counts = enso_anim_data_filtered['fase'].value_counts()
                nino_months = phase_counts.get('El Niño', 0)
                nina_months = phase_counts.get('La Niña', 0)

                enso_events_df = enso_anim_data_filtered[enso_anim_data_filtered['fase'] != 'Neutral']
                if not enso_events_df.empty:
                    try:
                        locale.setlocale(locale.LC_TIME, 'es_ES.UTF-8')
                    except locale.Error:
                        locale.setlocale(locale.LC_TIME, '')
                    enso_events_df.loc[:, 'mes_nombre'] = enso_events_df['fecha_mes_año'].dt.strftime('%B').str.capitalize()
                    most_frequent_month = enso_events_df['mes_nombre'].mode()[0]
                else:
                    most_frequent_month = "N/A"

                st.info(f"""
                **Meses en Fase 'El Niño':** {nino_months}
                **Meses en Fase 'La Niña':** {nina_months}
                **Mes más frecuente para eventos:** {most_frequent_month}
                """)

            with map_col:
                stations_subset = gdf_stations[['nom_est', 'latitud_geo', 'longitud_geo']]
                enso_anim_data_filtered.loc[:, 'fecha_str'] = enso_anim_data_filtered['fecha_mes_año'].dt.strftime('%Y-%m')
                enso_anim_data_filtered.loc[:, 'key'] = 1
                stations_subset['key'] = 1
                animation_df = pd.merge(stations_subset, enso_anim_data_filtered, on='key').drop('key', axis=1)

                fig_enso_anim = px.scatter_geo(
                    animation_df, lat='latitud_geo', lon='longitud_geo',
                    color='fase', animation_frame='fecha_str',
                    hover_name='nom_est',
                    color_discrete_map={'El Niño': 'red', 'La Niña': 'blue', 'Neutral': 'lightgrey'},
                    category_orders={"fase": ["El Niño", "La Niña", "Neutral"]},
                    projection='natural earth'
                )
                fig_enso_anim.update_geos(fitbounds="locations", visible=True)
                fig_enso_anim.update_layout(
                    height=700,
                    title="Fase ENSO por Mes en las Estaciones Seleccionadas",
                    sliders=[dict(currentvalue=dict(font=dict(size=24, color="#707070"), prefix='<b>Fecha: </b>', visible=True))],
                    legend=dict(font=dict(size=16), title_font_size=18, itemsizing='constant')
                )
                st.plotly_chart(fig_enso_anim, use_container_width=True)

with tendencias_tab:
    st.header("Análisis de Tendencias y Pronósticos")
    if not selected_stations:
        st.warning("Por favor, seleccione al menos una estación para ver esta sección.")
    else:
        tendencia_tab, pronostico_tab = st.tabs(["Análisis de Tendencia", "Pronóstico SARIMA"])

        with tendencia_tab:
            st.subheader("Tendencia de Precipitación Anual")
            
            analysis_type = st.radio("Tipo de Análisis de Tendencia:", ["Promedio de la selección", "Estación individual"], horizontal=True)

            df_to_analyze = None
            if analysis_type == "Promedio de la selección":
                df_to_analyze = df_anual_melted.groupby('año')['precipitacion'].mean().reset_index()
            else:
                station_to_analyze = st.selectbox("Seleccione una estación para analizar:", options=selected_stations)
                if station_to_analyze:
                    df_to_analyze = df_anual_melted[df_anual_melted['nom_est'] == station_to_analyze]
            
            if df_to_analyze is not None and len(df_to_analyze) > 2:
                df_to_analyze['año_num'] = pd.to_numeric(df_to_analyze['año'])
                slope, intercept, r_value, p_value, std_err = stats.linregress(df_to_analyze['año_num'], df_to_analyze['precipitacion'])
                
                tendencia_texto = "aumentando" if slope > 0 else "disminuyendo"
                significancia_texto = "**estadísticamente significativa**" if p_value < 0.05 else "no es estadísticamente significativa"
                st.markdown(f"La tendencia de la precipitación es de **{slope:.2f} mm/año** (es decir, está {tendencia_texto}). Con un valor p de **{p_value:.3f}**, esta tendencia **{significancia_texto}**.")
                
                df_to_analyze['tendencia'] = slope * df_to_analyze['año_num'] + intercept

                fig_tendencia = px.scatter(df_to_analyze, x='año_num', y='precipitacion', title='Tendencia de la Precipitación Anual')
                fig_tendencia.add_trace(go.Scatter(x=df_to_analyze['año_num'], y=df_to_analyze['tendencia'], mode='lines', name='Línea de Tendencia', line=dict(color='red')))
                fig_tendencia.update_layout(xaxis_title="Año", yaxis_title="Precipitación Anual (mm)")
                st.plotly_chart(fig_tendencia, use_container_width=True)
            else:
                st.warning("No hay suficientes datos en el período seleccionado para calcular una tendencia.")

        with pronostico_tab:
            st.subheader("Pronóstico de Precipitación Mensual (Modelo SARIMA)")
            
            station_to_forecast = st.selectbox(
                "Seleccione una estación para el pronóstico:",
                options=selected_stations,
                help="El pronóstico se realiza para una única serie de tiempo."
            )
            
            forecast_horizon = st.slider("Meses a pronosticar:", 12, 36, 12, step=12)

            if st.button("Generar Pronóstico"):
                with st.spinner("Entrenando modelo y generando pronóstico... Esto puede tardar un momento."):
                    try:
                        ts_data = df_monthly_to_process[df_monthly_to_process['nom_est'] == station_to_forecast][['fecha_mes_año', 'precipitation']].copy()
                        ts_data = ts_data.set_index('fecha_mes_año').sort_index()
                        ts_data = ts_data['precipitation'].asfreq('MS')

                        model = sm.tsa.statespace.SARIMAX(ts_data, order=(1, 1, 1), seasonal_order=(1, 1, 1, 12), enforce_stationarity=False, enforce_invertibility=False)
                        results = model.fit(disp=False)
                        
                        forecast = results.get_forecast(steps=forecast_horizon)
                        forecast_mean = forecast.predicted_mean
                        forecast_ci = forecast.conf_int()

                        fig_pronostico = go.Figure()
                        fig_pronostico.add_trace(go.Scatter(x=ts_data.index, y=ts_data, mode='lines', name='Datos Históricos'))
                        fig_pronostico.add_trace(go.Scatter(x=forecast_mean.index, y=forecast_mean, mode='lines', name='Pronóstico', line=dict(color='red', dash='dash')))
                        fig_pronostico.add_trace(go.Scatter(x=forecast_ci.index, y=forecast_ci.iloc[:, 0], fill=None, mode='lines', line=dict(color='rgba(255,0,0,0.2)'), showlegend=False))
                        fig_pronostico.add_trace(go.Scatter(x=forecast_ci.index, y=forecast_ci.iloc[:, 1], fill='tonexty', mode='lines', line=dict(color='rgba(255,0,0,0.2)'), name='Intervalo de Confianza'))
                        
                        fig_pronostico.update_layout(title=f"Pronóstico de Precipitación para {station_to_forecast}", xaxis_title="Fecha", yaxis_title="Precipitación (mm)")
                        st.plotly_chart(fig_pronostico, use_container_width=True)
                        st.info("Este pronóstico se basa en modelos estadísticos (SARIMA) que identifican patrones históricos y estacionales en los datos. Los resultados son probabilísticos y deben ser interpretados según el grado de incertidumbre.")

                    except Exception as e:
                        st.error(f"No se pudo generar el pronóstico para '{station_to_forecast}'. El modelo estadístico no pudo converger. Esto puede ocurrir si la serie de datos es demasiado corta o inestable. Error: {e}")

with descargas_tab:
    st.header("Opciones de Descarga")
    if not selected_stations:
        st.warning("Por favor, seleccione al menos una estación para activar las descargas.")
    else:
        @st.cache_data
        def convert_df_to_csv(df):
            return df.to_csv(index=False).encode('utf-8')

        st.markdown("**Datos de Precipitación Anual (Filtrados)**")
        csv_anual = convert_df_to_csv(df_anual_melted)
        st.download_button("Descargar CSV Anual", csv_anual, 'precipitacion_anual.csv', 'text/csv', key='download-anual')

        st.markdown("**Datos de Precipitación Mensual (Filtrados)**")
        csv_mensual = convert_df_to_csv(df_monthly_filtered)
        st.download_button("Descargar CSV Mensual", csv_mensual, 'precipitacion_mensual.csv', 'text/csv', key='download-mensual')

        if analysis_mode == "Completar series (interpolación)":
            st.markdown("**Datos de Precipitación Mensual (Series Completadas y Filtradas)**")
            st.download_button("Descargar CSV con Series Completadas", csv_mensual, 'precipitacion_mensual_completada.csv', 'text/csv', key='download-completado')
        else:
            st.info("Para descargar las series completadas, seleccione la opción 'Completar series (interpolación)' en el panel lateral.")
