# --- Importaciones ---
import streamlit as st
import pandas as pd
import altair as alt
import folium
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

try:
    from folium.plugins import ScaleControl
except ImportError:
    class ScaleControl:
        def __init__(self, *args, **kwargs): pass
        def add_to(self, m): pass

# --- Configuración de la página ---
st.set_page_config(layout="wide", page_title="Visor de Precipitación y ENSO")

# --- CSS para optimizar el espacio ---
st.markdown("""
<style>
div.block-container {padding-top: 2rem;}
.sidebar .sidebar-content {font-size: 13px; }
h1 { margin-top: 0px; padding-top: 0px; }
</style>
""", unsafe_allow_html=True)

# --- Funciones de Carga y Procesamiento ---
@st.cache_data
def load_data(file_path, sep=';'):
    if file_path is None: return None
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
            df = pd.read_csv(io.BytesIO(content), sep=sep, encoding=encoding)
            df.columns = df.columns.str.strip()
            return df
        except Exception:
            continue
    st.error("No se pudo decodificar el archivo.")
    return None

@st.cache_data
def load_shapefile(file_path):
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
            gdf.columns = gdf.columns.str.strip()
            if gdf.crs is None:
                gdf.set_crs("EPSG:9377", inplace=True)
            return gdf.to_crs("EPSG:4326")
    except Exception as e:
        st.error(f"Error al procesar el shapefile: {e}")
        return None

@st.cache_data
def complete_series(df):
    all_completed_dfs = []
    station_list = df['Nom_Est'].unique()
    progress_bar = st.progress(0, text="Completando series...")
    for i, station in enumerate(station_list):
        df_station = df[df['Nom_Est'] == station].copy()
        df_station['Fecha'] = pd.to_datetime(df_station['Fecha'])
        df_station.set_index('Fecha', inplace=True)
        if not df_station.index.is_unique:
            df_station = df_station[~df_station.index.duplicated(keep='first')]
        df_resampled = df_station.resample('MS').asfreq()
        df_resampled['Precipitation'] = df_resampled['Precipitation'].interpolate(method='time')
        df_resampled['Nom_Est'] = station
        df_resampled['Año'] = df_resampled.index.year
        df_resampled['mes'] = df_resampled.index.month
        df_resampled.reset_index(inplace=True)
        all_completed_dfs.append(df_resampled)
        progress_bar.progress((i + 1) / len(station_list), text=f"Completando series... Estación: {station}")
    progress_bar.empty()
    return pd.concat(all_completed_dfs, ignore_index=True)

def create_enso_chart(enso_data):
    if enso_data.empty or 'anomalia_oni' not in enso_data.columns:
        return go.Figure()

    fig = go.Figure()
    
    for i, row in enso_data.iterrows():
        oni_value = row['anomalia_oni']
        color = 'rgba(200, 200, 200, 0.3)' # Neutral
        if oni_value >= 0.5:
            color = 'rgba(255, 100, 100, 0.3)' # El Niño
        elif oni_value <= -0.5:
            color = 'rgba(100, 100, 255, 0.3)' # La Niña
        
        fig.add_vrect(x0=row['Fecha'], x1=row['Fecha'] + pd.DateOffset(months=1), 
                      fillcolor=color, layer="below", line_width=0)

    fig.add_trace(go.Scatter(x=enso_data['Fecha'], y=enso_data['anomalia_oni'],
                             mode='lines', name='Anomalía ONI', line=dict(color='black', width=2)))

    fig.add_hline(y=0.5, line_dash="dash", line_color="red", annotation_text="El Niño")
    fig.add_hline(y=-0.5, line_dash="dash", line_color="blue", annotation_text="La Niña")

    fig.update_layout(
        title="Fases del Fenómeno ENSO y Anomalía ONI",
        yaxis_title="Anomalía ONI (°C)",
        xaxis_title="Fecha",
        showlegend=False
    )
    return fig

# --- Interfaz y Carga de Archivos ---
st.title('Visor de Precipitación y Fenómeno ENSO')
st.sidebar.header("Panel de Control")
with st.sidebar.expander("**Cargar Archivos**", expanded=True):
    uploaded_file_mapa = st.file_uploader("1. Cargar archivo de estaciones (mapaCVENSO.csv)", type="csv")
    uploaded_file_precip = st.file_uploader("2. Cargar archivo de precipitación mensual y ENSO (DatosPptnmes_ENSO.csv)", type="csv")
    uploaded_zip_shapefile = st.file_uploader("3. Cargar shapefile de municipios (.zip)", type="zip")

if not all([uploaded_file_mapa, uploaded_file_precip, uploaded_zip_shapefile]):
    st.info("Por favor, suba los 3 archivos requeridos para habilitar la aplicación.")
    st.stop()

# --- Carga y Preprocesamiento de Datos ---
df_precip_anual = load_data(uploaded_file_mapa)
df_precip_mensual_raw = load_data(uploaded_file_precip)
gdf_municipios = load_shapefile(uploaded_zip_shapefile)
if any(df is None for df in [df_precip_anual, df_precip_mensual_raw, gdf_municipios]):
    st.stop()
    
df_precip_mensual = df_precip_mensual_raw.copy()
df_precip_mensual.columns = df_precip_mensual.columns.str.strip().str.lower()
year_col_precip = next((col for col in df_precip_mensual.columns if ('año' in col or 'ano' in col) and 'enso' not in col), None)
if not year_col_precip:
    st.error(f"No se encontró columna de año principal ('año' o 'ano') en el archivo de precipitación mensual.")
    st.stop()
df_precip_mensual.rename(columns={year_col_precip: 'año'}, inplace=True)

enso_cols_base = ['año', 'mes', 'anomalia_oni', 'temp_media', 'temp_sst']
enso_cols_present = [col for col in enso_cols_base if col in df_precip_mensual.columns]
df_enso = pd.DataFrame() 
if 'año' in enso_cols_present and 'mes' in enso_cols_present:
    df_enso = df_precip_mensual[enso_cols_present].drop_duplicates().copy()
    df_enso.dropna(subset=['año', 'mes'], inplace=True)
    df_enso['año'] = pd.to_numeric(df_enso['año'], errors='coerce')
    df_enso['mes'] = pd.to_numeric(df_enso['mes'], errors='coerce')
    df_enso.dropna(subset=['año', 'mes'], inplace=True)
    df_enso = df_enso.astype({'año': int, 'mes': int})
    df_enso['Fecha'] = pd.to_datetime(df_enso['año'].astype(str) + '-' + df_enso['mes'].astype(str), errors='coerce')
    df_enso['fecha_merge'] = df_enso['Fecha'].dt.strftime('%Y-%m')
    df_enso.dropna(subset=['Fecha'], inplace=True)
    for col in ['anomalia_oni', 'temp_sst', 'temp_media']:
        if col in df_enso.columns:
            df_enso[col] = pd.to_numeric(df_enso[col].astype(str).str.replace(',', '.'), errors='coerce')

lon_col = next((col for col in df_precip_anual.columns if 'longitud' in col.lower() or 'lon' in col.lower()), None)
lat_col = next((col for col in df_precip_anual.columns if 'latitud' in col.lower() or 'lat' in col.lower()), None)
if not all([lon_col, lat_col]):
    st.error(f"No se encontraron las columnas de longitud y/o latitud en el archivo de estaciones.")
    st.stop()
df_precip_anual[lon_col] = pd.to_numeric(df_precip_anual[lon_col].astype(str).str.replace(',', '.'), errors='coerce')
df_precip_anual[lat_col] = pd.to_numeric(df_precip_anual[lat_col].astype(str).str.replace(',', '.'), errors='coerce')
df_precip_anual.dropna(subset=[lon_col, lat_col], inplace=True)
gdf_temp = gpd.GeoDataFrame(df_precip_anual, geometry=gpd.points_from_xy(df_precip_anual[lon_col], df_precip_anual[lat_col]), crs="EPSG:9377")
gdf_stations = gdf_temp.to_crs("EPSG:4326")
gdf_stations['Longitud_geo'] = gdf_stations.geometry.x
gdf_stations['Latitud_geo'] = gdf_stations.geometry.y

station_cols = [col for col in df_precip_mensual.columns if col.isdigit()]
if not station_cols:
    st.error("No se encontraron columnas de estación (ej: '12345') en el archivo de precipitación mensual.")
    st.stop()
id_vars = ['año', 'mes']
df_long = df_precip_mensual.melt(id_vars=id_vars, value_vars=station_cols, var_name='Id_estacion', value_name='Precipitation')
df_long['Precipitation'] = pd.to_numeric(df_long['Precipitation'].astype(str).str.replace(',', '.'), errors='coerce')
df_long.dropna(subset=['Precipitation'], inplace=True)
df_long['Fecha'] = pd.to_datetime(df_long['año'].astype(str) + '-' + df_long['mes'].astype(str), errors='coerce')
df_long.dropna(subset=['Fecha'], inplace=True)

gdf_stations['Id_estacio'] = gdf_stations['Id_estacio'].astype(str).str.strip()
df_long['Id_estacion'] = df_long['Id_estacion'].astype(str).str.strip()
station_mapping = gdf_stations.set_index('Id_estacio')['Nom_Est'].to_dict()
df_long['Nom_Est'] = df_long['Id_estacion'].map(station_mapping)
df_long.dropna(subset=['Nom_Est'], inplace=True)
df_long.rename(columns={'año': 'Año'}, inplace=True)
if df_long.empty:
    st.warning("El dataframe de precipitación mensual está vacío después del preprocesamiento.")
    st.stop()

# --- Controles en la Barra Lateral ---
st.sidebar.markdown("### Filtros de Visualización")
municipios_list = sorted(gdf_stations['municipio'].unique())
celdas_list = sorted(gdf_stations['Celda_XY'].unique())
selected_municipios = st.sidebar.multiselect('1. Filtrar por Municipio', options=municipios_list)
selected_celdas = st.sidebar.multiselect('2. Filtrar por Celda_XY', options=celdas_list)
stations_available = gdf_stations.copy()
if selected_municipios:
    stations_available = stations_available[stations_available['municipio'].isin(selected_municipios)]
if selected_celdas:
    stations_available = stations_available[stations_available['Celda_XY'].isin(selected_celdas)]
stations_options = sorted(stations_available['Nom_Est'].unique())
select_all = st.sidebar.checkbox("Seleccionar/Deseleccionar Todas las Estaciones", value=False)

if select_all:
    default_selection = stations_options
else:
    if 'selected_stations' in st.session_state and st.session_state.selected_stations:
        default_selection = st.session_state.selected_stations
    elif stations_options:
        default_selection = [stations_options[0]]
    else:
        default_selection = []
        
selected_stations = st.sidebar.multiselect('3. Seleccionar Estaciones', options=stations_options, default=default_selection)
st.session_state.selected_stations = selected_stations

años_disponibles = sorted([int(col) for col in gdf_stations.columns if str(col).isdigit()])
if not años_disponibles:
    st.error("No se encontraron columnas de años (ej: '2020', '2021') en el archivo de estaciones.")
    st.stop()
year_range = st.sidebar.slider("4. Seleccionar Rango de Años", min(años_disponibles), max(años_disponibles), (min(años_disponibles), max(años_disponibles)))
meses_dict = {'Enero': 1, 'Febrero': 2, 'Marzo': 3, 'Abril': 4, 'Mayo': 5, 'Junio': 6, 'Julio': 7, 'Agosto': 8, 'Septiembre': 9, 'Octubre': 10, 'Noviembre': 11, 'Diciembre': 12}
meses_nombres = st.sidebar.multiselect("5. Seleccionar Meses", list(meses_dict.keys()), default=list(meses_dict.keys()))
meses_numeros = [meses_dict[m] for m in meses_nombres]
st.sidebar.markdown("### Opciones de Análisis Avanzado")
analysis_mode = st.sidebar.radio("Análisis de Series Mensuales", ("Usar datos originales", "Completar series (interpolación)"))
if not selected_stations or not meses_numeros:
    st.warning("Por favor, seleccione al menos una estación y un mes.")
    st.stop()
df_monthly_for_analysis = df_long.copy()
if analysis_mode == "Completar series (interpolación)":
    df_monthly_for_analysis = complete_series(df_monthly_for_analysis[df_monthly_for_analysis['Nom_Est'].isin(selected_stations)])

# --- Preparación de datos filtrados ---
df_anual_melted = gdf_stations[gdf_stations['Nom_Est'].isin(selected_stations)].melt(
    id_vars=['Nom_Est', 'Longitud_geo', 'Latitud_geo'],
    value_vars=[str(y) for y in range(year_range[0], year_range[1] + 1) if str(y) in gdf_stations.columns],
    var_name='Año', value_name='Precipitación')
df_monthly_filtered = df_monthly_for_analysis[
    (df_
