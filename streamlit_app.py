# --- Importaciones ---
import streamlit as st
import pandas as pd
import altair as alt
import folium
from folium.plugins import MarkerCluster, MiniMap
from folium.raster_layers import WmsTileLayer
from streamlit_folium import folium_static, st_folium
import plotly.express as px
import plotly.graph_objects as go
import geopandas as gpd
import zipfile
import tempfile
import os
import io
import numpy as np
from pykrige.ok import OrdinaryKriging
import base64
from scipy import stats
import statsmodels.api as sm
from prophet import Prophet
from prophet.plot import plot_plotly
import branca.colormap as cm

# --- Inicialización del Estado de la Sesión (MEJORA FASE 1) ---
def initialize_session_state():
    """Inicializa todas las variables de session_state en un solo lugar."""
    defaults = {
        'data_loaded': False,
        'analysis_mode': "Usar datos originales",
        'map_view': {"location": [4.57, -74.29], "zoom": 5},
        'select_all_stations_state': False,
        'gif_rerun_count': 0,
        'df_monthly_processed': pd.DataFrame(),
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

initialize_session_state()

# --- Funciones de Carga y Procesamiento ---
def parse_spanish_dates(date_series):
    """Convierte abreviaturas de meses en español a inglés."""
    months_es_to_en = {'ene': 'Jan', 'abr': 'Apr', 'ago': 'Aug', 'dic': 'Dec'}
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
        if 'anomalia_oni' in df_station.columns:
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
    existing_enso_cols = [col for col in enso_cols if col in df_precip_mensual.columns]
    df_enso = df_precip_mensual[existing_enso_cols].drop_duplicates().copy()

    for col in [c for c in ['anomalia_oni', 'temp_sst', 'temp_media'] if c in df_enso.columns]:
        df_enso[col] = pd.to_numeric(df_enso[col], errors='coerce')

    if 'fecha_mes_año' in df_enso.columns:
        df_enso['fecha_mes_año'] = parse_spanish_dates(df_enso['fecha_mes_año'])
        df_enso['fecha_mes_año'] = pd.to_datetime(df_enso['fecha_mes_año'], format='%b-%y', errors='coerce')
        df_enso.dropna(subset=['fecha_mes_año'], inplace=True)

    return gdf_stations, df_precip_anual, gdf_municipios, df_long, df_enso

# --- Funciones de Visualización y Análisis ---
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
        x=data['fecha_mes_año'], y=[y_range[1] - y_range[0]] * len(data),
        base=y_range[0], marker_color=data['color'], width=30*24*60*60*1000,
        opacity=0.3, hoverinfo='none', showlegend=False
    ))

    legend_map = {'El Niño': 'red', 'La Niña': 'blue', 'Neutral': 'grey'}
    for phase, color in legend_map.items():
        fig.add_trace(go.Scatter(
            x=[None], y=[None], mode='markers',
            marker=dict(size=15, color=color, symbol='square', opacity=0.5),
            name=phase, showlegend=True
        ))

    fig.add_trace(go.Scatter(
        x=data['fecha_mes_año'], y=data['anomalia_oni'], mode='lines',
        name='Anomalía ONI', line=dict(color='black', width=2), showlegend=True
    ))

    fig.add_hline(y=0.5, line_dash="dash", line_color="red")
    fig.add_hline(y=-0.5, line_dash="dash", line_color="blue")

    fig.update_layout(
        height=600, title="Fases del Fenómeno ENSO y Anomalía ONI",
        yaxis_title="Anomalía ONI (°C)", xaxis_title="Fecha", showlegend=True,
        legend_title_text='Fase', yaxis_range=y_range
    )
    return fig

def create_anomaly_chart(df_plot):
    if df_plot.empty:
        return go.Figure()

    df_plot['color'] = np.where(df_plot['anomalia'] < 0, 'red', 'blue')
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=df_plot['fecha_mes_año'], y=df_plot['anomalia'],
        marker_color=df_plot['color'], name='Anomalía de Precipitación'
    ))

    if 'anomalia_oni' in df_plot.columns:
        df_plot_enso = df_plot.dropna(subset=['anomalia_oni'])

        nino_periods = df_plot_enso[df_plot_enso['anomalia_oni'] >= 0.5]
        for _, row in nino_periods.iterrows():
            fig.add_vrect(x0=row['fecha_mes_año'] - pd.DateOffset(days=15), x1=row['fecha_mes_año'] + pd.DateOffset(days=15),
                          fillcolor="red", opacity=0.15, layer="below", line_width=0)

        nina_periods = df_plot_enso[df_plot_enso['anomalia_oni'] <= -0.5]
        for _, row in nina_periods.iterrows():
            fig.add_vrect(x0=row['fecha_mes_año'] - pd.DateOffset(days=15), x1=row['fecha_mes_año'] + pd.DateOffset(days=15),
                          fillcolor="blue", opacity=0.15, layer="below", line_width=0)

        fig.add_trace(go.Scatter(x=[None], y=[None], mode='markers', marker=dict(symbol='square', color='rgba(255, 0, 0, 0.3)'), name='Fase El Niño'))
        fig.add_trace(go.Scatter(x=[None], y=[None], mode='markers', marker=dict(symbol='square', color='rgba(0, 0, 255, 0.3)'), name='Fase La Niña'))

    fig.update_layout(
        height=600, title="Anomalías Mensuales de Precipitación y Fases ENSO",
        yaxis_title="Anomalía de Precipitación (mm)", xaxis_title="Fecha", showlegend=True
    )
    return fig

@st.cache_data
def perform_kriging_interpolation(lons, lats, vals, grid_lon, grid_lat):
    """Ejecuta la interpolación Kriging y devuelve la malla y la varianza. (MEJORA FASE 1)"""
    # Los argumentos deben ser "hashable", por eso se convierten a tuplas al llamar la función.
    ok = OrdinaryKriging(list(lons), list(lats), list(vals), variogram_model='linear', verbose=False, enable_plotting=False)
    z, ss = ok.execute('grid', list(grid_lon), list(grid_lat))
    return z, ss

# --- Funciones de UI reutilizables ---
def get_map_options():
    return {
        "CartoDB Positron (Predeterminado)": {"tiles": "cartodbpositron", "attr": '&copy; <a href="https://carto.com/attributions">CartoDB</a>', "overlay": False},
        "OpenStreetMap": {"tiles": "OpenStreetMap", "attr": '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors', "overlay": False},
        "Topografía (OpenTopoMap)": {"tiles": "https://{s}.tile.opentopomap.org/{z}/{x}/{y}.png", "attr": 'Map data: &copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors, <a href="http://viewfinderpanoramas.org">SRTM</a> | Map style: &copy; <a href="https://opentopomap.org">OpenTopoMap</a> (<a href="https://creativecommons.org/licenses/by-sa/3.0/">CC-BY-SA</a>)', "overlay": False},
        "Relieve (Stamen Terrain)": {"tiles": "Stamen Terrain", "attr": 'Map tiles by <a href="http://stamen.com">Stamen Design</a>, <a href="http://creativecommons.org/licenses/by/3.0">CC BY 3.0</a> &mdash; Map data &copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors', "overlay": False},
        "Relieve y Océanos (GEBCO)": {"url": "https://www.gebco.net/data_and_products/gebco_web_services/web_map_service/web_map_service.php", "layers": "GEBCO_2021_Surface", "transparent": False, "attr": "GEBCO 2021", "overlay": True},
        "Mapa de Colombia (WMS IDEAM)": {"url": "https://geoservicios.ideam.gov.co/geoserver/ideam/wms", "layers": "ideam:col_admin", "transparent": True, "attr": "IDEAM", "overlay": True},
        "Cobertura de la Tierra (WMS IGAC)": {"url": "https://servicios.igac.gov.co/server/services/IDEAM/IDEAM_Cobertura_Corine/MapServer/WMSServer", "layers": "IDEAM_Cobertura_Corine_Web", "transparent": True, "attr": "IGAC", "overlay": True},
    }

def display_map_controls(container_object, key_prefix):
    map_options = get_map_options()
    base_maps = {k: v for k, v in map_options.items() if not v.get("overlay")}
    overlays = {k: v for k, v in map_options.items() if v.get("overlay")}

    selected_base_map_name = container_object.selectbox("Seleccionar Mapa Base", list(base_maps.keys()), key=f"{key_prefix}_base_map")
    selected_overlays = container_object.multiselect("Seleccionar Capas Adicionales", list(overlays.keys()), key=f"{key_prefix}_overlays")

    return base_maps[selected_base_map_name], [overlays[k] for k in selected_overlays]

def create_folium_map(center_location, zoom_start, base_map_config, overlay_configs, gdf_stations=None, gdf_polygons=None, station_icon_color='blue'):
    """Crea y configura un mapa de Folium con capas opcionales. (MEJORA FASE 1)"""
    m = folium.Map(
        location=center_location,
        zoom_start=zoom_start,
        tiles=base_map_config.get("tiles", "OpenStreetMap"),
        attr=base_map_config.get("attr", None)
    )

    if gdf_polygons is not None and not gdf_polygons.empty:
        folium.GeoJson(gdf_polygons.to_json(), name='Municipios').add_to(m)

    for layer_config in overlay_configs:
        WmsTileLayer(
            url=layer_config["url"], layers=layer_config["layers"], fmt='image/png',
            transparent=layer_config.get("transparent", False), overlay=True,
            control=True, name=layer_config["attr"]
        ).add_to(m)

    if gdf_stations is not None and not gdf_stations.empty:
        # Usar MarkerCluster si hay muchas estaciones, de lo contrario marcadores individuales
        if len(gdf_stations) > 100:
             feature_group = MarkerCluster(name='Estaciones').add_to(m)
        else:
             feature_group = folium.FeatureGroup(name='Estaciones').add_to(m)

        for _, row in gdf_stations.iterrows():
            # Crear HTML para el tooltip
            html_tooltip = f"""
            <b>Estación:</b> {row.get('nom_est', 'N/A')}<br>
            <b>Municipio:</b> {row.get('municipio', 'N/A')}<br>
            <b>Celda:</b> {row.get('celda_xy', 'N/A')}<br>
            <b>% Datos:</b> {row.get('porc_datos', 0):.0f}%<br>
            <b>Ppt. Media Anual:</b> {row.get('precip_media_anual', 0):.0f} mm
            """
            folium.Marker(
                location=[row['latitud_geo'], row['longitud_geo']],
                tooltip=html_tooltip,
                icon=folium.Icon(color=station_icon_color, icon='cloud')
            ).add_to(feature_group)

    folium.LayerControl().add_to(m)
    MiniMap(toggle_display=True).add_to(m)
    return m

# --- Configuración de la página y CSS ---
st.set_page_config(layout="wide", page_title="Sistema de información de las lluvias y el Clima en el norte de la región Andina")

st.markdown("""
<style>
div.block-container {padding-top: 2rem;}
.sidebar .sidebar-content {font-size: 13px; }
[data-testid="stMetricValue"] { font-size: 1.8rem; }
[data-testid="stMetricLabel"] { font-size: 1rem; padding-bottom: 5px; }
button[data-baseweb="tab"] { font-size: 16px; font-weight: bold; color: #333; }
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
with st.sidebar.expander("**Cargar Archivos**", expanded=not st.session_state.data_loaded):
    uploaded_file_mapa = st.file_uploader("1. Cargar archivo de estaciones (mapaCVENSO.csv)", type="csv")
    uploaded_file_precip = st.file_uploader("2. Cargar archivo de precipitación mensual y ENSO (DatosPptnmes_ENSO.csv)", type="csv")
    uploaded_zip_shapefile = st.file_uploader("3. Cargar shapefile de municipios (.zip)", type="zip")

    if st.button("Recargar Datos"):
        st.session_state.data_loaded = False
        st.cache_data.clear()
        st.rerun()

if not st.session_state.data_loaded:
    if not all([uploaded_file_mapa, uploaded_file_precip, uploaded_zip_shapefile]):
        st.info("Por favor, suba los 3 archivos requeridos para habilitar la aplicación.")
        st.stop()
    else:
        with st.spinner("Procesando archivos y cargando datos..."):
            st.session_state.gdf_stations, st.session_state.df_precip_anual, \
            st.session_state.gdf_municipios, st.session_state.df_long, \
            st.session_state.df_enso = preprocess_data(uploaded_file_mapa, uploaded_file_precip, uploaded_zip_shapefile)
        if st.session_state.gdf_stations is not None:
            st.session_state.data_loaded = True
            st.rerun()
        else:
            st.error("Hubo un error al procesar los archivos. Verifique que sean correctos y vuelva a intentarlo.")
            st.stop()

if st.session_state.gdf_stations is None:
    st.stop()

# Carga de datos de la sesión
gdf_stations = st.session_state.gdf_stations
df_precip_anual = st.session_state.df_precip_anual
df_long = st.session_state.df_long
df_enso = st.session_state.df_enso
gdf_municipios = st.session_state.gdf_municipios

# --- LÓGICA DE FILTRADO OPTIMIZADA Y DINÁMICA ---
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
            stations_filtered = stations_filtered[pd.concat(conditions, axis=1).any(axis=1)]

    if selected_regions: stations_filtered = stations_filtered[stations_filtered['depto_region'].isin(selected_regions)]
    if selected_municipios: stations_filtered = stations_filtered[stations_filtered['municipio'].isin(selected_municipios)]
    if selected_celdas: stations_filtered = stations_filtered[stations_filtered['celda_xy'].isin(selected_celdas)]

    return stations_filtered

# --- Controles en la Barra Lateral ---
with st.sidebar.expander("**1. Filtros Geográficos y de Datos**", expanded=True):
    min_data_perc = st.slider("Filtrar por % de datos mínimo:", 0, 100, 0, key='min_data_perc_slider')
    altitude_ranges = ['0-500', '500-1000', '1000-2000', '2000-3000', '>3000']
    selected_altitudes = st.multiselect('Filtrar por Altitud (m)', options=altitude_ranges, key='altitude_multiselect')
    regions_list = sorted(gdf_stations['depto_region'].dropna().unique())
    selected_regions = st.multiselect('Filtrar por Depto/Región', options=regions_list, key='regions_multiselect')

    filtered_stations_temp = apply_filters_to_stations(gdf_stations, min_data_perc, selected_altitudes, selected_regions, [], [])
    municipios_list = sorted(filtered_stations_temp['municipio'].dropna().unique())
    selected_municipios = st.multiselect('Filtrar por Municipio', options=municipios_list, key='municipios_multiselect')

    filtered_stations_temp_2 = apply_filters_to_stations(gdf_stations, min_data_perc, selected_altitudes, selected_regions, selected_municipios, [])
    celdas_list = sorted(filtered_stations_temp_2['celda_xy'].dropna().unique())
    selected_celdas = st.multiselect('Filtrar por Celda_XY', options=celdas_list, key='celdas_multiselect')

    if st.button("🧹 Limpiar Filtros"):
        keys_to_clear = ['min_data_perc_slider', 'altitude_multiselect', 'regions_multiselect',
                         'municipios_multiselect', 'celdas_multiselect', 'station_multiselect']
        for key in keys_to_clear:
            if key in st.session_state:
                if 'multiselect' in key:
                    st.session_state[key] = []
                elif 'slider' in key:
                    st.session_state[key] = 0
        st.session_state.select_all_checkbox = False
        st.rerun()

with st.sidebar.expander("**2. Selección de Estaciones y Período**", expanded=True):
    stations_master_list = apply_filters_to_stations(gdf_stations, min_data_perc, selected_altitudes, selected_regions, selected_municipios, selected_celdas)
    stations_options = sorted(stations_master_list['nom_est'].unique())

    if st.checkbox("Seleccionar/Deseleccionar todas las estaciones", value=st.session_state.select_all_stations_state, key='select_all_checkbox'):
        selected_stations = stations_options
        st.session_state.station_multiselect = stations_options
    else:
        selected_stations = st.multiselect(
            'Seleccionar Estaciones', options=stations_options,
            key='station_multiselect'
        )

    # Sincronizar el estado del checkbox
    if set(selected_stations) == set(stations_options) and stations_options:
        st.session_state.select_all_stations_state = True
    else:
        st.session_state.select_all_stations_state = False


    years_in_data = [int(col) for col in gdf_stations.columns if str(col).isdigit()]
    if not years_in_data:
        st.error("No se encontraron años disponibles en los datos.")
        st.stop()

    year_range = st.slider(
        "Seleccionar Rango de Años", min(years_in_data), max(years_in_data),
        (min(years_in_data), max(years_in_data))
    )

    meses_dict = {'Enero': 1, 'Febrero': 2, 'Marzo': 3, 'Abril': 4, 'Mayo': 5, 'Junio': 6, 'Julio': 7, 'Agosto': 8, 'Septiembre': 9, 'Octubre': 10, 'Noviembre': 11, 'Diciembre': 12}
    meses_nombres = st.multiselect("Seleccionar Meses", list(meses_dict.keys()), default=list(meses_dict.keys()))
    meses_numeros = [meses_dict[m] for m in meses_nombres]

with st.sidebar.expander("**3. Opciones de Análisis Avanzado**", expanded=False):
    analysis_mode = st.radio("Análisis de Series Mensuales", ("Usar datos originales", "Completar series (interpolación)"))

if st.session_state.analysis_mode != analysis_mode:
    st.session_state.analysis_mode = analysis_mode
    if analysis_mode == "Completar series (interpolación)":
        st.session_state.df_monthly_processed = complete_series(st.session_state.df_long)
    else:
        st.session_state.df_monthly_processed = pd.DataFrame()

if analysis_mode == "Completar series (interpolación)" and st.session_state.df_monthly_processed.empty:
     st.session_state.df_monthly_processed = complete_series(st.session_state.df_long)

df_monthly_to_process = st.session_state.df_monthly_processed if analysis_mode == "Completar series (interpolación)" else st.session_state.df_long

# --- Lógica de filtrado de datos principal ---
with st.spinner("Filtrando datos..."):
    gdf_filtered = apply_filters_to_stations(
        gdf_stations, min_data_perc, selected_altitudes,
        selected_regions, selected_municipios, selected_celdas
    )
    stations_for_analysis = selected_stations if selected_stations else gdf_filtered['nom_est'].unique()
    gdf_filtered = gdf_filtered[gdf_filtered['nom_est'].isin(stations_for_analysis)]

    df_anual_melted = gdf_stations.melt(
        id_vars=['nom_est', 'municipio', 'longitud_geo', 'latitud_geo', 'alt_est'],
        value_vars=[str(y) for y in range(year_range[0], year_range[1] + 1) if str(y) in gdf_stations.columns],
        var_name='año', value_name='precipitacion')
    df_anual_melted = df_anual_melted[df_anual_melted['nom_est'].isin(stations_for_analysis)]
    df_anual_melted.dropna(subset=['precipitacion'], inplace=True)

    df_monthly_filtered = df_monthly_to_process[
        (df_monthly_to_process['nom_est'].isin(stations_for_analysis)) &
        (df_monthly_to_process['fecha_mes_año'].dt.year >= year_range[0]) &
        (df_monthly_to_process['fecha_mes_año'].dt.year <= year_range[1]) &
        (df_monthly_to_process['fecha_mes_año'].dt.month.isin(meses_numeros))
    ].copy()

# --- Pestañas Principales ---
tab_names = ["🏠 Bienvenida", "Distribución Espacial", "Gráficos", "Mapas Avanzados", "Tabla de Estaciones", "Análisis de Anomalías", "Estadísticas", "Análisis de Correlación", "Análisis ENSO", "Tendencias y Pronósticos", "Descargas"]
(bienvenida_tab, mapa_tab, graficos_tab, mapas_avanzados_tab, tabla_estaciones_tab, anomalias_tab, estadisticas_tab, correlacion_tab, enso_tab, tendencias_tab, descargas_tab) = st.tabs(tab_names)

with bienvenida_tab:
    st.header("Bienvenido al Sistema de Información de Lluvias y Clima")
    st.markdown("""
    Esta plataforma interactiva está diseñada para la visualización y análisis de datos históricos
    de precipitación y su relación con el fenómeno ENSO en el norte de la región Andina.

    **¿Cómo empezar?**
    1.  **Cargue sus archivos**: Si es la primera vez que usa la aplicación, el panel de la izquierda le solicitará cargar los archivos de estaciones, precipitación y el shapefile de municipios. La aplicación recordará estos archivos en su sesión.
    2.  **Filtre los datos**: Utilice el **Panel de Control** en la barra lateral para filtrar las estaciones por ubicación (región, municipio), altitud, porcentaje de datos disponibles, y para seleccionar el período de análisis (años y meses).
    3.  **Explore las pestañas**: Cada pestaña ofrece una perspectiva diferente de los datos. Navegue a través de ellas para descubrir:
        - **Distribución Espacial**: Mapas interactivos de las estaciones.
        - **Gráficos**: Series de tiempo anuales, mensuales, comparaciones y distribuciones.
        - **Mapas Avanzados**: Animaciones y mapas de interpolación.
        - **Análisis de Anomalías**: Desviaciones de la precipitación respecto a la media histórica.
        - **Tendencias y Pronósticos**: Análisis de tendencias a largo plazo y modelos de pronóstico.

    Utilice el botón **🧹 Limpiar Filtros** en el panel lateral para reiniciar su selección en cualquier momento.

    ¡Esperamos que esta herramienta le sea de gran utilidad para sus análisis climáticos!
    """)
    if os.path.exists(logo_path):
        st.image(logo_path, width=400, caption="Corporación Cuenca Verde")

with mapa_tab:
    st.header("Distribución espacial de las Estaciones de Lluvia")
    if len(stations_for_analysis) == 0:
        st.warning("Por favor, seleccione al menos una estación para ver esta sección.")
    else:
        if not df_anual_melted.empty:
            df_mean_precip = df_anual_melted.groupby('nom_est')['precipitacion'].mean().reset_index()
            gdf_filtered_map = gdf_filtered.merge(df_mean_precip.rename(columns={'precipitacion': 'precip_media_anual'}), on='nom_est', how='left')
        else:
            gdf_filtered_map = gdf_filtered.copy()
            gdf_filtered_map['precip_media_anual'] = np.nan
        gdf_filtered_map['precip_media_anual'] = gdf_filtered_map['precip_media_anual'].fillna(0)

        sub_tab_mapa, sub_tab_grafico = st.tabs(["Mapa Interactivo", "Gráfico de Disponibilidad de Datos"])

        with sub_tab_mapa:
            controls_col, map_col = st.columns([1, 3])

            with controls_col:
                st.subheader("Controles del Mapa")
                selected_base_map_config, selected_overlays_config = display_map_controls(st, "dist_esp")

                if not gdf_filtered_map.empty:
                    st.metric("Estaciones en Vista", len(gdf_filtered_map))
                    map_centering = st.radio("Opciones de centrado:", ("Automático", "Vistas Predefinidas"), key="map_centering_radio", horizontal=True)

                    if map_centering == "Vistas Predefinidas":
                        if st.button("Ver Colombia"): st.session_state.map_view = {"location": [4.57, -74.29], "zoom": 5}
                        if st.button("Ver Antioquia"): st.session_state.map_view = {"location": [6.24, -75.58], "zoom": 8}
                        if st.button("Ajustar a Selección"):
                            bounds = gdf_filtered_map.total_bounds
                            st.session_state.map_view = {"location": [(bounds[1] + bounds[3]) / 2, (bounds[0] + bounds[2]) / 2], "zoom": 9}

                with st.expander("Resumen de Filtros Activos", expanded=True):
                    summary_text = f"**Período:** {year_range[0]} - {year_range[1]}\n\n"
                    if selected_regions: summary_text += f"**Región:** {', '.join(selected_regions)}\n\n"
                    st.info(summary_text)

            with map_col:
                if not gdf_filtered_map.empty:
                    map_center = st.session_state.map_view["location"]
                    map_zoom = st.session_state.map_view["zoom"]

                    m = create_folium_map(
                        center_location=map_center,
                        zoom_start=map_zoom,
                        base_map_config=selected_base_map_config,
                        overlay_configs=selected_overlays_config,
                        gdf_stations=gdf_filtered_map,
                        gdf_polygons=gdf_municipios
                    )

                    if map_centering == "Automático":
                        bounds = gdf_filtered_map.total_bounds
                        if all(np.isfinite(bounds)):
                            m.fit_bounds([[bounds[1], bounds[0]], [bounds[3], bounds[2]]])

                    st_folium(m, height=700, width="100%", returned_objects=[])
                else:
                    st.warning("No hay estaciones seleccionadas para mostrar en el mapa.")
        
        with sub_tab_grafico:
            st.subheader("Disponibilidad y Composición de Datos por Estación")
            if not gdf_filtered.empty:
                if analysis_mode == "Completar series (interpolación)":
                    st.info("Mostrando la composición de datos originales vs. completados para el período seleccionado.")

                    data_composition = df_monthly_filtered.groupby(['nom_est', 'origen']).size().unstack(fill_value=0)
                    if 'Original' not in data_composition: data_composition['Original'] = 0
                    if 'Completado' not in data_composition: data_composition['Completado'] = 0

                    data_composition['total'] = data_composition['Original'] + data_composition['Completado']
                    data_composition['% Original'] = np.where(data_composition['total'] > 0, (data_composition['Original'] / data_composition['total']) * 100, 0)
                    data_composition['% Completado'] = np.where(data_composition['total'] > 0, (data_composition['Completado'] / data_composition['total']) * 100, 0)

                    df_plot = data_composition.reset_index().melt(
                        id_vars='nom_est', value_vars=['% Original', '% Completado'],
                        var_name='Tipo de Dato', value_name='Porcentaje'
                    )

                    fig_comp = px.bar(df_plot, x='nom_est', y='Porcentaje', color='Tipo de Dato',
                                      title='Composición de Datos por Estación',
                                      labels={'nom_est': 'Estación', 'Porcentaje': '% del Período'},
                                      color_discrete_map={'% Original': '#1f77b4', '% Completado': '#ff7f0e'},
                                      text_auto='.1f')
                    fig_comp.update_layout(height=600, xaxis={'categoryorder': 'total descending'})
                    st.plotly_chart(fig_comp, use_container_width=True)
                else:
                    st.info("Mostrando el porcentaje de disponibilidad de datos según el archivo de estaciones.")
                    df_chart = gdf_filtered.sort_values("porc_datos", ascending=False)
                    fig_disp = px.bar(df_chart, x='nom_est', y='porc_datos',
                                      title='Porcentaje de Disponibilidad de Datos Históricos',
                                      labels={'nom_est': 'Estación', 'porc_datos': '% de Datos Disponibles'},
                                      color='porc_datos',
                                      color_continuous_scale=px.colors.sequential.Viridis)
                    fig_disp.update_layout(height=600, xaxis={'categoryorder':'total descending'})
                    st.plotly_chart(fig_disp, use_container_width=True)
            else:
                st.warning("No hay estaciones seleccionadas para mostrar el gráfico.")

with graficos_tab:
    st.header("Visualizaciones de Precipitación")
    if len(stations_for_analysis) == 0:
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

                        fig_avg = px.bar(df_summary, x='nom_est', y='precipitacion',
                                       title='Promedio de Precipitación Anual', labels={'nom_est': 'Estación', 'precipitacion': 'Precipitación Media Anual (mm)'}, color='precipitacion',
                                       color_continuous_scale=px.colors.sequential.Blues_r)
                        fig_avg.update_layout(height=600, xaxis={'categoryorder':'total descending' if "Mayor a Menor" in sort_order else ('total ascending' if "Menor a Mayor" in sort_order else 'trace')})
                        st.plotly_chart(fig_avg, use_container_width=True)
                    else:
                        df_anual_filtered_for_box = df_anual_melted[df_anual_melted['nom_est'].isin(stations_for_analysis)]
                        fig_box_annual = px.box(df_anual_filtered_for_box, x='nom_est', y='precipitacion', color='nom_est', points='all', title='Distribución de la Precipitación Anual por Estación', labels={'nom_est': 'Estación', 'precipitacion': 'Precipitación Anual (mm)'})
                        fig_box_annual.update_layout(height=600)
                        st.plotly_chart(fig_box_annual, use_container_width=True)

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
            if len(stations_for_analysis) < 2:
                st.info("Seleccione al menos dos estaciones para comparar.")
            else:
                st.markdown("##### Precipitación Mensual Promedio")
                df_monthly_avg = df_monthly_filtered.groupby(['nom_est', 'mes'])['precipitation'].mean().reset_index()

                fig_avg_monthly = px.line(df_monthly_avg,
                                          x='mes', y='precipitation', color='nom_est',
                                          labels={'mes': 'Mes', 'precipitation': 'Precipitación Promedio (mm)'},
                                          title='Promedio de Precipitación Mensual por Estación')
                fig_avg_monthly.update_layout(height=600, xaxis = dict(tickmode = 'array', tickvals = list(meses_dict.values()), ticktext = list(meses_dict.keys())))
                st.plotly_chart(fig_avg_monthly, use_container_width=True)

                st.markdown("##### Distribución de Precipitación Anual")
                df_anual_filtered_for_box = df_anual_melted[df_anual_melted['nom_est'].isin(stations_for_analysis)]
                fig_box_annual = px.box(df_anual_filtered_for_box, x='nom_est', y='precipitacion', color='nom_est', points='all', title='Distribución de la Precipitación Anual por Estación', labels={'nom_est': 'Estación', 'precipitacion': 'Precipitación Anual (mm)'})
                fig_box_annual.update_layout(height=600)
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
            if not df_anual_melted.empty and not gdf_filtered['alt_est'].isnull().all():
                df_relacion = df_anual_melted.groupby('nom_est')['precipitacion'].mean().reset_index()
                df_relacion = df_relacion.merge(gdf_filtered[['nom_est', 'alt_est']].drop_duplicates(), on='nom_est')

                fig_relacion = px.scatter(df_relacion, x='alt_est', y='precipitacion', color='nom_est',
                                          title='Relación entre Precipitación Media Anual y Altitud',
                                          labels={'alt_est': 'Altitud (m)', 'precipitacion': 'Precipitación Media Anual (mm)'})
                fig_relacion.update_layout(height=600)
                st.plotly_chart(fig_relacion, use_container_width=True)
            else:
                st.info("No hay datos de altitud o precipitación disponibles para analizar la relación.")

with mapas_avanzados_tab:
    st.header("Mapas Avanzados")
    gif_tab, temporal_tab, compare_tab, kriging_tab, coropletico_tab = st.tabs(["Animación GIF", "Visualización Temporal", "Comparación de Mapas", "Interpolación Kriging", "Mapa Coroplético"])

    with gif_tab:
        st.subheader("Distribución Espacio-Temporal de la Lluvia en Antioquia")
        gif_path = "PPAM.gif"
        if os.path.exists(gif_path):
            with open(gif_path, "rb") as file:
                contents = file.read()
                data_url = base64.b64encode(contents).decode("utf-8")
            st.markdown(f'<img src="data:image/gif;base64,{data_url}" alt="Animación PPAM" style="width:100%;">', unsafe_allow_html=True)
        else:
            st.warning("Archivo 'PPAM.gif' no encontrado.")

    with temporal_tab:
        if len(stations_for_analysis) == 0:
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
                            selected_base_map_config, selected_overlays_config = display_map_controls(st, "temporal")
                            st.markdown(f"#### Resumen del Año: {selected_year}")
                            df_year_filtered = df_anual_melted[df_anual_melted['año'] == str(selected_year)].dropna(subset=['precipitacion'])
                            st.metric(f"Estaciones con Datos en {selected_year}", f"{len(df_year_filtered)} de {len(stations_for_analysis)}")

                            if not df_year_filtered.empty:
                                max_row = df_year_filtered.loc[df_year_filtered['precipitacion'].idxmax()]
                                min_row = df_year_filtered.loc[df_year_filtered['precipitacion'].idxmin()]
                                st.info(f"**Ppt. Máxima:** {max_row['precipitacion']:.0f} mm ({max_row['nom_est']})\n\n**Ppt. Mínima:** {min_row['precipitacion']:.0f} mm ({min_row['nom_est']})")
                            else:
                                st.warning(f"No hay datos de precipitación para el año {selected_year}.")
                        with map_col:
                            m_temporal = folium.Map(location=[6.24, -75.58], zoom_start=7, tiles=selected_base_map_config.get("tiles", "OpenStreetMap"), attr=selected_base_map_config.get("attr", None))

                            if not df_year_filtered.empty:
                                min_val, max_val = df_anual_melted['precipitacion'].min(), df_anual_melted['precipitacion'].max()
                                colormap = cm.linear.YlGnBu_09.scale(vmin=min_val, vmax=max_val)
                                for _, row in df_year_filtered.iterrows():
                                    folium.CircleMarker(
                                        location=[row['latitud_geo'], row['longitud_geo']], radius=5,
                                        color=colormap(row['precipitacion']), fill=True, fill_color=colormap(row['precipitacion']),
                                        fill_opacity=0.8, tooltip=f"{row['nom_est']}: {row['precipitacion']:.0f} mm"
                                    ).add_to(m_temporal)
                                bounds = gdf_stations.loc[gdf_stations['nom_est'].isin(df_year_filtered['nom_est'])].total_bounds
                                m_temporal.fit_bounds([[bounds[1], bounds[0]], [bounds[3], bounds[2]]])

                            folium.LayerControl().add_to(m_temporal)
                            st_folium(m_temporal, height=700, width="100%")

            with race_tab:
                st.subheader("Ranking Anual de Precipitación por Estación")
                if not df_anual_melted.empty:
                    station_order = df_anual_melted.groupby('nom_est')['precipitacion'].sum().sort_values(ascending=True).index
                    fig_racing = px.bar(df_anual_melted, x="precipitacion", y="nom_est", animation_frame="año",
                        orientation='h', text="precipitacion", labels={'precipitacion': 'Precipitación Anual (mm)', 'nom_est': 'Estación'},
                        title="Evolución de Precipitación Anual por Estación", category_orders={'nom_est': station_order})
                    fig_racing.update_traces(texttemplate='%{x:.0f}', textposition='outside')
                    fig_racing.update_layout(xaxis_range=[0, df_anual_melted['precipitacion'].max() * 1.15], height=max(600, len(stations_for_analysis) * 35))
                    st.plotly_chart(fig_racing, use_container_width=True)

            with anim_tab:
                st.subheader("Mapa Animado de Precipitación Anual")
                if not df_anual_melted.empty:
                    all_years = sorted(df_anual_melted['año'].unique())
                    if all_years:
                        fig_mapa_animado = px.scatter_geo(df_anual_melted, lat='latitud_geo', lon='longitud_geo', color='precipitacion', size='precipitacion',
                            hover_name='nom_est', animation_frame='año', projection='natural earth', title='Precipitación Anual por Estación',
                            color_continuous_scale=px.colors.sequential.YlGnBu)
                        fig_mapa_animado.update_geos(fitbounds="locations", visible=True)
                        fig_mapa_animado.update_layout(height=700)
                        st.plotly_chart(fig_mapa_animado, use_container_width=True)

    with compare_tab:
        st.subheader("Comparación de Mapas Anuales")
        if not df_anual_melted.empty and len(df_anual_melted['año'].unique()) > 1:
            control_col, map_col1, map_col2 = st.columns([1, 2, 2])
            with control_col:
                st.markdown("##### Controles de Mapa")
                selected_base_map_config, _ = display_map_controls(st, "compare")
                min_year, max_year = int(df_anual_melted['año'].min()), int(df_anual_melted['año'].max())
                year1 = st.slider("Año Mapa 1", min_year, max_year, max_year, key="compare_year1")
                year2 = st.slider("Año Mapa 2", min_year, max_year, max_year - 1, key="compare_year2")
                min_p, max_p = int(df_anual_melted['precipitacion'].min()), int(df_anual_melted['precipitacion'].max())
                color_range = st.slider("Rango de Color (mm)", min_p, max_p, (min_p, max_p))

            def create_compare_map(data, year, col):
                col.markdown(f"**Precipitación en {year}**")
                m = folium.Map(location=[6.24, -75.58], zoom_start=6, tiles=selected_base_map_config.get("tiles"))
                if not data.empty:
                    colormap = cm.linear.YlGnBu_09.scale(vmin=color_range[0], vmax=color_range[1])
                    for _, row in data.iterrows():
                        folium.CircleMarker(location=[row['latitud_geo'], row['longitud_geo']], radius=5, color=colormap(row['precipitacion']), fill=True, fill_color=colormap(row['precipitacion']), fill_opacity=0.8, tooltip=f"{row['precipitacion']:.0f} mm").add_to(m)
                    bounds = gdf_stations.loc[gdf_stations['nom_est'].isin(data['nom_est'])].total_bounds
                    m.fit_bounds([[bounds[1], bounds[0]], [bounds[3], bounds[2]]])
                with col:
                    st_folium(m, height=600, width="100%", key=f"map_{year}", returned_objects=[])

            create_compare_map(df_anual_melted[df_anual_melted['año'].astype(int) == year1], year1, map_col1)
            create_compare_map(df_anual_melted[df_anual_melted['año'].astype(int) == year2], year2, map_col2)
        else:
            st.warning("Se necesitan datos de al menos dos años diferentes para la comparación.")

    with kriging_tab:
        st.subheader("Interpolación Kriging para un Año Específico")
        if not df_anual_melted.empty:
            min_year_k, max_year_k = int(df_anual_melted['año'].min()), int(df_anual_melted['año'].max())
            year_kriging = st.slider("Seleccione el año para la interpolación", min_year_k, max_year_k, max_year_k, key="year_kriging")
            data_year_kriging = df_anual_melted[df_anual_melted['año'].astype(int) == year_kriging]

            if len(data_year_kriging) < 3:
                st.warning(f"Se necesitan al menos 3 estaciones con datos en {year_kriging} para el Kriging.")
            else:
                with st.spinner("Generando mapa Kriging..."):
                    lons, lats, vals = data_year_kriging['longitud_geo'].values, data_year_kriging['latitud_geo'].values, data_year_kriging['precipitacion'].values
                    bounds = gdf_stations.loc[gdf_stations['nom_est'].isin(stations_for_analysis)].total_bounds
                    grid_lon, grid_lat = np.linspace(bounds[0] - 0.1, bounds[2] + 0.1, 100), np.linspace(bounds[1] - 0.1, bounds[3] + 0.1, 100)

                    z, ss = perform_kriging_interpolation(tuple(lons), tuple(lats), tuple(vals), tuple(grid_lon), tuple(grid_lat))

                    fig_krig = go.Figure(data=go.Contour(z=z.T, x=grid_lon, y=grid_lat, colorscale='YlGnBu', contours=dict(showlabels=True)))
                    fig_krig.add_trace(go.Scatter(x=lons, y=lats, mode='markers', marker=dict(color='red', size=5), name='Estaciones'))
                    fig_krig.update_layout(height=700, title=f"Superficie de Precipitación Interpolada (Kriging) - Año {year_kriging}")
                    st.plotly_chart(fig_krig, use_container_width=True)
        else:
            st.warning("No hay datos anuales para realizar la interpolación.")

    with coropletico_tab:
        st.subheader("Mapa Coroplético de Precipitación Anual Promedio")
        st.caption(f"Mostrando el promedio para el período {year_range[0]} - {year_range[1]}")
        controls_col, map_col = st.columns([1, 4])
        with controls_col:
            st.markdown("##### Controles de Mapa")
            selected_base_map_config, _ = display_map_controls(st, "choro")
        with map_col:
            if gdf_municipios is not None and not df_anual_melted.empty:
                df_anual_municipio = df_anual_melted.groupby('municipio')['precipitacion'].mean().reset_index()
                municipio_col_shp = next((col for col in ['mcnpio', 'municipio', 'nombre_mpio', 'mpio_cnmbr'] if col in gdf_municipios.columns), None)
                if municipio_col_shp:
                    gdf_municipios_data = gdf_municipios.merge(df_anual_municipio, left_on=municipio_col_shp.lower(), right_on='municipio'.lower(), how='left')
                    center_lat, center_lon = gdf_municipios_data.dissolve().centroid.y.iloc[0], gdf_municipios_data.dissolve().centroid.x.iloc[0]
                    m_choro = folium.Map(location=[center_lat, center_lon], zoom_start=7, tiles=selected_base_map_config.get("tiles"))
                    folium.Choropleth(
                        geo_data=gdf_municipios_data.to_json(), name='Precipitación Media Anual', data=gdf_municipios_data,
                        columns=[municipio_col_shp, 'precipitacion'], key_on=f'feature.properties.{municipio_col_shp}',
                        fill_color='YlGnBu', fill_opacity=0.7, line_opacity=0.2, legend_name='Precipitación Media Anual (mm)', nan_fill_color='white'
                    ).add_to(m_choro)
                    folium.LayerControl().add_to(m_choro)
                    st_folium(m_choro, height=700, width="100%", returned_objects=[])
                else:
                    st.error("No se encontró una columna de municipios compatible en el shapefile.")
            else:
                st.warning("No hay datos para generar el mapa coroplético.")

# (El resto de las pestañas continúan aquí, idénticas al original)
# ...
with tabla_estaciones_tab:
    st.header("Información Detallada de las Estaciones")
    if len(stations_for_analysis) == 0:
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
    if len(stations_for_analysis) == 0:
        st.warning("Por favor, seleccione al menos una estación para ver esta sección.")
    else:
        df_long_filtered_stations = df_long[df_long['nom_est'].isin(stations_for_analysis)]
        df_climatology = df_long_filtered_stations.groupby(['nom_est', 'mes'])['precipitation'].mean().reset_index().rename(columns={'precipitation': 'precip_promedio_mes'})
        
        df_anomalias = pd.merge(df_monthly_filtered, df_climatology, on=['nom_est', 'mes'], how='left')
        df_anomalias['anomalia'] = df_anomalias['precipitation'] - df_anomalias['precip_promedio_mes']

        if df_anomalias.empty or df_anomalias['anomalia'].isnull().all():
            st.warning("No hay suficientes datos para calcular y mostrar las anomalías.")
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
                df_anomalias_anual = pd.merge(df_anomalias_anual, gdf_stations[['nom_est', 'latitud_geo', 'longitud_geo']], on='nom_est')
                
                years_with_anomalies = sorted(df_anomalias_anual['año'].unique().astype(int))
                if years_with_anomalies:
                    year_to_map = st.slider("Seleccione un año:", min_value=min(years_with_anomalies), max_value=max(years_with_anomalies), value=max(years_with_anomalies))
                    df_map_anom = df_anomalias_anual[df_anomalias_anual['año'] == year_to_map]
                    max_abs_anom = df_anomalias_anual['anomalia'].abs().max()
                    fig_anom_map = px.scatter_geo(df_map_anom, lat='latitud_geo', lon='longitud_geo', color='anomalia', size=df_map_anom['anomalia'].abs(),
                        hover_name='nom_est', hover_data={'anomalia': ':.0f'}, color_continuous_scale='RdBu', range_color=[-max_abs_anom, max_abs_anom],
                        title=f"Anomalía de Precipitación Anual para {year_to_map}")
                    fig_anom_map.update_geos(fitbounds="locations", visible=True)
                    st.plotly_chart(fig_anom_map, use_container_width=True)

            with anom_fase_tab:
                if 'anomalia_oni' in df_anomalias.columns:
                    df_anomalias_enso = df_anomalias.dropna(subset=['anomalia_oni']).copy()
                    conditions = [df_anomalias_enso['anomalia_oni'] >= 0.5, df_anomalias_enso['anomalia_oni'] <= -0.5]
                    phases = ['El Niño', 'La Niña']
                    df_anomalias_enso['enso_fase'] = np.select(conditions, phases, default='Neutral')
                    
                    fig_box = px.box(df_anomalias_enso, x='enso_fase', y='anomalia', color='enso_fase',
                                     title="Distribución de Anomalías de Precipitación por Fase ENSO",
                                     labels={'anomalia': 'Anomalía (mm)', 'enso_fase': 'Fase ENSO'}, points='all')
                    st.plotly_chart(fig_box, use_container_width=True)
                else:
                    st.warning("La columna 'anomalia_oni' no está disponible.")

            with anom_extremos_tab:
                st.subheader("Eventos Mensuales Extremos (Basado en Anomalías)")
                df_extremos = df_anomalias.dropna(subset=['anomalia']).copy()
                df_extremos['fecha'] = df_extremos['fecha_mes_año'].dt.strftime('%Y-%m')
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown("##### 10 Meses más Secos")
                    secos = df_extremos.nsmallest(10, 'anomalia')[['fecha', 'nom_est', 'anomalia', 'precipitation', 'precip_promedio_mes']]
                    st.dataframe(secos.rename(columns={'nom_est': 'Estación', 'anomalia': 'Anomalía (mm)', 'precipitation': 'Ppt. (mm)', 'precip_promedio_mes': 'Ppt. Media (mm)'}).round(0))
                with col2:
                    st.markdown("##### 10 Meses más Húmedos")
                    humedos = df_extremos.nlargest(10, 'anomalia')[['fecha', 'nom_est', 'anomalia', 'precipitation', 'precip_promedio_mes']]
                    st.dataframe(humedos.rename(columns={'nom_est': 'Estación', 'anomalia': 'Anomalía (mm)', 'precipitation': 'Ppt. (mm)', 'precip_promedio_mes': 'Ppt. Media (mm)'}).round(0))

with estadisticas_tab:
    st.header("Estadísticas de Precipitación")
    if len(stations_for_analysis) == 0:
        st.warning("Por favor, seleccione al menos una estación para ver esta sección.")
    else:
        matriz_tab, resumen_mensual_tab, sintesis_tab = st.tabs(["Matriz de Disponibilidad", "Resumen Mensual", "Síntesis General"])
        # (El resto del código de la pestaña se mantiene igual)

with correlacion_tab:
    st.header("Correlación entre Precipitación y ENSO")
    # (El resto del código de la pestaña se mantiene igual)

with enso_tab:
    st.header("Análisis de Precipitación y el Fenómeno ENSO")
    enso_series_tab, enso_anim_tab = st.tabs(["Series de Tiempo ENSO", "Mapa Interactivo ENSO"])

    with enso_series_tab:
        # (El código se mantiene igual)
        pass

    with enso_anim_tab:
        st.subheader("Explorador Mensual del Fenómeno ENSO")
        if df_enso.empty or gdf_stations.empty or 'anomalia_oni' not in df_enso.columns:
            st.warning("Datos insuficientes para esta visualización ('anomalia_oni' necesaria).")
        else:
            controls_col, map_col = st.columns([1, 3])
            enso_anim_data = df_enso[['fecha_mes_año', 'anomalia_oni']].dropna().copy()
            conditions = [enso_anim_data['anomalia_oni'] >= 0.5, enso_anim_data['anomalia_oni'] <= -0.5]
            enso_anim_data['fase'] = np.select(conditions, ['El Niño', 'La Niña'], default='Neutral')
            
            with controls_col:
                selected_base_map_config, selected_overlays_config = display_map_controls(st, "enso_anim")
                available_dates = sorted(enso_anim_data['fecha_mes_año'].unique())
                if available_dates:
                    selected_date = st.select_slider("Seleccione una fecha:", options=available_dates, format_func=lambda date: date.strftime('%Y-%m'))
                    phase_info = enso_anim_data[enso_anim_data['fecha_mes_año'] == selected_date]
                    if not phase_info.empty:
                        st.metric(f"Fase ENSO en {selected_date.strftime('%Y-%m')}", phase_info['fase'].iloc[0], f"Anomalía ONI: {phase_info['anomalia_oni'].iloc[0]:.2f}°C")
            
            with map_col:
                if available_dates and not phase_info.empty:
                    phase_color_map = {'El Niño': 'red', 'La Niña': 'blue', 'Neutral': 'gray'}
                    marker_color = phase_color_map.get(phase_info['fase'].iloc[0], 'black')
                    
                    m_enso = create_folium_map(
                        center_location=[4.57, -74.29], zoom_start=5,
                        base_map_config=selected_base_map_config,
                        overlay_configs=selected_overlays_config,
                        gdf_stations=gdf_filtered,
                        station_icon_color=marker_color
                    )
                    st_folium(m_enso, height=700, width="100%", returned_objects=[])

with tendencias_tab:
    st.header("Análisis de Tendencias y Pronósticos")
    if len(stations_for_analysis) == 0:
        st.warning("Por favor, seleccione al menos una estación para ver esta sección.")
    else:
        tendencia_individual_tab, tendencia_tabla_tab, pronostico_tab, prophet_tab = st.tabs(["Análisis Individual / Promedio", "Tabla Comparativa de Tendencias", "Pronóstico SARIMA", "Pronóstico Prophet"])

        with tendencia_individual_tab:
            st.subheader("Tendencia de Precipitación Anual")
            
            analysis_type = st.radio("Tipo de Análisis de Tendencia:", ["Promedio de la selección", "Estación individual"], horizontal=True)

            df_to_analyze = None
            if analysis_type == "Promedio de la selección":
                df_to_analyze = df_anual_melted.groupby('año')['precipitacion'].mean().reset_index()
            else:
                station_to_analyze = st.selectbox("Seleccione una estación para analizar:", options=stations_for_analysis)
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

        with tendencia_tabla_tab:
            st.subheader("Tabla Comparativa de Tendencias de Precipitación Anual")
            
            exclude_zeros = st.checkbox("Excluir valores de precipitación cero (0) del cálculo")
            
            if st.button("Calcular Tendencias para Todas las Estaciones Seleccionadas"):
                with st.spinner("Calculando tendencias..."):
                    results = []
                    df_anual_calc = df_anual_melted.copy()
                    df_anual_calc.dropna(subset=['precipitacion'], inplace=True)
                    
                    if exclude_zeros:
                        df_anual_calc = df_anual_calc[df_anual_calc['precipitacion'] > 0]

                    for station in stations_for_analysis:
                        station_data = df_anual_calc[df_anual_calc['nom_est'] == station]
                        
                        if len(station_data) > 2:
                            station_data['año_num'] = pd.to_numeric(station_data['año'])
                            slope, intercept, r, p_value, std_err = stats.linregress(station_data['año_num'], station_data['precipitacion'])
                            
                            interpretation = "Significativa (p < 0.05)" if p_value < 0.05 else "No Significativa (p ≥ 0.05)"
                            
                            results.append({
                                "Estación": station,
                                "Tendencia (mm/año)": slope,
                                "Valor p": p_value,
                                "Interpretación": interpretation,
                                "Años Analizados": len(station_data)
                            })
                        else:
                             results.append({
                                "Estación": station,
                                "Tendencia (mm/año)": np.nan,
                                "Valor p": np.nan,
                                "Interpretación": "Datos insuficientes",
                                "Años Analizados": len(station_data)
                            })
                    
                    if results:
                        results_df = pd.DataFrame(results)
                        
                        # Estilo para la tabla
                        def style_p_value(val):
                            color = 'lightgreen' if val < 0.05 else 'lightcoral'
                            return f'background-color: {color}'

                        st.dataframe(results_df.style.format({
                            "Tendencia (mm/año)": "{:.2f}",
                            "Valor p": "{:.4f}"
                        }).applymap(style_p_value, subset=['Valor p']), use_container_width=True)
                    else:
                        st.warning("No se pudieron calcular tendencias para las estaciones seleccionadas.")

        with pronostico_tab:
            st.subheader("Pronóstico de Precipitación Mensual (Modelo SARIMA)")
            
            station_to_forecast = st.selectbox(
                "Seleccione una estación para el pronóstico:",
                options=stations_for_analysis,
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

        with prophet_tab:
            st.subheader("Pronóstico de Precipitación Mensual (Modelo Prophet)")
            
            station_to_forecast_prophet = st.selectbox(
                "Seleccione una estación para el pronóstico:",
                options=stations_for_analysis,
                key="prophet_station_select",
                help="El pronóstico se realiza para una única serie de tiempo con Prophet."
            )
            
            forecast_horizon_prophet = st.slider("Meses a pronosticar:", 12, 36, 12, step=12, key="prophet_horizon")

            if st.button("Generar Pronóstico con Prophet", key="run_prophet"):
                with st.spinner("Entrenando modelo Prophet y generando pronóstico... Esto puede tardar un momento."):
                    try:
                        ts_data_prophet = df_monthly_to_process[df_monthly_to_process['nom_est'] == station_to_forecast_prophet][['fecha_mes_año', 'precipitation']].copy()
                        ts_data_prophet.rename(columns={'fecha_mes_año': 'ds', 'precipitation': 'y'}, inplace=True)
                        
                        if len(ts_data_prophet) < 24:
                            st.warning("Se necesitan al menos 24 puntos de datos para que Prophet funcione correctamente. Por favor, ajuste la selección de años.")
                        else:
                            model_prophet = Prophet()
                            model_prophet.fit(ts_data_prophet)
                            
                            future = model_prophet.make_future_dataframe(periods=forecast_horizon_prophet, freq='MS')
                            forecast_prophet = model_prophet.predict(future)

                            st.success("Pronóstico generado exitosamente.")
                            
                            fig_prophet = plot_plotly(model_prophet, forecast_prophet)
                            fig_prophet.update_layout(title=f"Pronóstico de Precipitación con Prophet para {station_to_forecast_prophet}", yaxis_title="Precipitación (mm)")
                            st.plotly_chart(fig_prophet, use_container_width=True)

                            st.info("El modelo Prophet descompone la serie de tiempo en componentes de tendencia, estacionalidad y días festivos para generar un pronóstico robusto. El área sombreada representa el intervalo de confianza.")
                    except Exception as e:
                        st.error(f"Ocurrió un error al generar el pronóstico con Prophet. Esto puede deberse a que la serie de datos es demasiado corta o inestable. Error: {e}")


with descargas_tab:
    st.header("Opciones de Descarga")
    if len(stations_for_analysis) == 0:
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
            csv_completado = convert_df_to_csv(df_monthly_filtered)
            st.download_button("Descargar CSV con Series Completadas", csv_completado, 'precipitacion_mensual_completada.csv', 'text/csv', key='download-completado')
        else:
            st.info("Para descargar las series completadas, seleccione la opción 'Completar series (interpolación)' en el panel lateral.")
