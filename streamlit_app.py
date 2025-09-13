# -*- coding: utf-8 -*-
# Importaciones
# ---
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
from scipy import stats
import statsmodels.api as sm
from prophet import Prophet
from prophet.plot import plot_plotly
import branca.colormap as cm
import base64

# ---
# Constantes y Configuración Centralizada
# ---
class Config:
    # Nombres de Columnas de Datos
    STATION_NAME_COL = 'nom_est'
    PRECIPITATION_COL = 'precipitation'
    LATITUDE_COL = 'latitud_geo'
    LONGITUDE_COL = 'longitud_geo'
    YEAR_COL = 'año'
    MONTH_COL = 'mes'
    DATE_COL = 'fecha_mes_año'
    ENSO_ONI_COL = 'anomalia_oni'
    ORIGIN_COL = 'origen'
    ALTITUDE_COL = 'alt_est'
    MUNICIPALITY_COL = 'municipio'
    REGION_COL = 'depto_region'
    PERCENTAGE_COL = 'porc_datos'
    CELL_COL = 'celda_xy'

    # Índices climáticos leídos del archivo principal
    SOI_COL = 'soi'
    IOD_COL = 'iod'

    # Rutas de Archivos
    LOGO_PATH = "CuencaVerdeLogo_V1.JPG"
    LOGO_DROP_PATH = "CuencaVerdeGoticaLogo.JPG"
    GIF_PATH = "PPAM.gif"

    # Mensajes de la UI
    APP_TITLE = "Sistema de información de las lluvias y el Clima en el norte de la región Andina"
    WELCOME_TEXT = """
    Esta plataforma interactiva está diseñada para la visualización y análisis de datos históricos de precipitación y su
    relación con el fenómeno ENSO en el norte de la región Andina.
    
    **¿Cómo empezar?**
    1.  **Cargue sus archivos**: Si es la primera vez que usa la aplicación, el panel de la izquierda le solicitará cargar los archivos de estaciones,
    precipitación y el shapefile de municipios. La aplicación recordará estos archivos en su sesión.
    2.  **Filtre los datos**: Utilice el **Panel de Control** en la barra lateral para filtrar las estaciones por ubicación (región, municipio), altitud,
    porcentaje de datos disponibles, y para seleccionar el período de análisis (años y meses).
    3.  **Explore las pestañas**: Cada pestaña ofrece una perspectiva diferente de los datos. Navegue a través de ellas para descubrir:
        - **Distribución Espacial**: Mapas interactivos de las estaciones.
        - **Gráficos**: Series de tiempo anuales, mensuales, comparaciones y distribuciones.
        - **Mapas Avanzados**: Animaciones y mapas de interpolación.
        - **Análisis de Anomalías**: Desviaciones de la precipitación respecto a la media histórica.
        - **Tendencias y Pronósticos**: Análisis de tendencias a largo plazo y modelos de pronóstico.
    
    Utilice el botón **🧹 Limpiar Filtros** en el panel lateral para reiniciar su selección en cualquier momento.
    
    ¡Esperamos que esta herramienta le sea de gran utilidad para sus análisis climáticos!
    """
    
    @staticmethod
    def initialize_session_state():
        """Inicializa todas las variables necesarias en el estado de la sesión de Streamlit."""
        state_defaults = {
            'data_loaded': False,
            'analysis_mode': "Usar datos originales",
            'select_all_stations_state': False,
            'df_monthly_processed': pd.DataFrame(),
            'gdf_stations': None,
            'df_precip_anual': None,
            'gdf_municipios': None,
            'df_long': None,
            'df_enso': None,
            'min_data_perc_slider': 0,
            'altitude_multiselect': [],
            'regions_multiselect': [],
            'municipios_multiselect': [],
            'celdas_multiselect': [],
            'station_multiselect': [],
            'exclude_na': False,
            'exclude_zeros': False,
        }
        for key, value in state_defaults.items():
            if key not in st.session_state:
                st.session_state[key] = value

# ---
# Funciones de Carga y Preprocesamiento
# ---
@st.cache_data
def parse_spanish_dates(date_series):
    """Convierte abreviaturas de meses en español a inglés."""
    months_es_to_en = {'ene': 'Jan', 'abr': 'Apr', 'ago': 'Aug', 'dic': 'Dec'}
    date_series_str = date_series.astype(str).str.lower()
    for es, en in months_es_to_en.items():
        date_series_str = date_series_str.str.replace(es, en, regex=False)
    return pd.to_datetime(date_series_str, format='%b-%y', errors='coerce')

@st.cache_data
def load_csv_data(file_uploader_object, sep=';', lower_case=True):
    """Carga y decodifica un archivo CSV de manera robusta desde un objeto de Streamlit."""
    if file_uploader_object is None:
        return None
    try:
        content = file_uploader_object.getvalue()
        if not content.strip():
            st.error(f"El archivo '{file_uploader_object.name}' parece estar vacío.")
            return None
    except Exception as e:
        st.error(f"Error al leer el archivo '{file_uploader_object.name}': {e}")
        return None

    encodings_to_try = ['utf-8', 'latin1', 'cp1252', 'iso-8859-1']
    for encoding in encodings_to_try:
        try:
            df = pd.read_csv(io.BytesIO(content), sep=sep, encoding=encoding)
            df.columns = df.columns.str.strip().str.replace(';', '')
            if lower_case:
                df.columns = df.columns.str.lower()
            return df
        except Exception:
            continue
    st.error(f"No se pudo decodificar el archivo '{file_uploader_object.name}' con las codificaciones probadas.")
    return None

@st.cache_data
def load_shapefile(file_uploader_object):
    """Procesa y carga un shapefile desde un archivo .zip subido a Streamlit."""
    if file_uploader_object is None:
        return None
    try:
        with tempfile.TemporaryDirectory() as temp_dir:
            with zipfile.ZipFile(file_uploader_object, 'r') as zip_ref:
                zip_ref.extractall(temp_dir)
            
            shp_files = [f for f in os.listdir(temp_dir) if f.endswith('.shp')]
            if not shp_files:
                st.error("No se encontró un archivo .shp en el archivo .zip.")
                return None
            
            shp_path = os.path.join(temp_dir, shp_files[0])
            gdf = gpd.read_file(shp_path)
            gdf.columns = gdf.columns.str.strip().str.lower()
            
            if gdf.crs is None:
                st.warning("El shapefile no tiene un sistema de coordenadas de referencia (CRS) definido. Asumiendo MAGNA-SIRGAS (EPSG:9377).")
                gdf.set_crs("EPSG:9377", inplace=True)
            return gdf.to_crs("EPSG:4326")
    except Exception as e:
        st.error(f"Error al procesar el shapefile: {e}")
        return None

@st.cache_data
def complete_series(_df):
    """Completa las series de tiempo de precipitación usando interpolación lineal temporal."""
    all_completed_dfs = []
    station_list = _df[Config.STATION_NAME_COL].unique()
    progress_bar = st.progress(0, text="Completando todas las series...")
    
    for i, station in enumerate(station_list):
        df_station = _df[_df[Config.STATION_NAME_COL] == station].copy()
        df_station[Config.DATE_COL] = pd.to_datetime(df_station[Config.DATE_COL])
        df_station.set_index(Config.DATE_COL, inplace=True)
        
        if not df_station.index.is_unique:
            df_station = df_station[~df_station.index.duplicated(keep='first')]

        date_range = pd.date_range(start=df_station.index.min(), end=df_station.index.max(), freq='MS')
        df_resampled = df_station.reindex(date_range)
        
        df_resampled[Config.PRECIPITATION_COL] = df_resampled[Config.PRECIPITATION_COL].interpolate(method='time')
        
        df_resampled[Config.ORIGIN_COL] = df_resampled[Config.ORIGIN_COL].fillna('Completado')
        df_resampled[Config.STATION_NAME_COL] = station
        df_resampled[Config.YEAR_COL] = df_resampled.index.year
        df_resampled[Config.MONTH_COL] = df_resampled.index.month
        df_resampled.reset_index(inplace=True)
        df_resampled.rename(columns={'index': Config.DATE_COL}, inplace=True)
        all_completed_dfs.append(df_resampled)
        
        progress_bar.progress((i + 1) / len(station_list), text=f"Completando series... Estación: {station}")
    
    progress_bar.empty()
    return pd.concat(all_completed_dfs, ignore_index=True)

@st.cache_data
def load_and_process_all_data(uploaded_file_mapa, uploaded_file_precip, uploaded_zip_shapefile):
    """
    Carga y procesa todos los archivos de entrada y los fusiona en dataframes listos para usar.
    """
    df_stations_raw = load_csv_data(uploaded_file_mapa)
    df_precip_raw = load_csv_data(uploaded_file_precip)
    gdf_municipios = load_shapefile(uploaded_zip_shapefile)

    if any(df is None for df in [df_stations_raw, df_precip_raw, gdf_municipios]):
        return None, None, None, None

    # --- 1. Procesar Estaciones (gdf_stations) ---
    lon_col = next((col for col in df_stations_raw.columns if 'longitud' in col.lower() or 'lon' in col.lower()), None)
    lat_col = next((col for col in df_stations_raw.columns if 'latitud' in col.lower() or 'lat' in col.lower()), None)
    if not all([lon_col, lat_col]):
        st.error("No se encontraron columnas de longitud y/o latitud en el archivo de estaciones.")
        return None, None, None, None
    
    df_stations_raw[lon_col] = pd.to_numeric(df_stations_raw[lon_col].astype(str).str.replace(',', '.'), errors='coerce')
    df_stations_raw[lat_col] = pd.to_numeric(df_stations_raw[lat_col].astype(str).str.replace(',', '.'), errors='coerce')
    df_stations_raw.dropna(subset=[lon_col, lat_col], inplace=True)

    gdf_stations = gpd.GeoDataFrame(df_stations_raw,
                                    geometry=gpd.points_from_xy(df_stations_raw[lon_col], df_stations_raw[lat_col]),
                                    crs="EPSG:9377").to_crs("EPSG:4326")
    gdf_stations[Config.LONGITUDE_COL] = gdf_stations.geometry.x
    gdf_stations[Config.LATITUDE_COL] = gdf_stations.geometry.y
    if Config.ALTITUDE_COL in gdf_stations.columns:
        gdf_stations[Config.ALTITUDE_COL] = pd.to_numeric(gdf_stations[Config.ALTITUDE_COL].astype(str).str.replace(',', '.'), errors='coerce')

    # --- 2. Procesar Precipitación (df_long) ---
    station_id_cols = [col for col in df_precip_raw.columns if col.isdigit()]
    if not station_id_cols:
        st.error("No se encontraron columnas de estación (ej: '12345') en el archivo de precipitación mensual.")
        return None, None, None, None

    id_vars = [col for col in df_precip_raw.columns if not col.isdigit()]
    df_long = df_precip_raw.melt(id_vars=id_vars, value_vars=station_id_cols, 
                               var_name='id_estacion', value_name=Config.PRECIPITATION_COL)

    # Limpieza y conversión de tipos
    cols_to_numeric = [Config.ENSO_ONI_COL, 'temp_sst', 'temp_media', Config.PRECIPITATION_COL, Config.SOI_COL, Config.IOD_COL]
    for col in cols_to_numeric:
        if col in df_long.columns:
            df_long[col] = pd.to_numeric(df_long[col].astype(str).str.replace(',', '.'), errors='coerce')
    
    df_long.dropna(subset=[Config.PRECIPITATION_COL], inplace=True)
    df_long[Config.DATE_COL] = parse_spanish_dates(df_long[Config.DATE_COL])
    df_long.dropna(subset=[Config.DATE_COL], inplace=True)
    df_long[Config.ORIGIN_COL] = 'Original'

    # Mapeo de nombres de estación
    gdf_stations['id_estacio'] = gdf_stations['id_estacio'].astype(str).str.strip()
    df_long['id_estacion'] = df_long['id_estacion'].astype(str).str.strip()
    station_mapping = gdf_stations.set_index('id_estacio')[Config.STATION_NAME_COL].to_dict()
    df_long[Config.STATION_NAME_COL] = df_long['id_estacion'].map(station_mapping)
    df_long.dropna(subset=[Config.STATION_NAME_COL], inplace=True)

    # --- 3. Extraer datos ENSO para gráficos aislados ---
    enso_cols = ['id', Config.DATE_COL, Config.ENSO_ONI_COL, 'temp_sst', 'temp_media']
    existing_enso_cols = [col for col in enso_cols if col in df_precip_raw.columns]
    df_enso = df_precip_raw[existing_enso_cols].drop_duplicates().copy()
    
    if Config.DATE_COL in df_enso.columns:
        df_enso[Config.DATE_COL] = parse_spanish_dates(df_enso[Config.DATE_COL])
        df_enso.dropna(subset=[Config.DATE_COL], inplace=True)

    for col in [Config.ENSO_ONI_COL, 'temp_sst', 'temp_media']:
        if col in df_enso.columns:
            df_enso[col] = pd.to_numeric(df_enso[col].astype(str).str.replace(',', '.'), errors='coerce')

    return gdf_stations, gdf_municipios, df_long, df_enso

# ---
# Funciones para Gráficos y Mapas
# ---
def create_enso_chart(enso_data):
    if enso_data.empty or Config.ENSO_ONI_COL not in enso_data.columns:
        return go.Figure()

    data = enso_data.copy().sort_values(Config.DATE_COL)
    data.dropna(subset=[Config.ENSO_ONI_COL], inplace=True)

    conditions = [data[Config.ENSO_ONI_COL] >= 0.5, data[Config.ENSO_ONI_COL] <= -0.5]
    phases = ['El Niño', 'La Niña']
    colors = ['red', 'blue']
    data['phase'] = np.select(conditions, phases, default='Neutral')
    data['color'] = np.select(conditions, colors, default='grey')
    y_range = [data[Config.ENSO_ONI_COL].min() - 0.5, data[Config.ENSO_ONI_COL].max() + 0.5]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=data[Config.DATE_COL], y=[y_range[1] - y_range[0]] * len(data),
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
        x=data[Config.DATE_COL], y=data[Config.ENSO_ONI_COL],
        mode='lines', name='Anomalía ONI', line=dict(color='black', width=2), showlegend=True
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
        x=df_plot[Config.DATE_COL], y=df_plot['anomalia'],
        marker_color=df_plot['color'], name='Anomalía de Precipitación'
    ))
    if Config.ENSO_ONI_COL in df_plot.columns:
        df_plot_enso = df_plot.dropna(subset=[Config.ENSO_ONI_COL])
        nino_periods = df_plot_enso[df_plot_enso[Config.ENSO_ONI_COL] >= 0.5]
        for _, row in nino_periods.iterrows():
            fig.add_vrect(x0=row[Config.DATE_COL] - pd.DateOffset(days=15), x1=row[Config.DATE_COL] + pd.DateOffset(days=15),
                          fillcolor="red", opacity=0.15, layer="below", line_width=0)
        nina_periods = df_plot_enso[df_plot_enso[Config.ENSO_ONI_COL] <= -0.5]
        for _, row in nina_periods.iterrows():
            fig.add_vrect(x0=row[Config.DATE_COL] - pd.DateOffset(days=15), x1=row[Config.DATE_COL] + pd.DateOffset(days=15),
                          fillcolor="blue", opacity=0.15, layer="below", line_width=0)
        fig.add_trace(go.Scatter(x=[None], y=[None], mode='markers', marker=dict(symbol='square', color='rgba(255, 0, 0, 0.3)'), name='Fase El Niño'))
        fig.add_trace(go.Scatter(x=[None], y=[None], mode='markers', marker=dict(symbol='square', color='rgba(0, 0, 255, 0.3)'), name='Fase La Niña'))
    fig.update_layout(
        height=600, title="Anomalías Mensuales de Precipitación y Fases ENSO",
        yaxis_title="Anomalía de Precipitación (mm)", xaxis_title="Fecha", showlegend=True
    )
    return fig

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
    default_overlays = ["Mapa de Colombia (WMS IDEAM)"]
    selected_overlays = container_object.multiselect("Seleccionar Capas Adicionales", list(overlays.keys()), default=default_overlays, key=f"{key_prefix}_overlays")
    
    return base_maps[selected_base_map_name], [overlays[k] for k in selected_overlays]

# ---
# --- SECCIÓN DE FUNCIONES DE PESTAÑAS RESTAURADAS ---
# ---
def display_welcome_tab():
    st.header("Bienvenido al Sistema de Información de Lluvias y Clima")
    st.markdown(Config.WELCOME_TEXT, unsafe_allow_html=True)
    if os.path.exists(Config.LOGO_PATH):
        st.image(Config.LOGO_PATH, width=400, caption="Corporación Cuenca Verde")

def display_spatial_distribution_tab(gdf_filtered, df_anual_melted, stations_for_analysis):
    st.header("Distribución espacial de las Estaciones de Lluvia")
    
    if len(stations_for_analysis) == 0:
        st.warning("Por favor, seleccione al menos una estación para ver esta sección.")
        return
    
    selected_stations_str = f"{len(stations_for_analysis)} estaciones" if len(stations_for_analysis) > 1 else f"1 estación: {stations_for_analysis[0]}"
    st.info(f"Mostrando análisis para {selected_stations_str} en el período {st.session_state.year_range[0]} - {st.session_state.year_range[1]}.")

    if not df_anual_melted.empty:
        df_mean_precip = df_anual_melted.groupby(Config.STATION_NAME_COL)[Config.PRECIPITATION_COL].mean().reset_index()
        gdf_filtered_map = gdf_filtered.merge(df_mean_precip.rename(columns={Config.PRECIPITATION_COL: 'precip_media_anual'}), on=Config.STATION_NAME_COL, how='left')
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
                st.markdown("---")
                m1, m2 = st.columns([1, 3])
                with m1:
                    if os.path.exists(Config.LOGO_DROP_PATH):
                        st.image(Config.LOGO_DROP_PATH, width=50)
                with m2:
                    st.metric("Estaciones en Vista", len(gdf_filtered_map))
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
                        bounds = gdf_filtered_map.total_bounds
                        center_lat = (bounds[1] + bounds[3]) / 2
                        center_lon = (bounds[0] + bounds[2]) / 2
                        st.session_state.map_view = {"location": [center_lat, center_lon], "zoom": 9}
                st.markdown("---")
                with st.expander("Resumen de Filtros Activos", expanded=True):
                    summary_text = f"**Período:** {st.session_state.year_range[0]} - {st.session_state.year_range[1]}\n\n"
                    summary_text += f"**% Mínimo de Datos:** {st.session_state.min_data_perc_slider}%\n\n"
                    if st.session_state.selected_altitudes: summary_text += f"**Altitud:** {', '.join(st.session_state.selected_altitudes)}\n\n"
                    if st.session_state.selected_regions: summary_text += f"**Región:** {', '.join(st.session_state.selected_regions)}\n\n"
                    if st.session_state.selected_municipios: summary_text += f"**Municipio:** {', '.join(st.session_state.selected_municipios)}\n\n"
                    if st.session_state.selected_celdas: summary_text += f"**Celda XY:** {', '.join(st.session_state.selected_celdas)}\n\n"
                    st.info(summary_text)

        with map_col:
            if not gdf_filtered_map.empty:
                m = folium.Map(
                    location=st.session_state.map_view["location"],
                    zoom_start=st.session_state.map_view["zoom"],
                    tiles=selected_base_map_config.get("tiles", "OpenStreetMap"),
                    attr=selected_base_map_config.get("attr", None)
                )
                if map_centering == "Automático":
                    bounds = gdf_filtered_map.total_bounds
                    m.fit_bounds([[bounds[1], bounds[0]], [bounds[3], bounds[2]]])
                folium.GeoJson(st.session_state.gdf_municipios.to_json(), name='Municipios').add_to(m)
                for layer_config in selected_overlays_config:
                    folium.raster_layers.WmsTileLayer(
                        url=layer_config["url"], layers=layer_config["layers"], fmt='image/png',
                        transparent=layer_config.get("transparent", False), overlay=True, control=True,
                        name=layer_config["attr"]
                    ).add_to(m)
                marker_cluster = MarkerCluster(name='Estaciones').add_to(m)
                for _, row in gdf_filtered_map.iterrows():
                    html = f"""
                    <b>Estación:</b> {row[Config.STATION_NAME_COL]}<br>
                    <b>Municipio:</b> {row[Config.MUNICIPALITY_COL]}<br>
                    <b>Celda:</b> {row[Config.CELL_COL]}<br>
                    <b>% Datos Disponibles:</b> {row[Config.PERCENTAGE_COL]:.0f}%<br>
                    <b>Ppt. Media Anual (mm):</b> {row['precip_media_anual']:.0f}
                    """
                    folium.Marker(location=[row[Config.LATITUDE_COL], row[Config.LONGITUDE_COL]], tooltip=html).add_to(marker_cluster)
                folium.LayerControl().add_to(m)
                minimap = MiniMap(toggle_display=True)
                m.add_child(minimap)
                folium_static(m, height=700, width="100%")
            else:
                st.warning("No hay estaciones seleccionadas para mostrar en el mapa.")

    with sub_tab_grafico:
        st.subheader("Disponibilidad y Composición de Datos por Estación")
        if not gdf_filtered.empty:
            if st.session_state.analysis_mode == "Completar series (interpolación)":
                st.info("Mostrando la composición de datos originales vs. completados para el período seleccionado.")
                data_composition = st.session_state.df_monthly_filtered.groupby([Config.STATION_NAME_COL, Config.ORIGIN_COL]).size().unstack(fill_value=0)
                if 'Original' not in data_composition: data_composition['Original'] = 0
                if 'Completado' not in data_composition: data_composition['Completado'] = 0
                data_composition['total'] = data_composition['Original'] + data_composition['Completado']
                data_composition['% Original'] = (data_composition['Original'] / data_composition['total']) * 100
                data_composition['% Completado'] = (data_composition['Completado'] / data_composition['total']) * 100
                sort_order_comp = st.radio("Ordenar por:", ["% Datos Originales (Mayor a Menor)", "% Datos Originales (Menor a Mayor)", "Alfabético"], horizontal=True, key="sort_comp")
                if "Mayor a Menor" in sort_order_comp: data_composition = data_composition.sort_values("% Original", ascending=False)
                elif "Menor a Mayor" in sort_order_comp: data_composition = data_composition.sort_values("% Original", ascending=True)
                else: data_composition = data_composition.sort_index(ascending=True)
                
                df_plot = data_composition.reset_index().melt(
                    id_vars=Config.STATION_NAME_COL, value_vars=['% Original', '% Completado'],
                    var_name='Tipo de Dato', value_name='Porcentaje')
                
                fig_comp = px.bar(df_plot, x=Config.STATION_NAME_COL, y='Porcentaje', color='Tipo de Dato',
                                  title='Composición de Datos por Estación',
                                  labels={Config.STATION_NAME_COL: 'Estación', 'Porcentaje': '% del Período'},
                                  color_discrete_map={'% Original': '#1f77b4', '% Completado': '#ff7f0e'}, text_auto='.1f')
                fig_comp.update_layout(height=600, xaxis={'categoryorder': 'trace'})
                st.plotly_chart(fig_comp, use_container_width=True)
            else:
                st.info("Mostrando el porcentaje de disponibilidad de datos según el archivo de estaciones.")
                sort_order_disp = st.radio("Ordenar estaciones por:", ["% Datos (Mayor a Menor)", "% Datos (Menor a Mayor)", "Alfabético"], horizontal=True, key="sort_disp")
                df_chart = gdf_filtered.copy()
                if "% Datos (Mayor a Menor)" in sort_order_disp: df_chart = df_chart.sort_values(Config.PERCENTAGE_COL, ascending=False)
                elif "% Datos (Menor a Mayor)" in sort_order_disp: df_chart = df_chart.sort_values(Config.PERCENTAGE_COL, ascending=True)
                else: df_chart = df_chart.sort_values(Config.STATION_NAME_COL, ascending=True)
                
                fig_disp = px.bar(df_chart, x=Config.STATION_NAME_COL, y=Config.PERCENTAGE_COL,
                                  title='Porcentaje de Disponibilidad de Datos Históricos',
                                  labels={Config.STATION_NAME_COL: 'Estación', Config.PERCENTAGE_COL: '% de Datos Disponibles'},
                                  color=Config.PERCENTAGE_COL, color_continuous_scale=px.colors.sequential.Viridis)
                fig_disp.update_layout(height=600, xaxis={'categoryorder':'trace'})
                st.plotly_chart(fig_disp, use_container_width=True)
        else:
            st.warning("No hay estaciones seleccionadas para mostrar el gráfico.")
            
def display_graphs_tab(df_anual_melted, df_monthly_filtered, stations_for_analysis):
    st.header("Visualizaciones de Precipitación")
    
    selected_stations_str = f"{len(stations_for_analysis)} estaciones" if len(stations_for_analysis) > 1 else f"1 estación: {stations_for_analysis[0]}"
    st.info(f"Mostrando análisis para {selected_stations_str} en el período {st.session_state.year_range[0]} - {st.session_state.year_range[1]} y meses {', '.join([str(m) for m in st.session_state.meses_numeros])}.")

    if len(stations_for_analysis) == 0:
        st.warning("Por favor, seleccione al menos una estación para ver esta sección.")
        return

    sub_tab_anual, sub_tab_mensual, sub_tab_comparacion, sub_tab_distribucion, sub_tab_acumulada, sub_tab_altitud = st.tabs(["Análisis Anual", "Análisis Mensual", "Comparación Rápida", "Distribución", "Acumulada", "Relación Altitud"])

    with sub_tab_anual:
        anual_graf_tab, anual_analisis_tab = st.tabs(["Gráfico de Serie Anual", "Análisis Multianual"])
        with anual_graf_tab:
            if not df_anual_melted.empty:
                st.subheader("Precipitación Anual (mm)")
                chart_anual = alt.Chart(df_anual_melted).mark_line(point=True).encode(
                    x=alt.X(f'{Config.YEAR_COL}:O', title='Año'),
                    y=alt.Y(f'{Config.PRECIPITATION_COL}:Q', title='Precipitación (mm)'),
                    color=f'{Config.STATION_NAME_COL}:N',
                    tooltip=[alt.Tooltip(Config.STATION_NAME_COL), alt.Tooltip(Config.YEAR_COL), alt.Tooltip(f'{Config.PRECIPITATION_COL}:Q', format='.0f')]
                ).properties(title=f'Precipitación Anual por Estación ({st.session_state.year_range[0]} - {st.session_state.year_range[1]})').interactive()
                st.altair_chart(chart_anual, use_container_width=True)

        with anual_analisis_tab:
            if not df_anual_melted.empty:
                st.subheader("Precipitación Media Multianual")
                st.caption(f"Período de análisis: {st.session_state.year_range[0]} - {st.session_state.year_range[1]}")
                chart_type_annual = st.radio("Seleccionar tipo de gráfico:", ("Gráfico de Barras (Promedio)", "Gráfico de Cajas (Distribución)"), key="avg_chart_type_annual", horizontal=True)
                if chart_type_annual == "Gráfico de Barras (Promedio)":
                    df_summary = df_anual_melted.groupby(Config.STATION_NAME_COL, as_index=False)[Config.PRECIPITATION_COL].mean().round(0)
                    sort_order = st.radio("Ordenar estaciones por:", ["Promedio (Mayor a Menor)", "Promedio (Menor a Mayor)", "Alfabético"], horizontal=True, key="sort_annual_avg")
                    if "Mayor a Menor" in sort_order: df_summary = df_summary.sort_values(Config.PRECIPITATION_COL, ascending=False)
                    elif "Menor a Mayor" in sort_order: df_summary = df_summary.sort_values(Config.PRECIPITATION_COL, ascending=True)
                    else: df_summary = df_summary.sort_values(Config.STATION_NAME_COL, ascending=True)
                    
                    fig_avg = px.bar(df_summary, x=Config.STATION_NAME_COL, y=Config.PRECIPITATION_COL, title=f'Promedio de Precipitación Anual por Estación ({st.session_state.year_range[0]} - {st.session_state.year_range[1]})', labels={Config.STATION_NAME_COL: 'Estación', Config.PRECIPITATION_COL: 'Precipitación Media Anual (mm)'}, color=Config.PRECIPITATION_COL, color_continuous_scale=px.colors.sequential.Blues_r)
                    fig_avg.update_layout(height=600, xaxis={'categoryorder':'total descending' if "Mayor a Menor" in sort_order else ('total ascending' if "Menor a Mayor" in sort_order else 'trace')})
                    st.plotly_chart(fig_avg, use_container_width=True)
                else:
                    df_anual_filtered_for_box = df_anual_melted[df_anual_melted[Config.STATION_NAME_COL].isin(stations_for_analysis)]
                    fig_box_annual = px.box(df_anual_filtered_for_box, x=Config.STATION_NAME_COL, y=Config.PRECIPITATION_COL, color=Config.STATION_NAME_COL, points='all', title='Distribución de la Precipitación Anual por Estación', labels={Config.STATION_NAME_COL: 'Estación', Config.PRECIPITATION_COL: 'Precipitación Anual (mm)'})
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
                    base_chart = alt.Chart(df_monthly_filtered).encode(x=alt.X(f'{Config.DATE_COL}:T', title='Fecha'), y=alt.Y(f'{Config.PRECIPITATION_COL}:Q', title='Precipitación (mm)'), tooltip=[alt.Tooltip(Config.DATE_COL, format='%Y-%m'), alt.Tooltip(Config.PRECIPITATION_COL, format='.0f'), Config.STATION_NAME_COL, Config.ORIGIN_COL, alt.Tooltip(f'{Config.MONTH_COL}:N', title="Mes")])
                    if color_by == "Estación":
                        color_encoding = alt.Color(f'{Config.STATION_NAME_COL}:N', legend=alt.Legend(title="Estaciones"))
                    else:
                        color_encoding = alt.Color(f'month({Config.DATE_COL}):N', legend=alt.Legend(title="Meses"), scale=alt.Scale(scheme='tableau20'))
                    
                    if chart_type == "Líneas y Puntos":
                        line_chart = base_chart.mark_line(opacity=0.4, color='lightgray').encode(detail=f'{Config.STATION_NAME_COL}:N')
                        point_chart = base_chart.mark_point(filled=True, size=60).encode(color=color_encoding)
                        final_chart = (line_chart + point_chart)
                    else:
                        point_chart = base_chart.mark_point(filled=True, size=60).encode(color=color_encoding)
                        final_chart = point_chart
                    st.altair_chart(final_chart.properties(title=f"Serie de Precipitación Mensual ({st.session_state.year_range[0]} - {st.session_state.year_range[1]})").interactive(), use_container_width=True)
                else:
                    st.subheader("Distribución de la Precipitación Mensual")
                    fig_box_monthly = px.box(df_monthly_filtered, x=Config.MONTH_COL, y=Config.PRECIPITATION_COL, color=Config.STATION_NAME_COL, title='Distribución de la Precipitación por Mes', labels={Config.MONTH_COL: 'Mes', Config.PRECIPITATION_COL: 'Precipitación Mensual (mm)', Config.STATION_NAME_COL: 'Estación'})
                    fig_box_monthly.update_layout(height=600)
                    st.plotly_chart(fig_box_monthly, use_container_width=True)
        with mensual_enso_tab:
            if st.session_state.df_enso is not None:
                enso_filtered = st.session_state.df_enso[(st.session_state.df_enso[Config.DATE_COL].dt.year >= st.session_state.year_range[0]) & (st.session_state.df_enso[Config.DATE_COL].dt.year <= st.session_state.year_range[1]) & (st.session_state.df_enso[Config.DATE_COL].dt.month.isin(st.session_state.meses_numeros))]
                fig_enso_mensual = create_enso_chart(enso_filtered)
                st.plotly_chart(fig_enso_mensual, use_container_width=True, key="enso_chart_mensual")

        with mensual_datos_tab:
            st.subheader("Datos de Precipitación Mensual Detallados")
            if not df_monthly_filtered.empty:
                df_values = df_monthly_filtered.pivot_table(index=Config.DATE_COL, columns=Config.STATION_NAME_COL, values=Config.PRECIPITATION_COL).round(0)
                st.dataframe(df_values)

    with sub_tab_comparacion:
        st.subheader("Comparación de Precipitación entre Estaciones")
        if len(stations_for_analysis) < 2:
            st.info("Seleccione al menos dos estaciones para comparar.")
        else:
            st.markdown("##### Precipitación Mensual Promedio")
            df_monthly_avg = df_monthly_filtered.groupby([Config.STATION_NAME_COL, Config.MONTH_COL])[Config.PRECIPITATION_COL].mean().reset_index()
            fig_avg_monthly = px.line(df_monthly_avg, x=Config.MONTH_COL, y=Config.PRECIPITATION_COL, color=Config.STATION_NAME_COL,
                                      labels={Config.MONTH_COL: 'Mes', Config.PRECIPITATION_COL: 'Precipitación Promedio (mm)'},
                                      title='Promedio de Precipitación Mensual por Estación')
            meses_dict = {'Enero': 1, 'Febrero': 2, 'Marzo': 3, 'Abril': 4, 'Mayo': 5, 'Junio': 6, 'Julio': 7, 'Agosto': 8, 'Septiembre': 9, 'Octubre': 10, 'Noviembre': 11, 'Diciembre': 12}
            fig_avg_monthly.update_layout(height=600, xaxis = dict(tickmode = 'array', tickvals = list(meses_dict.values()), ticktext = list(meses_dict.keys())))
            st.plotly_chart(fig_avg_monthly, use_container_width=True)
            st.markdown("##### Distribución de Precipitación Anual")
            df_anual_filtered_for_box = df_anual_melted[df_anual_melted[Config.STATION_NAME_COL].isin(stations_for_analysis)]
            fig_box_annual = px.box(df_anual_filtered_for_box, x=Config.STATION_NAME_COL, y=Config.PRECIPITATION_COL, color=Config.STATION_NAME_COL, points='all', title='Distribución de la Precipitación Anual por Estación', labels={Config.STATION_NAME_COL: 'Estación', Config.PRECIPITATION_COL: 'Precipitación Anual (mm)'})
            fig_box_annual.update_layout(height=600)
            st.plotly_chart(fig_box_annual, use_container_width=True)

    with sub_tab_distribucion:
        st.subheader("Distribución de la Precipitación")
        distribucion_tipo = st.radio("Seleccionar tipo de distribución:", ("Anual", "Mensual"), horizontal=True)
        if distribucion_tipo == "Anual":
            if not df_anual_melted.empty:
                fig_hist_anual = px.histogram(df_anual_melted, x=Config.PRECIPITATION_COL, color=Config.STATION_NAME_COL,
                                              title=f'Distribución Anual de Precipitación ({st.session_state.year_range[0]} - {st.session_state.year_range[1]})',
                                              labels={Config.PRECIPITATION_COL: 'Precipitación Anual (mm)', 'count': 'Frecuencia'})
                fig_hist_anual.update_layout(height=600)
                st.plotly_chart(fig_hist_anual, use_container_width=True)
            else:
                st.info("No hay datos anuales para mostrar la distribución.")
        else:
            if not df_monthly_filtered.empty:
                fig_hist_mensual = px.histogram(df_monthly_filtered, x=Config.PRECIPITATION_COL, color=Config.STATION_NAME_COL,
                                                  title=f'Distribución Mensual de Precipitación ({st.session_state.year_range[0]} - {st.session_state.year_range[1]})',
                                                  labels={Config.PRECIPITATION_COL: 'Precipitación Mensual (mm)', 'count': 'Frecuencia'})
                fig_hist_mensual.update_layout(height=600)
                st.plotly_chart(fig_hist_mensual, use_container_width=True)
            else:
                st.info("No hay datos mensuales para mostrar la distribución.")

    with sub_tab_acumulada:
        st.subheader("Precipitación Acumulada Anual")
        if not df_anual_melted.empty:
            df_acumulada = df_anual_melted.groupby([Config.YEAR_COL, Config.STATION_NAME_COL])[Config.PRECIPITATION_COL].sum().reset_index()
            fig_acumulada = px.bar(df_acumulada, x=Config.YEAR_COL, y=Config.PRECIPITATION_COL, color=Config.STATION_NAME_COL,
                                   title=f'Precipitación Acumulada por Año ({st.session_state.year_range[0]} - {st.session_state.year_range[1]})',
                                   labels={Config.YEAR_COL: 'Año', Config.PRECIPITATION_COL: 'Precipitación Acumulada (mm)'})
            fig_acumulada.update_layout(barmode='group', height=600)
            st.plotly_chart(fig_acumulada, use_container_width=True)
        else:
            st.info("No hay datos para calcular la precipitación acumulada.")

    with sub_tab_altitud:
        st.subheader("Relación entre Altitud y Precipitación")
        if not df_anual_melted.empty and not st.session_state.gdf_filtered[Config.ALTITUDE_COL].isnull().all():
            df_relacion = df_anual_melted.groupby(Config.STATION_NAME_COL)[Config.PRECIPITATION_COL].mean().reset_index()
            df_relacion = df_relacion.merge(st.session_state.gdf_filtered[[Config.STATION_NAME_COL, Config.ALTITUDE_COL]].drop_duplicates(), on=Config.STATION_NAME_COL, how='left')
            fig_relacion = px.scatter(df_relacion, x=Config.ALTITUDE_COL, y=Config.PRECIPITATION_COL, color=Config.STATION_NAME_COL,
                                      title='Relación entre Precipitación Media Anual y Altitud',
                                      labels={Config.ALTITUDE_COL: 'Altitud (m)', Config.PRECIPITATION_COL: 'Precipitación Media Anual (mm)'})
            fig_relacion.update_layout(height=600)
            st.plotly_chart(fig_relacion, use_container_width=True)
        else:
            st.info("No hay datos de altitud o precipitación disponibles para analizar la relación.")

def display_advanced_maps_tab(gdf_filtered, df_anual_melted, stations_for_analysis):
    st.header("Mapas Avanzados")
    selected_stations_str = f"{len(stations_for_analysis)} estaciones" if len(stations_for_analysis) > 1 else f"1 estación: {stations_for_analysis[0]}"
    st.info(f"Mostrando análisis para {selected_stations_str} en el período {st.session_state.year_range[0]} - {st.session_state.year_range[1]}.")
    
    gif_tab, temporal_tab, race_tab, anim_tab, compare_tab, kriging_tab = st.tabs(["Animación GIF (Antioquia)", "Visualización Temporal", "Gráfico de Carrera", "Mapa Animado", "Comparación de Mapas", "Interpolación Kriging"])

    with gif_tab:
        st.subheader("Distribución Espacio-Temporal de la Lluvia en Antioquia")
        if os.path.exists(Config.GIF_PATH):
            img_col1, img_col2 = st.columns([1, 1])
            with img_col1:
                if 'gif_rerun_count' not in st.session_state: st.session_state.gif_rerun_count = 0
                gif_placeholder = st.empty()
                with open(Config.GIF_PATH, "rb") as file:
                    contents = file.read()
                    data_url = base64.b64encode(contents).decode("utf-8")
                
                gif_placeholder.markdown(
                    f'<img src="data:image/gif;base64,{data_url}" alt="Animación PPAM" style="width:100%;">', 
                    unsafe_allow_html=True)
                
                if st.button("Reiniciar Animación", key="restart_gif"):
                    st.session_state.gif_rerun_count += 1
                    with open(Config.GIF_PATH, "rb") as file:
                        contents = file.read()
                        data_url = base64.b64encode(contents).decode("utf-8")
                    gif_placeholder.markdown(
                        f'<img src="data:image/gif;base64,{data_url}" alt="Animación PPAM {st.session_state.gif_rerun_count}" style="width:100%;">',
                        unsafe_allow_html=True)
        else:
            st.warning("No se encontró el archivo GIF 'PPAM.gif'. Asegúrate de que esté en el directorio principal de la aplicación.")

    with temporal_tab:
        if len(stations_for_analysis) == 0:
            st.warning("Por favor, seleccione al menos una estación para ver esta sección.")
            return
        
        st.subheader("Explorador Anual de Precipitación")
        if not df_anual_melted.empty:
            all_years_int = sorted([int(y) for y in df_anual_melted[Config.YEAR_COL].unique()])
            if all_years_int:
                selected_year = st.slider('Seleccione un Año para Explorar', min_value=min(all_years_int), max_value=max(all_years_int), value=min(all_years_int))
                controls_col, map_col = st.columns([1, 3])
                with controls_col:
                    st.markdown("##### Opciones de Visualización")
                    selected_base_map_config, selected_overlays_config = display_map_controls(st, "temporal")
                    st.markdown(f"#### Resumen del Año: {selected_year}")
                    df_year_filtered = df_anual_melted[df_anual_melted[Config.YEAR_COL] == str(selected_year)].dropna(subset=[Config.PRECIPITATION_COL])
                    logo_col, info_col = st.columns([1, 4])
                    with logo_col:
                        if os.path.exists(Config.LOGO_DROP_PATH): st.image(Config.LOGO_DROP_PATH, width=40)
                    with info_col:
                        st.metric(f"Estaciones con Datos en {selected_year}", f"{len(df_year_filtered)} de {len(stations_for_analysis)}")
                    if not df_year_filtered.empty:
                        max_row = df_year_filtered.loc[df_year_filtered[Config.PRECIPITATION_COL].idxmax()]
                        min_row = df_year_filtered.loc[df_year_filtered[Config.PRECIPITATION_COL].idxmin()]
                        st.info(f"""
                        **Ppt. Máxima ({selected_year}):**
                        {max_row[Config.STATION_NAME_COL]} ({max_row[Config.PRECIPITATION_COL]:.0f} mm)

                        **Ppt. Mínima ({selected_year}):**
                        {min_row[Config.STATION_NAME_COL]} ({min_row[Config.PRECIPITATION_COL]:.0f} mm)
                        """)
                    else:
                        st.warning(f"No hay datos de precipitación para el año {selected_year}.")
                with map_col:
                    m_temporal = folium.Map(location=[6.24, -75.58], zoom_start=7, tiles=selected_base_map_config.get("tiles", "OpenStreetMap"), attr=selected_base_map_config.get("attr", None))
                    if not df_year_filtered.empty:
                        min_val, max_val = df_anual_melted[Config.PRECIPITATION_COL].min(), df_anual_melted[Config.PRECIPITATION_COL].max()
                        colormap = cm.linear.YlGnBu_09.scale(vmin=min_val, vmax=max_val)
                        for _, row in df_year_filtered.iterrows():
                            folium.CircleMarker(
                                location=[row[Config.LATITUDE_COL], row[Config.LONGITUDE_COL]], radius=5,
                                color=colormap(row[Config.PRECIPITATION_COL]), fill=True, fill_color=colormap(row[Config.PRECIPITATION_COL]),
                                fill_opacity=0.8, tooltip=f"{row[Config.STATION_NAME_COL]}: {row[Config.PRECIPITATION_COL]:.0f} mm"
                            ).add_to(m_temporal)
                        bounds = st.session_state.gdf_stations.loc[st.session_state.gdf_stations[Config.STATION_NAME_COL].isin(df_year_filtered[Config.STATION_NAME_COL])].total_bounds
                        m_temporal.fit_bounds([[bounds[1], bounds[0]], [bounds[3], bounds[2]]])
                    for layer_config in selected_overlays_config:
                        folium.raster_layers.WmsTileLayer(url=layer_config["url"], layers=layer_config["layers"], fmt='image/png', transparent=layer_config.get("transparent", False), overlay=True, control=True, name=layer_config["attr"]).add_to(m_temporal)
                    folium.LayerControl().add_to(m_temporal)
                    folium_static(m_temporal, height=700, width="100%")

    with race_tab:
        st.subheader("Ranking Anual de Precipitación por Estación")
        if not df_anual_melted.empty:
            station_order = df_anual_melted.groupby(Config.STATION_NAME_COL)[Config.PRECIPITATION_COL].sum().sort_values(ascending=True).index
            fig_racing = px.bar(
                df_anual_melted, x=Config.PRECIPITATION_COL, y=Config.STATION_NAME_COL,
                animation_frame=Config.YEAR_COL, orientation='h', text=Config.PRECIPITATION_COL,
                labels={Config.PRECIPITATION_COL: 'Precipitación Anual (mm)', Config.STATION_NAME_COL: 'Estación'},
                title=f"Evolución de Precipitación Anual por Estación ({st.session_state.year_range[0]} - {st.session_state.year_range[1]})",
                category_orders={Config.STATION_NAME_COL: station_order}
            )
            fig_racing.update_traces(texttemplate='%{x:.0f}', textposition='outside')
            fig_racing.update_layout(
                xaxis_range=[0, df_anual_melted[Config.PRECIPITATION_COL].max() * 1.15],
                height=max(600, len(stations_for_analysis) * 35),
                title_font_size=20, font_size=12
            )
            fig_racing.layout.sliders[0]['currentvalue']['font']['size'] = 24
            fig_racing.layout.sliders[0]['currentvalue']['prefix'] = '<b>Año: </b>'
            fig_racing.layout.updatemenus[0].buttons[0].args[1]['frame']['duration'] = 800
            fig_racing.layout.updatemenus[0].buttons[0].args[1]['transition']['duration'] = 500
            st.plotly_chart(fig_racing, use_container_width=True)

    with anim_tab:
        st.subheader("Mapa Animado de Precipitación Anual")
        st.info("Este mapa utiliza Plotly. Los controles de mapa se encuentran en las otras pestañas de mapas.")
        if not df_anual_melted.empty:
            all_years = sorted(df_anual_melted[Config.YEAR_COL].unique())
            if all_years:
                all_selected_stations_info = st.session_state.gdf_stations.loc[st.session_state.gdf_stations[Config.STATION_NAME_COL].isin(stations_for_analysis)][[Config.STATION_NAME_COL, Config.LATITUDE_COL, Config.LONGITUDE_COL, Config.ALTITUDE_COL]].drop_duplicates()
                full_grid = pd.MultiIndex.from_product([all_selected_stations_info[Config.STATION_NAME_COL], all_years], names=[Config.STATION_NAME_COL, Config.YEAR_COL]).to_frame(index=False)
                full_grid = pd.merge(full_grid, all_selected_stations_info, on=Config.STATION_NAME_COL)
                df_anim_complete = pd.merge(full_grid, df_anual_melted[[Config.STATION_NAME_COL, Config.YEAR_COL, Config.PRECIPITATION_COL]], on=[Config.STATION_NAME_COL, Config.YEAR_COL], how='left')
                df_anim_complete['texto_tooltip'] = df_anim_complete.apply(lambda row: f"<b>Estación:</b> {row[Config.STATION_NAME_COL]}<br><b>Precipitación:</b> {row[Config.PRECIPITATION_COL]:.0f} mm" if pd.notna(row[Config.PRECIPITATION_COL]) else f"<b>Estación:</b> {row[Config.STATION_NAME_COL]}<br><b>Precipitación:</b> Sin datos", axis=1)
                df_anim_complete['precipitacion_plot'] = df_anim_complete[Config.PRECIPITATION_COL].fillna(0)
                min_precip_anim, max_precip_anim = df_anual_melted[Config.PRECIPITATION_COL].min(), df_anual_melted[Config.PRECIPITATION_COL].max()
                
                fig_mapa_animado = px.scatter_geo(df_anim_complete, lat=Config.LATITUDE_COL, lon=Config.LONGITUDE_COL,
                                                  color='precipitacion_plot', size='precipitacion_plot', hover_name=Config.STATION_NAME_COL,
                                                  hover_data={Config.LATITUDE_COL: False, Config.LONGITUDE_COL: False, 'precipitacion_plot': False, 'texto_tooltip': True},
                                                  animation_frame=Config.YEAR_COL,
                                                  projection='natural earth', 
                                                  title=f'Precipitación Anual por Estación ({st.session_state.year_range[0]} - {st.session_state.year_range[1]})',
                                                  color_continuous_scale=px.colors.sequential.YlGnBu, range_color=[min_precip_anim, max_precip_anim])
                fig_mapa_animado.update_traces(hovertemplate='%{customdata[0]}')
                fig_mapa_animado.update_geos(fitbounds="locations", visible=True, showcoastlines=True, coastlinewidth=0.5, showland=True, landcolor="rgb(243, 243, 243)", showocean=True, oceancolor="rgb(220, 235, 255)", showcountries=True, countrywidth=0.5)
                fig_mapa_animado.update_layout(height=700, sliders=[dict(currentvalue=dict(font=dict(size=24, color="#707070"), prefix='<b>Año: </b>', visible=True))])
                st.plotly_chart(fig_mapa_animado, use_container_width=True)

    with compare_tab:
        st.subheader("Comparación de Mapas Anuales")
        if len(stations_for_analysis) < 1:
            st.warning("Por favor, seleccione al menos una estación para ver esta sección.")
            return
        if not df_anual_melted.empty and len(df_anual_melted[Config.YEAR_COL].unique()) > 0:
            control_col, map_col1, map_col2 = st.columns([1, 2, 2])
            with control_col:
                st.markdown("##### Controles de Mapa")
                selected_base_map_config, selected_overlays_config = display_map_controls(st, "compare")
                min_year, max_year = int(df_anual_melted[Config.YEAR_COL].min()), int(df_anual_melted[Config.YEAR_COL].max())
                year1 = st.slider("Seleccione el año para el Mapa 1", min_year, max_year, max_year, key="compare_year1")
                year2 = st.slider("Seleccione el año para el Mapa 2", min_year, max_year, max_year - 1 if max_year > min_year else max_year, key="compare_year2")
                min_precip_comp, max_precip_comp = int(df_anual_melted[Config.PRECIPITATION_COL].min()), int(df_anual_melted[Config.PRECIPITATION_COL].max())
                color_range_comp = st.slider("Rango de Escala de Color (mm)", min_precip_comp, max_precip_comp, (min_precip_comp, max_precip_comp), key="color_comp")

            data_year1 = df_anual_melted[df_anual_melted[Config.YEAR_COL].astype(int) == year1]
            data_year2 = df_anual_melted[df_anual_melted[Config.YEAR_COL].astype(int) == year2]
            
            colormap = cm.linear.YlGnBu_09.scale(vmin=color_range_comp[0], vmax=color_range_comp[1])

            def create_compare_map(data, year, col):
                col.markdown(f"**Precipitación en {year}**")
                m = folium.Map(location=[6.24, -75.58], zoom_start=6, tiles=selected_base_map_config.get("tiles", "OpenStreetMap"), attr=selected_base_map_config.get("attr", None))
                if not data.empty:
                    for _, row in data.iterrows():
                        folium.CircleMarker(
                            location=[row[Config.LATITUDE_COL], row[Config.LONGITUDE_COL]], radius=5, color=colormap(row[Config.PRECIPITATION_COL]),
                            fill=True, fill_color=colormap(row[Config.PRECIPITATION_COL]), fill_opacity=0.8,
                            tooltip=f"{row[Config.STATION_NAME_COL]}: {row[Config.PRECIPITATION_COL]:.0f} mm"
                        ).add_to(m)
                    bounds = st.session_state.gdf_stations.loc[st.session_state.gdf_stations[Config.STATION_NAME_COL].isin(data[Config.STATION_NAME_COL])].total_bounds
                    m.fit_bounds([[bounds[1], bounds[0]], [bounds[3], bounds[2]]])
                for layer_config in selected_overlays_config:
                    folium.raster_layers.WmsTileLayer(url=layer_config["url"], layers=layer_config["layers"], fmt='image/png', transparent=layer_config.get("transparent", False), overlay=True, control=True, name=layer_config["attr"]).add_to(m)
                folium.LayerControl().add_to(m)
                with col:
                    folium_static(m, height=600, width="100%")

            create_compare_map(data_year1, year1, map_col1)
            create_compare_map(data_year2, year2, map_col2)
        else:
            st.warning("No hay años disponibles para la comparación.")

    with kriging_tab:
        st.subheader("Interpolación Kriging para un Año Específico")
        if len(stations_for_analysis) == 0:
            st.warning("Por favor, seleccione al menos una estación para ver esta sección.")
            return
        if not df_anual_melted.empty and len(df_anual_melted[Config.YEAR_COL].unique()) > 0:
            min_year, max_year = int(df_anual_melted[Config.YEAR_COL].min()), int(df_anual_melted[Config.YEAR_COL].max())
            year_kriging = st.slider("Seleccione el año para la interpolación", min_year, max_year, max_year, key="year_kriging")
            data_year_kriging = df_anual_melted[df_anual_melted[Config.YEAR_COL].astype(int) == year_kriging].copy()
            logo_col_k, metric_col_k = st.columns([1,8])
            with logo_col_k:
                if os.path.exists(Config.LOGO_DROP_PATH): st.image(Config.LOGO_DROP_PATH, width=40)
            with metric_col_k:
                st.metric(f"Estaciones con datos en {year_kriging}", f"{len(data_year_kriging)} de {len(stations_for_analysis)}")
            if len(data_year_kriging) < 3:
                st.warning(f"Se necesitan al menos 3 estaciones con datos en el año {year_kriging} para generar el mapa Kriging.")
            else:
                with st.spinner("Generando mapa Kriging..."):
                    with st.expander("¿Cómo interpretar este análisis?"):
                        st.markdown("""
                            La **interpolación Kriging** es un método geoestadístico para estimar valores en ubicaciones no muestreadas a partir de mediciones en puntos cercanos.
                            - El mapa muestra una superficie continua de precipitación, creada a partir de los datos de tus estaciones.
                            - Los círculos rojos representan las estaciones de lluvia.
                            - Este método considera no solo la distancia, sino también las propiedades de varianza espacial de los datos.
                        """)
                    data_year_kriging['tooltip'] = data_year_kriging.apply(
                        lambda row: f"<b>Estación:</b> {row[Config.STATION_NAME_COL]}<br>Municipio: {row[Config.MUNICIPALITY_COL]}<br>Ppt: {row[Config.PRECIPITATION_COL]:.0f} mm",
                        axis=1
                    )
                    lons, lats, vals = data_year_kriging[Config.LONGITUDE_COL].values, data_year_kriging[Config.LATITUDE_COL].values, data_year_kriging[Config.PRECIPITATION_COL].values
                    bounds = st.session_state.gdf_stations.loc[st.session_state.gdf_stations[Config.STATION_NAME_COL].isin(stations_for_analysis)].total_bounds
                    lon_range = [bounds[0] - 0.1, bounds[2] + 0.1]
                    lat_range = [bounds[1] - 0.1, bounds[3] + 0.1]
                    grid_lon, grid_lat = np.linspace(lon_range[0], lon_range[1], 100), np.linspace(lat_range[0], lat_range[1], 100)
                    OK = OrdinaryKriging(lons, lats, vals, variogram_model='linear', verbose=False, enable_plotting=False)
                    z, ss = OK.execute('grid', grid_lon, grid_lat)
                    fig_krig = go.Figure(data=go.Contour(z=z.T, x=grid_lon, y=grid_lat, colorscale='YlGnBu', contours=dict(showlabels=True, labelfont=dict(size=12, color='white'))))
                    
                    fig_krig.add_trace(go.Scatter(
                        x=lons, y=lats, mode='markers',
                        marker=dict(color='red', size=5, symbol='circle'),
                        name='Estaciones', text=data_year_kriging['tooltip'], hoverinfo='text'
                    ))
                    fig_krig.update_layout(height=700, title=f"Superficie de Precipitación Interpolada (Kriging) - Año {year_kriging}", xaxis_title="Longitud", yaxis_title="Latitud")
                    st.plotly_chart(fig_krig, use_container_width=True)
        else:
            st.warning("No hay datos para realizar la interpolación.")

# ---
# FUNCIÓN RESTAURADA
# ---
def display_station_table_tab(gdf_filtered, df_anual_melted, stations_for_analysis):
    st.header("Información Detallada de las Estaciones")
    selected_stations_str = f"{len(stations_for_analysis)} estaciones" if len(stations_for_analysis) > 1 else f"1 estación: {stations_for_analysis[0]}"
    st.info(f"Mostrando análisis para {selected_stations_str} en el período {st.session_state.year_range[0]} - {st.session_state.year_range[1]}.")
    
    if len(stations_for_analysis) == 0:
        st.warning("Por favor, seleccione al menos una estación para ver esta sección.")
        return
    if not df_anual_melted.empty:
        df_info_table = gdf_filtered[[Config.STATION_NAME_COL, Config.ALTITUDE_COL, Config.MUNICIPALITY_COL, Config.REGION_COL, Config.PERCENTAGE_COL]].copy()
        df_mean_precip = df_anual_melted.groupby(Config.STATION_NAME_COL)[Config.PRECIPITATION_COL].mean().round(0).reset_index()
        df_mean_precip.rename(columns={Config.PRECIPITATION_COL: 'Precipitación media anual (mm)'}, inplace=True)
        df_info_table = df_info_table.merge(df_mean_precip, on=Config.STATION_NAME_COL, how='left')
        st.dataframe(df_info_table)
    else:
        st.info("No hay datos de precipitación anual para mostrar en la selección actual.")
# ---

# ---
# Cuerpo Principal del Script
# ---
def main():
    st.set_page_config(layout="wide", page_title=Config.APP_TITLE)
    st.markdown("""
        <style>
        div.block-container {padding-top: 2rem;}
        .sidebar .sidebar-content {font-size: 13px; }
        [data-testid="stMetricValue"] { font-size: 1.8rem; }
        [data-testid="stMetricLabel"] { font-size: 1rem; padding-bottom: 5px; }
        button[data-baseweb="tab"] { font-size: 16px; font-weight: bold; color: #333; }
        </style>
    """, unsafe_allow_html=True)

    Config.initialize_session_state()

    title_col1, title_col2 = st.columns([0.07, 0.93])
    with title_col1:
        if os.path.exists(Config.LOGO_DROP_PATH): st.image(Config.LOGO_DROP_PATH, width=50)
    with title_col2:
        st.markdown(f'<h1 style="font-size:28px; margin-top:1rem;">{Config.APP_TITLE}</h1>', unsafe_allow_html=True)
    
    st.sidebar.header("Panel de Control")

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
            st.info("Por favor, suba los 3 archivos requeridos (estaciones, precipitación, shapefile) para habilitar la aplicación.")
            st.stop()
        else:
            with st.spinner("Procesando archivos y cargando datos... Esto puede tomar un momento."):
                gdf_stations, gdf_municipios, df_long, df_enso = load_and_process_all_data(
                    uploaded_file_mapa, uploaded_file_precip, uploaded_zip_shapefile
                )
                if gdf_stations is not None:
                    st.session_state.gdf_stations = gdf_stations
                    st.session_state.gdf_municipios = gdf_municipios
                    st.session_state.df_long = df_long
                    st.session_state.df_enso = df_enso
                    st.session_state.data_loaded = True
                    st.rerun()
                else:
                    st.error("Hubo un error al procesar los archivos. Por favor, verifique que sean correctos y vuelva a intentarlo.")
                    st.stop()
    
    if st.session_state.gdf_stations is None:
        st.stop()

    with st.sidebar.expander("**1. Filtros Geográficos y de Datos**", expanded=True):
        def apply_filters_to_stations(df, min_perc, altitudes, regions, municipios, celdas):
            stations_filtered = df.copy()
            if Config.PERCENTAGE_COL in stations_filtered.columns:
                stations_filtered[Config.PERCENTAGE_COL] = pd.to_numeric(stations_filtered[Config.PERCENTAGE_COL].astype(str).str.replace(',', '.'), errors='coerce').fillna(0)
                stations_filtered = stations_filtered[stations_filtered[Config.PERCENTAGE_COL] >= min_perc]
            
            if altitudes:
                conditions = []
                for r in altitudes:
                    if r == '0-500': conditions.append((stations_filtered[Config.ALTITUDE_COL] >= 0) & (stations_filtered[Config.ALTITUDE_COL] <= 500))
                    elif r == '500-1000': conditions.append((stations_filtered[Config.ALTITUDE_COL] > 500) & (stations_filtered[Config.ALTITUDE_COL] <= 1000))
                    elif r == '1000-2000': conditions.append((stations_filtered[Config.ALTITUDE_COL] > 1000) & (stations_filtered[Config.ALTITUDE_COL] <= 2000))
                    elif r == '2000-3000': conditions.append((stations_filtered[Config.ALTITUDE_COL] > 2000) & (stations_filtered[Config.ALTITUDE_COL] <= 3000))
                    elif r == '>3000': conditions.append(stations_filtered[Config.ALTITUDE_COL] > 3000)
                if conditions: stations_filtered = stations_filtered[pd.concat(conditions, axis=1).any(axis=1)]

            if regions: stations_filtered = stations_filtered[stations_filtered[Config.REGION_COL].isin(regions)]
            if municipios: stations_filtered = stations_filtered[stations_filtered[Config.MUNICIPALITY_COL].isin(municipios)]
            if celdas: stations_filtered = stations_filtered[stations_filtered[Config.CELL_COL].isin(celdas)]
            return stations_filtered

        min_data_perc = st.slider("Filtrar por % de datos mínimo:", 0, 100, st.session_state.get('min_data_perc_slider', 0), key='min_data_perc_slider')
        altitude_ranges = ['0-500', '500-1000', '1000-2000', '2000-3000', '>3000']
        selected_altitudes = st.multiselect('Filtrar por Altitud (m)', options=altitude_ranges, default=st.session_state.get('altitude_multiselect', []), key='altitude_multiselect')
        
        regions_list = sorted(st.session_state.gdf_stations[Config.REGION_COL].dropna().unique())
        selected_regions = st.multiselect('Filtrar por Depto/Región', options=regions_list, default=st.session_state.get('regions_multiselect', []), key='regions_multiselect')
        
        temp_filtered_for_ui = apply_filters_to_stations(st.session_state.gdf_stations, min_data_perc, selected_altitudes, selected_regions, [], [])
        municipios_list = sorted(temp_filtered_for_ui[Config.MUNICIPALITY_COL].dropna().unique())
        selected_municipios = st.multiselect('Filtrar por Municipio', options=municipios_list, default=st.session_state.get('municipios_multiselect', []), key='municipios_multiselect')
        
        temp_filtered_for_ui = apply_filters_to_stations(temp_filtered_for_ui, min_data_perc, selected_altitudes, selected_regions, selected_municipios, [])
        celdas_list = sorted(temp_filtered_for_ui[Config.CELL_COL].dropna().unique())
        selected_celdas = st.multiselect('Filtrar por Celda_XY', options=celdas_list, default=st.session_state.get('celdas_multiselect', []), key='celdas_multiselect')

        st.session_state.selected_altitudes = selected_altitudes
        st.session_state.selected_regions = selected_regions
        st.session_state.selected_municipios = selected_municipios
        st.session_state.selected_celdas = selected_celdas

        if st.button("🧹 Limpiar Filtros"):
            st.session_state.min_data_perc_slider = 0
            st.session_state.altitude_multiselect = []
            st.session_state.regions_multiselect = []
            st.session_state.municipios_multiselect = []
            st.session_state.celdas_multiselect = []
            st.session_state.station_multiselect = []
            st.session_state.select_all_checkbox = False
            st.rerun()

    with st.sidebar.expander("**2. Selección de Estaciones y Período**", expanded=True):
        stations_master_list = apply_filters_to_stations(st.session_state.gdf_stations, min_data_perc, selected_altitudes, selected_regions, selected_municipios, selected_celdas)
        stations_options = sorted(stations_master_list[Config.STATION_NAME_COL].unique())
        
        select_all = st.checkbox("Seleccionar/Deseleccionar todas las estaciones", key='select_all_checkbox')
        if select_all:
            selected_stations = st.multiselect('Seleccionar Estaciones', options=stations_options, default=stations_options, key='station_multiselect')
        else:
            selected_stations = st.multiselect('Seleccionar Estaciones', options=stations_options, default=st.session_state.get('station_multiselect', []), key='station_multiselect')

        years_with_data_in_selection = sorted([int(col) for col in st.session_state.gdf_stations.columns if str(col).isdigit()])
        if not years_with_data_in_selection:
            st.error("No se encontraron años disponibles en el archivo de estaciones.")
            st.stop()

        year_range = st.slider("Seleccionar Rango de Años", min(years_with_data_in_selection), max(years_with_data_in_selection), (min(years_with_data_in_selection), max(years_with_data_in_selection)))
        meses_dict = {'Enero': 1, 'Febrero': 2, 'Marzo': 3, 'Abril': 4, 'Mayo': 5, 'Junio': 6, 'Julio': 7, 'Agosto': 8, 'Septiembre': 9, 'Octubre': 10, 'Noviembre': 11, 'Diciembre': 12}
        meses_nombres = st.multiselect("Seleccionar Meses", list(meses_dict.keys()), default=list(meses_dict.keys()))
        meses_numeros = [meses_dict[m] for m in meses_nombres]

    with st.sidebar.expander("Opciones de Preprocesamiento de Datos", expanded=False):
        analysis_mode = st.radio("Análisis de Series Mensuales", ("Usar datos originales", "Completar series (interpolación)"), key="analysis_mode_radio")
        exclude_na = st.checkbox("Excluir datos nulos (NaN)", value=st.session_state.exclude_na, key='exclude_na_checkbox')
        exclude_zeros = st.checkbox("Excluir valores cero (0)", value=st.session_state.exclude_zeros, key='exclude_zeros_checkbox')

    st.session_state.gdf_filtered = apply_filters_to_stations(st.session_state.gdf_stations, min_data_perc, selected_altitudes, selected_regions, selected_municipios, selected_celdas)
    
    stations_for_analysis = selected_stations if selected_stations else st.session_state.gdf_filtered[Config.STATION_NAME_COL].unique()
    st.session_state.gdf_filtered = st.session_state.gdf_filtered[st.session_state.gdf_filtered[Config.STATION_NAME_COL].isin(stations_for_analysis)]

    cols_to_melt = [str(y) for y in range(year_range[0], year_range[1] + 1) if str(y) in st.session_state.gdf_stations.columns]
    df_anual_melted = st.session_state.gdf_stations.melt(
        id_vars=[Config.STATION_NAME_COL, Config.MUNICIPALITY_COL, Config.LONGITUDE_COL, Config.LATITUDE_COL, Config.ALTITUDE_COL],
        value_vars=cols_to_melt, var_name=Config.YEAR_COL, value_name=Config.PRECIPITATION_COL
    )
    df_anual_melted = df_anual_melted[df_anual_melted[Config.STATION_NAME_COL].isin(stations_for_analysis)]
    
    if st.session_state.df_long is not None:
        if analysis_mode == "Completar series (interpolación)":
            df_monthly_processed = complete_series(st.session_state.df_long.copy())
        else:
            df_monthly_processed = st.session_state.df_long.copy()
        
        st.session_state.df_monthly_processed = df_monthly_processed
        
        df_monthly_filtered = df_monthly_processed[
            (df_monthly_processed[Config.STATION_NAME_COL].isin(stations_for_analysis)) &
            (df_monthly_processed[Config.DATE_COL].dt.year >= year_range[0]) &
            (df_monthly_processed[Config.DATE_COL].dt.year <= year_range[1]) &
            (df_monthly_processed[Config.DATE_COL].dt.month.isin(meses_numeros))
        ].copy()
    else:
        df_monthly_filtered = pd.DataFrame()
        st.session_state.df_monthly_processed = pd.DataFrame()
    
    if exclude_na:
        df_anual_melted.dropna(subset=[Config.PRECIPITATION_COL], inplace=True)
        if not df_monthly_filtered.empty:
            df_monthly_filtered.dropna(subset=[Config.PRECIPITATION_COL], inplace=True)
    if exclude_zeros:
        df_anual_melted = df_anual_melted[df_anual_melted[Config.PRECIPITATION_COL] > 0]
        if not df_monthly_filtered.empty:
            df_monthly_filtered = df_monthly_filtered[df_monthly_filtered[Config.PRECIPITATION_COL] > 0]

    st.session_state.year_range = year_range
    st.session_state.meses_numeros = meses_numeros
    st.session_state.df_monthly_filtered = df_monthly_filtered

    tab_names = [
        "🏠 Bienvenida", "Distribución Espacial", "Gráficos", "Mapas Avanzados", 
        "Análisis de Anomalías", "Estadísticas", "Análisis de Correlación", 
        "Análisis ENSO", "Tendencias y Pronósticos", "Descargas", "Tabla de Estaciones"
    ]
    
    tabs = st.tabs(tab_names)
    (
        bienvenida_tab, mapa_tab, graficos_tab, mapas_avanzados_tab, 
        anomalias_tab, estadisticas_tab, correlacion_tab, 
        enso_tab, tendencias_tab, descargas_tab, tabla_estaciones_tab
    ) = tabs

    with bienvenida_tab:
        display_welcome_tab()
    with mapa_tab:
        display_spatial_distribution_tab(st.session_state.gdf_filtered, df_anual_melted, stations_for_analysis)
    with graficos_tab:
        display_graphs_tab(df_anual_melted, df_monthly_filtered, stations_for_analysis)
    with mapas_avanzados_tab:
        display_advanced_maps_tab(st.session_state.gdf_filtered, df_anual_melted, stations_for_analysis)
    with anomalias_tab:
        display_anomalies_tab(st.session_state.df_long, df_monthly_filtered, stations_for_analysis)
    with estadisticas_tab:
        display_stats_tab(st.session_state.df_long, df_anual_melted, df_monthly_filtered, stations_for_analysis)
    with correlacion_tab:
        display_correlation_tab(df_monthly_filtered, stations_for_analysis)
    with enso_tab:
        display_enso_tab(df_monthly_filtered, st.session_state.df_enso, st.session_state.gdf_filtered, stations_for_analysis)
    with tendencias_tab:
        display_trends_and_forecast_tab(df_anual_melted, st.session_state.df_monthly_processed, stations_for_analysis)
    with descargas_tab:
        display_downloads_tab(df_anual_melted, df_monthly_filtered, stations_for_analysis)
    with tabla_estaciones_tab:
        display_station_table_tab(st.session_state.gdf_filtered, df_anual_melted, stations_for_analysis)

if __name__ == "__main__":
    main()
