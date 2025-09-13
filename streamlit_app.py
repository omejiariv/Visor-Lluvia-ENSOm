# ---
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
from rasterstats import zonal_stats
import xarray as xr
import locale
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
    # Se añade MPIO_SHP_COL para el mapa coroplético
    MPIO_SHP_COL = 'nombre_mpio' 

    # Nuevas constantes para índices climáticos
    SOI_COL = 'soi'
    IOD_COL = 'iod'
    SST_COL = 'temp_sst'
    MEDIA_COL = 'temp_media'

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
        if 'data_loaded' not in st.session_state:
            st.session_state.data_loaded = False
        if 'analysis_mode' not in st.session_state:
            st.session_state.analysis_mode = "Usar datos originales"
        if 'select_all_stations_state' not in st.session_state:
            st.session_state.select_all_stations_state = False
        if 'df_monthly_processed' not in st.session_state:
            st.session_state.df_monthly_processed = pd.DataFrame()
        if 'gdf_stations' not in st.session_state:
            st.session_state.gdf_stations = None
            st.session_state.df_precip_anual = None
            st.session_state.gdf_municipios = None
        if 'df_long' not in st.session_state:
            st.session_state.df_long = None
        if 'df_enso' not in st.session_state:
            st.session_state.df_enso = None
        if 'min_data_perc_slider' not in st.session_state: st.session_state.min_data_perc_slider = 0
        if 'altitude_multiselect' not in st.session_state: st.session_state.altitude_multiselect = []
        if 'regions_multiselect' not in st.session_state: st.session_state.regions_multiselect = []
        if 'municipios_multiselect' not in st.session_state: st.session_state.municipios_multiselect = []
        if 'celdas_multiselect' not in st.session_state: st.session_state.celdas_multiselect = []
        if 'station_multiselect' not in st.session_state: st.session_state.station_multiselect = []
        if 'exclude_na' not in st.session_state:
            st.session_state.exclude_na = False
        if 'exclude_zeros' not in st.session_state:
            st.session_state.exclude_zeros = False
        if 'gdf_municipal_stats' not in st.session_state:
            st.session_state.gdf_municipal_stats = None

# ---
# Funciones de Carga y Preprocesamiento
# ---
@st.cache_data
def parse_spanish_dates(date_series):
    """Convierte abreviaturas de meses en español a inglés."""
    months_es_to_en = {'ene': 'Jan', 'abr': 'Apr', 'ago': 'Aug', 'dic': 'Dec'}
    for es, en in months_es_to_en.items():
        date_series = date_series.str.replace(es, en, regex=False, case=False)
    return date_series

@st.cache_data
def load_data(file_path, sep=';', date_cols=None, lower_case=True, header=0, decimal='.'):
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
            df = pd.read_csv(io.BytesIO(content), sep=sep, encoding=encoding, parse_dates=date_cols, header=header, decimal=decimal)
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
    station_list = _df[Config.STATION_NAME_COL].unique()
    progress_bar = st.progress(0, text="Completando todas las series...")
    
    if not pd.api.types.is_datetime64_any_dtype(_df[Config.DATE_COL]):
        _df[Config.DATE_COL] = pd.to_datetime(_df[Config.DATE_COL], errors='coerce')
    
    for i, station in enumerate(station_list):
        df_station = _df[_df[Config.STATION_NAME_COL] == station].copy()
        
        df_station.dropna(subset=[Config.DATE_COL], inplace=True)
        df_station.drop_duplicates(subset=[Config.DATE_COL], inplace=True)
        
        if df_station.empty:
            continue

        df_station.set_index(Config.DATE_COL, inplace=True)

        start_date = df_station.index.min()
        end_date = df_station.index.max()
        full_date_range = pd.date_range(start=start_date, end=end_date, freq='MS')
        
        df_resampled = df_station.reindex(full_date_range)
        
        df_resampled[Config.PRECIPITATION_COL] = df_resampled[Config.PRECIPITATION_COL].interpolate(method='time')
        
        metadata_cols = [c for c in df_station.columns if c != Config.PRECIPITATION_COL]
        df_resampled[metadata_cols] = df_resampled[metadata_cols].ffill().bfill()
        
        df_resampled[Config.ORIGIN_COL] = df_resampled[Config.ORIGIN_COL].fillna('Completado')
        df_resampled[Config.STATION_NAME_COL] = station 
        df_resampled[Config.YEAR_COL] = df_resampled.index.year
        df_resampled[Config.MONTH_COL] = df_resampled.index.month
        df_resampled.reset_index(inplace=True)
        df_resampled.rename(columns={'index': Config.DATE_COL}, inplace=True) 
        all_completed_dfs.append(df_resampled)
        
        progress_bar.progress((i + 1) / len(station_list), text=f"Completando series... Estación: {station}")
    
    progress_bar.empty()
    
    if not all_completed_dfs:
        return pd.DataFrame()
        
    return pd.concat(all_completed_dfs, ignore_index=True)

@st.cache_data
def preprocess_data(uploaded_file_mapa, uploaded_file_precip, uploaded_zip_shapefile):
    """Procesa todos los archivos de entrada."""
    df_precip_anual = load_data(uploaded_file_mapa, sep=';', decimal=',')
    df_precip_mensual_raw = load_data(uploaded_file_precip, sep=';', decimal=',')
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
    if Config.ALTITUDE_COL in df_precip_anual.columns:
        df_precip_anual[Config.ALTITUDE_COL] = pd.to_numeric(df_precip_anual[Config.ALTITUDE_COL].astype(str).str.replace(',', '.'), errors='coerce')
    df_precip_anual.dropna(subset=[lon_col, lat_col], inplace=True)
    gdf_temp = gpd.GeoDataFrame(df_precip_anual,
                                  geometry=gpd.points_from_xy(df_precip_anual[lon_col], df_precip_anual[lat_col]),
                                  crs="EPSG:9377")
    gdf_stations = gdf_temp.to_crs("EPSG:4326")
    gdf_stations[Config.LONGITUDE_COL] = gdf_stations.geometry.x
    gdf_stations[Config.LATITUDE_COL] = gdf_stations.geometry.y

    df_precip_mensual = df_precip_mensual_raw.copy()
    station_cols = [col for col in df_precip_mensual.columns if col.isdigit()]
    if not station_cols:
        st.error("No se encontraron columnas de estación (ej: '12345') en el archivo de precipitación mensual.")
        return None, None, None, None, None

    id_vars_base = ['id', Config.DATE_COL, Config.YEAR_COL, Config.MONTH_COL, 'enso_año', 'enso_mes']
    id_vars_enso = [Config.ENSO_ONI_COL, Config.SST_COL, Config.MEDIA_COL, Config.SOI_COL, Config.IOD_COL]
    id_vars = id_vars_base + id_vars_enso
    
    df_precip_mensual[Config.DATE_COL] = pd.to_datetime(df_precip_mensual[Config.DATE_COL], format='%b-%y', errors='coerce')

    df_long = df_precip_mensual.melt(id_vars=[col for col in id_vars if col in df_precip_mensual.columns],
                                       value_vars=station_cols, var_name='id_estacion', value_name=Config.PRECIPITATION_COL)

    df_long[Config.PRECIPITATION_COL] = pd.to_numeric(df_long[Config.PRECIPITATION_COL], errors='coerce').round(0)
    for col in id_vars_enso:
        if col in df_long.columns:
            df_long[col] = pd.to_numeric(df_long[col], errors='coerce')

    df_long.dropna(subset=[Config.PRECIPITATION_COL], inplace=True)
    df_long[Config.ORIGIN_COL] = 'Original'

    gdf_stations['id_estacio'] = gdf_stations['id_estacio'].astype(str).str.strip()
    df_long['id_estacion'] = df_long['id_estacion'].astype(str).str.strip()
    station_mapping = gdf_stations.set_index('id_estacio')[Config.STATION_NAME_COL].to_dict()
    df_long[Config.STATION_NAME_COL] = df_long['id_estacion'].map(station_mapping)
    df_long.dropna(subset=[Config.STATION_NAME_COL], inplace=True)

    enso_cols = ['id', Config.DATE_COL, Config.ENSO_ONI_COL, Config.SST_COL, Config.MEDIA_COL, Config.SOI_COL, Config.IOD_COL]
    existing_enso_cols = [col for col in enso_cols if col in df_precip_mensual.columns]
    df_enso = df_precip_mensual[existing_enso_cols].drop_duplicates().copy()
    for col in [c for c in [Config.ENSO_ONI_COL, Config.SST_COL, Config.MEDIA_COL, Config.SOI_COL, Config.IOD_COL] if c in df_enso.columns]:
        df_enso[col] = pd.to_numeric(df_enso[col], errors='coerce')

    if Config.DATE_COL in df_enso.columns:
        df_enso[Config.DATE_COL] = df_enso[Config.DATE_COL]
        df_enso.dropna(subset=[Config.DATE_COL], inplace=True)
        
    return gdf_stations, df_precip_anual, gdf_municipios, df_long, df_enso

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
        "Topografía (OpenTopoMap)": {"tiles": "https://{s}.tile.opentomap.org/{z}/{x}/{y}.png", "attr": 'Map data: &copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors, <a href="http://viewfinderpanoramas.org">SRTM</a> | Map style: &copy; <a href="https://opentopomap.org">OpenTopoMap</a> (<a href="https://creativecommons.org/licenses/by-sa/3.0/">CC-BY-SA</a>)', "overlay": False},
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
# Funciones de las Pestañas de la UI
# ---
def display_welcome_tab():
    st.header("Bienvenido al Sistema de Información de Lluvias y Clima")
    st.markdown(Config.WELCOME_TEXT, unsafe_allow_html=True)
    if os.path.exists(Config.LOGO_PATH):
        st.image(Config.LOGO_PATH, width=400, caption="Corporación Cuenca Verde")

def display_spatial_distribution_tab(gdf_filtered, df_anual_melted, stations_for_analysis):
    st.header("Distribución espacial de las Estaciones de Lluvia (1970 - 2021)")
    
    if len(stations_for_analysis) == 0:
        st.warning("Por favor, seleccione al menos una estación para ver esta sección.")
        return
    
    selected_stations_str = f"{len(stations_for_analysis)} estaciones" if len(stations_for_analysis) > 1 else f"1 estación: {stations_for_analysis[0]}"
    st.info(f"Mostrando análisis para {selected_stations_str} en el período {st.session_state.year_range[0]} - {st.session_state.year_range[1]}.")

    if not df_anual_melted.empty:
        df_mean_precip = df_anual_melted.groupby(Config.STATION_NAME_COL)[Config.PRECIPITATION_COL].mean().round(0).reset_index()
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
                    base_chart = alt.Chart(df_monthly_filtered).mark_line(opacity=0.4, color='lightgray').encode(detail=f'{Config.STATION_NAME_COL}:N')
                    point_chart = alt.Chart(df_monthly_filtered).mark_point(filled=True, size=60).encode(x=alt.X(f'{Config.DATE_COL}:T', title='Fecha'), y=alt.Y(f'{Config.PRECIPITATION_COL}:Q', title='Precipitación (mm)'), tooltip=[alt.Tooltip(Config.DATE_COL, format='%Y-%m'), alt.Tooltip(Config.PRECIPITATION_COL, format='.0f'), Config.STATION_NAME_COL, Config.ORIGIN_COL, alt.Tooltip(f'{Config.MONTH_COL}:N', title="Mes")])
                    
                    if color_by == "Estación":
                        point_chart = point_chart.encode(color=alt.Color(f'{Config.STATION_NAME_COL}:N', legend=alt.Legend(title="Estaciones")))
                    else:
                        point_chart = point_chart.encode(color=alt.Color(f'month({Config.DATE_COL}):N', legend=alt.Legend(title="Meses"), scale=alt.Scale(scheme='tableau20')))
                    
                    final_chart = (base_chart + point_chart).properties(title=f"Serie de Precipitación Mensual ({st.session_state.year_range[0]} - {st.session_state.year_range[1]})").interactive()
                    st.altair_chart(final_chart, use_container_width=True)
                else:
                    st.subheader("Distribución de la Precipitación Mensual")
                    fig_box_monthly = px.box(df_monthly_filtered, x=Config.MONTH_COL, y=Config.PRECIPITATION_COL, color=Config.STATION_NAME_COL, title='Distribución de la Precipitación por Mes', labels={Config.MONTH_COL: 'Mes', Config.PRECIPITATION_COL: 'Precipitación Mensual (mm)', Config.STATION_NAME_COL: 'Estación'})
                    fig_box_monthly.update_layout(height=600)
                    st.plotly_chart(fig_box_monthly, use_container_width=True)
        with mensual_enso_tab:
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
            df_monthly_avg = df_monthly_filtered.groupby([Config.STATION_NAME_COL, Config.MONTH_COL])[Config.PRECIPITATION_COL].mean().round(0).reset_index()
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
            df_acumulada = df_anual_melted.groupby([Config.YEAR_COL, Config.STATION_NAME_COL])[Config.PRECIPITATION_COL].sum().round(0).reset_index()
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
            df_relacion = df_anual_melted.groupby(Config.STATION_NAME_COL)[Config.PRECIPITATION_COL].mean().round(0).reset_index()
            df_relacion = df_relacion.merge(st.session_state.gdf_filtered[[Config.STATION_NAME_COL, Config.ALTITUDE_COL]], on=Config.STATION_NAME_COL, how='inner')
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
                    data_year_kriging = data_year_kriging.merge(st.session_state.gdf_filtered[[Config.STATION_NAME_COL, Config.LATITUDE_COL, Config.LONGITUDE_COL]], on=Config.STATION_NAME_COL, how='inner')
                    data_year_kriging['tooltip'] = data_year_kriging.apply(
                        lambda row: f"<b>Estación:</b> {row[Config.STATION_NAME_COL]}<br>Municipio: {row[Config.MUNICIPALITY_COL]}<br>Ppt: {row[Config.PRECIPITATION_COL]:.0f} mm",
                        axis=1
                    )
                    lons, lats, vals = data_year_kriging[Config.LONGITUDE_COL].values, data_year_kriging[Config.LATITUDE_COL].values, data_year_kriging[Config.PRECIPITATION_COL].values
                    bounds = st.session_state.gdf_filtered.loc[st.session_state.gdf_filtered[Config.STATION_NAME_COL].isin(stations_for_analysis)].total_bounds
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


def display_station_table_tab(gdf_filtered, df_anual_melted, stations_for_analysis):
    st.header("Información Detallada de las Estaciones")
    selected_stations_str = f"{len(stations_for_analysis)} estaciones" if len(stations_for_analysis) > 1 else f"1 estación: {stations_for_analysis[0]}"
    st.info(f"Mostrando análisis para {selected_stations_str} en el período {st.session_state.year_range[0]} - {st.session_state.year_range[1]}.")
    
    if len(stations_for_analysis) == 0:
        st.warning("Por favor, seleccione al menos una estación para ver esta sección.")
        return
    if not df_anual_melted.empty:
        df_info_table = gdf_filtered[[Config.STATION_NAME_COL, Config.ALTITUDE_COL, Config.MUNICIPALITY_COL, Config.REGION_COL, Config.PERCENTAGE_COL, Config.LATITUDE_COL, Config.LONGITUDE_COL]].copy()
        df_mean_precip = df_anual_melted.groupby(Config.STATION_NAME_COL)[Config.PRECIPITATION_COL].mean().round(0).reset_index()
        df_mean_precip.rename(columns={Config.PRECIPITATION_COL: 'Precipitación media anual (mm)'}, inplace=True)
        df_info_table = df_info_table.merge(df_mean_precip, on=Config.STATION_NAME_COL, how='left')
        st.dataframe(df_info_table)
    else:
        st.info("No hay datos de precipitación anual para mostrar en la selección actual.")

def display_anomalies_tab(df_long, df_monthly_filtered, df_enso, stations_for_analysis):
    st.header("Análisis de Anomalías de Precipitación")
    selected_stations_str = f"{len(stations_for_analysis)} estaciones" if len(stations_for_analysis) > 1 else f"1 estación: {stations_for_analysis[0]}"
    st.info(f"Mostrando análisis para {selected_stations_str} en el período {st.session_state.year_range[0]} - {st.session_state.year_range[1]}.")
    
    if len(stations_for_analysis) == 0:
        st.warning("Por favor, seleccione al menos una estación para ver esta sección.")
        return

    if df_long is not None and not df_long.empty:
        df_long_filtered_stations = df_long[df_long[Config.STATION_NAME_COL].isin(stations_for_analysis)]
        if df_long_filtered_stations.empty:
            st.warning("No hay datos de anomalías para la selección actual.")
            return

        df_climatology = df_long_filtered_stations.groupby([Config.STATION_NAME_COL, Config.MONTH_COL])[Config.PRECIPITATION_COL].mean().reset_index().rename(columns={Config.PRECIPITATION_COL: 'precip_promedio_mes'})
        df_anomalias = pd.merge(df_monthly_filtered, df_climatology, on=[Config.STATION_NAME_COL, Config.MONTH_COL], how='left')
        df_anomalias['anomalia'] = df_anomalias[Config.PRECIPITATION_COL] - df_anomalias['precip_promedio_mes']

        if st.session_state.exclude_na:
            df_anomalias.dropna(subset=['anomalia'], inplace=True)

        if df_anomalias.empty or df_anomalias['anomalia'].isnull().all():
            st.warning("No hay suficientes datos históricos para las estaciones y el período seleccionado para calcular y mostrar las anomalías.")
            return

        anom_graf_tab, anom_fase_tab, anom_extremos_tab = st.tabs(["Gráfico de Anomalías", "Anomalías por Fase ENSO", "Tabla de Eventos Extremos"])

        with anom_graf_tab:
            avg_monthly_anom = df_anomalias.groupby([Config.DATE_COL, Config.MONTH_COL])['anomalia'].mean().reset_index()
            df_plot = pd.merge(avg_monthly_anom, st.session_state.df_enso[[Config.DATE_COL, Config.ENSO_ONI_COL]], on=Config.DATE_COL, how='left')
            fig = create_anomaly_chart(df_plot)
            st.plotly_chart(fig, use_container_width=True)

        with anom_fase_tab:
            if Config.ENSO_ONI_COL in df_anomalias.columns:
                df_anomalias_enso = df_anomalias.dropna(subset=[Config.ENSO_ONI_COL]).copy()
                conditions = [df_anomalias_enso[Config.ENSO_ONI_COL] >= 0.5, df_anomalias_enso[Config.ENSO_ONI_COL] <= -0.5]
                phases = ['El Niño', 'La Niña']
                df_anomalias_enso['enso_fase'] = np.select(conditions, phases, default='Neutral')
                fig_box = px.box(df_anomalias_enso, x='enso_fase', y='anomalia', color='enso_fase',
                                 title="Distribución de Anomalías de Precipitación por Fase ENSO",
                                 labels={'anomalia': 'Anomalía de Precipitación (mm)', 'enso_fase': 'Fase ENSO'},
                                 points='all')
                st.plotly_chart(fig_box, use_container_width=True)
            else:
                st.warning("La columna 'anomalia_oni' no está disponible para este análisis.")

        with anom_extremos_tab:
            st.subheader("Eventos Mensuales Extremos (Basado en Anomalías)")
            df_extremos = df_anomalias.dropna(subset=['anomalia']).copy()
            df_extremos['fecha'] = df_extremos[Config.DATE_COL].dt.strftime('%Y-%m')
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("##### 10 Meses más Secos")
                secos = df_extremos.nsmallest(10, 'anomalia')[['fecha', Config.STATION_NAME_COL, 'anomalia', Config.PRECIPITATION_COL, 'precip_promedio_mes']]
                st.dataframe(secos.rename(columns={Config.STATION_NAME_COL: 'Estación', 'anomalia': 'Anomalía (mm)', Config.PRECIPITATION_COL: 'Ppt. (mm)', 'precip_promedio_mes': 'Ppt. Media (mm)'}).round(0), use_container_width=True)
            with col2:
                st.markdown("##### 10 Meses más Húmedos")
                humedos = df_extremos.nlargest(10, 'anomalia')[['fecha', Config.STATION_NAME_COL, 'anomalia', Config.PRECIPITATION_COL, 'precip_promedio_mes']]
                st.dataframe(humedos.rename(columns={Config.STATION_NAME_COL: 'Estación', 'anomalia': 'Anomalía (mm)', Config.PRECIPITATION_COL: 'Ppt. (mm)', 'precip_promedio_mes': 'Ppt. Media (mm)'}).round(0), use_container_width=True)
    else:
        st.warning("No se puede realizar el análisis de anomalías. El DataFrame de datos mensuales no está disponible.")

def display_stats_tab(df_long, df_anual_melted, df_monthly_filtered, stations_for_analysis):
    st.header("Estadísticas de Precipitación")
    selected_stations_str = f"{len(stations_for_analysis)} estaciones" if len(stations_for_analysis) > 1 else f"1 estación: {stations_for_analysis[0]}"
    st.info(f"Mostrando análisis para {selected_stations_str} en el período {st.session_state.year_range[0]} - {st.session_state.year_range[1]}.")
    
    if len(stations_for_analysis) == 0:
        st.warning("Por favor, seleccione al menos una estación para ver esta sección.")
        return
    matriz_tab, resumen_mensual_tab, sintesis_tab = st.tabs(["Matriz de Disponibilidad", "Resumen Mensual", "Síntesis General"])

    with matriz_tab:
        st.subheader("Matriz de Disponibilidad de Datos Anual")
        original_data_counts = df_long[df_long[Config.STATION_NAME_COL].isin(stations_for_analysis)]
        original_data_counts = original_data_counts.groupby([Config.STATION_NAME_COL, Config.YEAR_COL]).size().reset_index(name='count')
        original_data_counts['porc_original'] = (original_data_counts['count'] / 12) * 100
        heatmap_original_df = original_data_counts.pivot(index=Config.STATION_NAME_COL, columns=Config.YEAR_COL, values='porc_original')
        heatmap_df = heatmap_original_df
        color_scale = "Greens"
        title_text = "Disponibilidad Promedio de Datos Originales"
        if st.session_state.analysis_mode == "Completar series (interpolación)":
            view_mode = st.radio("Seleccione la vista de la matriz:", ("Porcentaje de Datos Originales", "Porcentaje de Datos Completados"), horizontal=True)
            if view_mode == "Porcentaje de Datos Completados":
                completed_data = st.session_state.df_monthly_processed[(st.session_state.df_monthly_processed[Config.STATION_NAME_COL].isin(stations_for_analysis)) & (st.session_state.df_monthly_processed[Config.ORIGIN_COL] == 'Completado')]
                if not completed_data.empty:
                    completed_counts = completed_data.groupby([Config.STATION_NAME_COL, Config.YEAR_COL]).size().reset_index(name='count')
                    completed_counts['porc_completado'] = (completed_counts['count'] / 12) * 100
                    heatmap_df = completed_counts.pivot(index=Config.STATION_NAME_COL, columns=Config.YEAR_COL, values='porc_completado')
                    color_scale = "Reds"
                    title_text = "Disponibilidad Promedio de Datos Completados"
                else: heatmap_df = pd.DataFrame()
        if not heatmap_df.empty:
            avg_availability = heatmap_df.stack().mean()
            logo_col, metric_col = st.columns([1, 5])
            with logo_col:
                if os.path.exists(Config.LOGO_DROP_PATH): st.image(Config.LOGO_DROP_PATH, width=50)
            with metric_col: st.metric(label=title_text, value=f"{avg_availability:.1f}%")
            styled_df = heatmap_df.style.background_gradient(cmap=color_scale, axis=None, vmin=0, vmax=100).format("{:.0f}%", na_rep="-").set_table_styles([
                {'selector': 'th', 'props': [('background-color', '#333'), ('color', 'white'), ('font-size', '14px')]},
                {'selector': 'td', 'props': [('text-align', 'center')]}])
            st.dataframe(styled_df, use_container_width=True)
        else:
            st.info("No hay datos para mostrar en la matriz con la selección actual.")

    with resumen_mensual_tab:
        st.subheader("Resumen de Estadísticas Mensuales por Estación")
        if not df_monthly_filtered.empty:
            summary_data = []
            for station_name, group in df_monthly_filtered.groupby(Config.STATION_NAME_COL):
                max_row = group.loc[group[Config.PRECIPITATION_COL].idxmax()]
                min_row = group.loc[group[Config.PRECIPITATION_COL].idxmin()]
                summary_data.append({
                    "Estación": station_name,
                    "Ppt. Máxima Mensual (mm)": max_row[Config.PRECIPITATION_COL],
                    "Fecha Máxima": max_row[Config.DATE_COL].strftime('%Y-%m'),
                    "Ppt. Mínima Mensual (mm)": min_row[Config.PRECIPITATION_COL],
                    "Fecha Mínima": min_row[Config.DATE_COL].strftime('%Y-%m'),
                    "Promedio Mensual (mm)": group[Config.PRECIPITATION_COL].mean()
                })
            summary_df = pd.DataFrame(summary_data)
            st.dataframe(summary_df.round(0), use_container_width=True)
        else:
            st.info("No hay datos para mostrar el resumen mensual.")

    with sintesis_tab:
        st.subheader("Síntesis General de Precipitación")
        if not df_monthly_filtered.empty and not df_anual_melted.empty:
            max_annual_row = df_anual_melted.loc[df_anual_melted[Config.PRECIPITATION_COL].idxmax()]
            max_monthly_row = df_monthly_filtered.loc[df_monthly_filtered[Config.PRECIPITATION_COL].idxmax()]
            meses_map = {1: 'Enero', 2: 'Febrero', 3: 'Marzo', 4: 'Abril', 5: 'Mayo', 6: 'Junio', 7: 'Julio', 8: 'Agosto', 9: 'Septiembre', 10: 'Octubre', 11: 'Noviembre', 12: 'Diciembre'}
            max_monthly_row['nom_mes'] = meses_map.get(max_monthly_row[Config.MONTH_COL])
            col1, col2 = st.columns(2)
            with col1:
                st.metric(
                    "Máxima Ppt. Anual Registrada",
                    f"{max_annual_row[Config.PRECIPITATION_COL]:.0f} mm",
                    f"{max_annual_row[Config.STATION_NAME_COL]} (Año {max_annual_row[Config.YEAR_COL]})"
                )
            with col2:
                st.metric(
                    "Máxima Ppt. Mensual Registrada",
                    f"{max_monthly_row[Config.PRECIPITATION_COL]:.0f} mm",
                    f"{max_monthly_row[Config.STATION_NAME_COL]} ({max_monthly_row['nom_mes']} {max_monthly_row[Config.DATE_COL].year})"
                )
        else:
            st.info("No hay datos para mostrar la síntesis general.")

def display_correlation_tab(df_monthly_filtered, stations_for_analysis):
    st.header("Análisis de Correlación")
    selected_stations_str = f"{len(stations_for_analysis)} estaciones" if len(stations_for_analysis) > 1 else f"1 estación: {stations_for_analysis[0]}"
    st.info(f"Mostrando análisis para {selected_stations_str} en el período {st.session_state.year_range[0]} - {st.session_state.year_range[1]}.")
    
    if len(stations_for_analysis) == 0:
        st.warning("Por favor, seleccione al menos una estación para ver esta sección.")
        return

    enso_corr_tab, station_corr_tab, indices_climaticos_tab = st.tabs(["Correlación con ENSO", "Comparación entre Estaciones", "Correlación con Índices Climáticos"])
    
    with enso_corr_tab:
        if Config.ENSO_ONI_COL not in df_monthly_filtered.columns:
            st.warning("No se puede realizar el análisis de correlación con ENSO porque la columna 'anomalia_oni' no fue encontrada.")
            return
        
        df_corr_analysis = df_monthly_filtered[[Config.DATE_COL, Config.STATION_NAME_COL, Config.PRECIPITATION_COL, Config.ENSO_ONI_COL]].dropna(subset=[Config.PRECIPITATION_COL, Config.ENSO_ONI_COL])
        
        if df_corr_analysis.empty:
            st.warning("No hay datos coincidentes entre la precipitación y el ENSO para la selección actual.")
            return

        analysis_level = st.radio("Nivel de Análisis de Correlación con ENSO", ["Promedio de la selección", "Por Estación Individual"], key="enso_corr_level")
        
        df_plot_corr = pd.DataFrame()
        title_text = ""
        if analysis_level == "Por Estación Individual":
            station_to_corr = st.selectbox("Seleccione Estación:", options=sorted(df_corr_analysis[Config.STATION_NAME_COL].unique()), key="enso_corr_station")
            if station_to_corr:
                df_plot_corr = df_corr_analysis[df_corr_analysis[Config.STATION_NAME_COL] == station_to_corr]
                title_text = f"Correlación para la estación: {station_to_corr}"
        else:
            df_plot_corr = df_corr_analysis.groupby(Config.DATE_COL).agg(
                precipitation=(Config.PRECIPITATION_COL, 'mean'),
                anomalia_oni=(Config.ENSO_ONI_COL, 'first')
            ).reset_index()
            title_text = "Correlación para el promedio de las estaciones seleccionadas"

        if len(df_plot_corr) > 2:
            corr, p_value = stats.pearsonr(df_plot_corr[Config.ENSO_ONI_COL], df_plot_corr['precipitation'])
            st.subheader(title_text)
            col1, col2 = st.columns(2)
            col1.metric("Coeficiente de Correlación (r)", f"{corr:.3f}")
            col2.metric("Significancia (valor p)", f"{p_value:.4f}")
            if p_value < 0.05:
                st.success("La correlación es estadísticamente significativa, lo que sugiere una relación lineal entre las variables.")
            else:
                st.warning("La correlación no es estadísticamente significativa. No hay evidencia de una relación lineal fuerte.")
            fig_corr = px.scatter(
                df_plot_corr, x=Config.ENSO_ONI_COL, y='precipitation', trendline='ols',
                title="Gráfico de Dispersión: Precipitación vs. Anomalía ONI",
                labels={Config.ENSO_ONI_COL: 'Anomalía ONI (°C)', 'precipitation': 'Precipitación Mensual (mm)'}
            )
            st.plotly_chart(fig_corr, use_container_width=True)

    with station_corr_tab:
        if len(stations_for_analysis) < 2:
            st.info("Seleccione al menos dos estaciones para comparar la correlación entre ellas.")
        else:
            st.subheader("Correlación de Precipitación entre dos Estaciones")
            station_options = sorted(stations_for_analysis)
            col1, col2 = st.columns(2)
            station1_name = col1.selectbox("Estación 1:", options=station_options, key="corr_station_1")
            station2_name = col2.selectbox("Estación 2:", options=station_options, key="corr_station_2")

            if station1_name and station2_name and station1_name != station2_name:
                df_station1 = df_monthly_filtered[df_monthly_filtered[Config.STATION_NAME_COL] == station1_name][[Config.DATE_COL, Config.PRECIPITATION_COL]]
                df_station2 = df_monthly_filtered[df_monthly_filtered[Config.STATION_NAME_COL] == station2_name][[Config.DATE_COL, Config.PRECIPITATION_COL]]
                
                df_merged = pd.merge(df_station1, df_station2, on=Config.DATE_COL, suffixes=('_1', '_2')).dropna()
                df_merged.rename(columns={f'{Config.PRECIPITATION_COL}_1': station1_name, f'{Config.PRECIPITATION_COL}_2': station2_name}, inplace=True)
                
                if not df_merged.empty and len(df_merged) > 2:
                    corr, p_value = stats.pearsonr(df_merged[station1_name], df_merged[station2_name])
                    
                    st.markdown(f"#### Resultados de la correlación ({station1_name} vs. {station2_name})")
                    st.metric("Coeficiente de Correlación (r)", f"{corr:.3f}")
                    
                    if p_value < 0.05:
                        st.success("La correlación es estadísticamente significativa (p < 0.05).")
                    else:
                        st.warning("La correlación no es estadísticamente significativa (p ≥ 0.05).")
                    
                    slope, intercept, _, _, _ = stats.linregress(df_merged[station1_name], df_merged[station2_name])
                    st.info(f"Ecuación de regresión: y = {slope:.2f}x + {intercept:.2f}")

                    fig_scatter = px.scatter(
                        df_merged, x=station1_name, y=station2_name, trendline='ols',
                        title=f'Dispersión de Precipitación: {station1_name} vs. {station2_name}',
                        labels={station1_name: f'Precipitación en {station1_name} (mm)', station2_name: f'Precipitación en {station2_name} (mm)'}
                    )
                    st.plotly_chart(fig_scatter, use_container_width=True)
                else:
                    st.warning("No hay suficientes datos superpuestos para calcular la correlación para las estaciones seleccionadas.")

    with indices_climaticos_tab:
        st.subheader("Análisis de Correlación con Índices Climáticos")
        
        df_corr_indices = df_monthly_filtered.copy()
        potential_indices = [Config.ENSO_ONI_COL, Config.SOI_COL, Config.IOD_COL]

        available_indices_cols = [
            idx for idx in potential_indices 
            if idx in df_corr_indices.columns and not df_corr_indices[idx].isnull().all()
        ]

        if not available_indices_cols:
            st.warning("No se encontraron datos para los índices climáticos (ONI, SOI, IOD) en la selección actual.")
            return

        selected_index = st.selectbox("Seleccione un índice climático:", available_indices_cols, key="indices_climaticos_selectbox")
        selected_station_corr_indices = st.selectbox("Seleccione una estación de precipitación:", options=sorted(stations_for_analysis), key="station_for_index_corr")

        if selected_index and selected_station_corr_indices:
            df_plot_indices = df_corr_indices[
                df_corr_indices[Config.STATION_NAME_COL] == selected_station_corr_indices
            ][[Config.DATE_COL, Config.PRECIPITATION_COL, selected_index]].dropna()

            if not df_plot_indices.empty and len(df_plot_indices) > 2:
                corr, p_value = stats.pearsonr(df_plot_indices[selected_index], df_plot_indices[Config.PRECIPITATION_COL])
                st.markdown(f"#### Resultados de la correlación ({selected_index.upper()} vs. Precipitación de {selected_station_corr_indices})")
                
                m_col1, m_col2 = st.columns(2)
                m_col1.metric("Coeficiente de Correlación (r)", f"{corr:.3f}")
                m_col2.metric("Valor p", f"{p_value:.4f}")
                
                if p_value < 0.05:
                    st.success("La correlación es estadísticamente significativa (p < 0.05).")
                else:
                    st.warning("La correlación no es estadísticamente significativa (p ≥ 0.05).")
                
                fig_scatter_indices = px.scatter(
                    df_plot_indices, x=selected_index, y=Config.PRECIPITATION_COL, trendline='ols',
                    title=f'Dispersión: {selected_index.upper()} vs. Precipitación de {selected_station_corr_indices}',
                    labels={selected_index: f'Valor del Índice {selected_index.upper()}', Config.PRECIPITATION_COL: 'Precipitación Mensual (mm)'}
                )
                st.plotly_chart(fig_scatter_indices, use_container_width=True)
            else:
                st.warning("No hay suficientes datos superpuestos entre la estación y el índice para calcular la correlación.")

def display_enso_tab(df_monthly_filtered, df_enso, gdf_filtered, stations_for_analysis):
    st.header("Análisis de Precipitación y el Fenómeno ENSO")
    selected_stations_str = f"{len(stations_for_analysis)} estaciones" if len(stations_for_analysis) > 1 else f"1 estación: {stations_for_analysis[0]}"
    st.info(f"Mostrando análisis para {selected_stations_str} en el período {st.session_state.year_range[0]} - {st.session_state.year_range[1]}.")
    
    enso_series_tab, enso_anim_tab, indices_multiples_tab = st.tabs(["Series de Tiempo ENSO", "Mapa Interactivo ENSO", "Análisis de Índices Múltiples"])

    with enso_series_tab:
        if df_enso.empty:
            st.warning("No se encontraron datos del fenómeno ENSO en el archivo de precipitación cargado.")
            return
        enso_vars_available = {
            Config.ENSO_ONI_COL: 'Anomalía ONI',
            Config.SST_COL: 'Temp. Superficial del Mar (SST)',
            Config.MEDIA_COL: 'Temp. Media'
        }
        available_tabs = [name for var, name in enso_vars_available.items() if var in df_enso.columns]
        if not available_tabs:
            st.warning("No hay variables ENSO disponibles en el archivo de datos para visualizar.")
            return
        enso_variable_tabs = st.tabs(available_tabs)
        for i, var_name in enumerate(available_tabs):
            with enso_variable_tabs[i]:
                var_code = [code for code, name in enso_vars_available.items() if name == var_name][0]
                df_enso_filtered = df_enso[(df_enso[Config.DATE_COL].dt.year >= st.session_state.year_range[0]) & (df_enso[Config.DATE_COL].dt.year <= st.session_state.year_range[1]) & (df_enso[Config.DATE_COL].dt.month.isin(st.session_state.meses_numeros))]
                if not df_enso_filtered.empty and var_code in df_enso_filtered.columns and not df_enso_filtered[var_code].isnull().all():
                    fig_enso_series = px.line(df_enso_filtered, x=Config.DATE_COL, y=var_code, title=f"Serie de Tiempo para {var_name}")
                    st.plotly_chart(fig_enso_series, use_container_width=True)
                else:
                    st.warning(f"No hay datos disponibles para '{var_code}' en el período seleccionado.")

    with enso_anim_tab:
        st.subheader("Explorador Mensual del Fenómeno ENSO")
        if df_enso.empty or st.session_state.gdf_filtered is None or st.session_state.gdf_filtered.empty:
            st.warning("No hay datos disponibles para generar esta visualización.")
            return
        if Config.ENSO_ONI_COL not in df_enso.columns:
            st.warning("La columna 'anomalia_oni' es necesaria para esta visualización y no se encontró.")
            return
        controls_col, map_col = st.columns([1, 3])
        enso_anim_data = df_enso[[Config.DATE_COL, Config.ENSO_ONI_COL]].copy()
        enso_anim_data.dropna(subset=[Config.ENSO_ONI_COL], inplace=True)
        conditions = [enso_anim_data[Config.ENSO_ONI_COL] >= 0.5, enso_anim_data[Config.ENSO_ONI_COL] <= -0.5]
        phases = ['El Niño', 'La Niña']
        enso_anim_data['fase'] = np.select(conditions, phases, default='Neutral')
        enso_anim_data_filtered = enso_anim_data[(enso_anim_data[Config.DATE_COL].dt.year >= st.session_state.year_range[0]) & (enso_anim_data[Config.DATE_COL].dt.year <= st.session_state.year_range[1])]

        with controls_col:
            st.markdown("##### Controles de Mapa")
            selected_base_map_config, selected_overlays_config = display_map_controls(st, "enso_anim")
            st.markdown("##### Selección de Fecha")
            available_dates = sorted(enso_anim_data_filtered[Config.DATE_COL].unique())
            if available_dates:
                selected_date = st.select_slider("Seleccione una fecha (Año-Mes)", options=available_dates, format_func=lambda date: date.strftime('%Y-%m'))
                phase_info = enso_anim_data_filtered[enso_anim_data_filtered[Config.DATE_COL] == selected_date]
                if not phase_info.empty:
                    current_phase = phase_info['fase'].iloc[0]
                    current_oni = phase_info[Config.ENSO_ONI_COL].iloc[0]
                    st.metric(f"Fase ENSO en {selected_date.strftime('%Y-%m')}", current_phase, f"Anomalía ONI: {current_oni:.2f}°C")
                else:
                    st.warning("No hay datos de ENSO para el período seleccionado.")

        with map_col:
            if available_dates:
                m_enso = folium.Map(location=[4.57, -74.29], zoom_start=5, tiles=selected_base_map_config.get("tiles", "OpenStreetMap"), attr=selected_base_map_config.get("attr", None))
                phase_color_map = {'El Niño': 'red', 'La Niña': 'blue', 'Neutral': 'grey'}
                marker_color = phase_color_map.get(current_phase, 'black')
                for _, station in gdf_filtered.iterrows():
                    folium.Marker(
                        location=[station[Config.LATITUDE_COL], station[Config.LONGITUDE_COL]],
                        tooltip=f"{station[Config.STATION_NAME_COL]}<br>Fase: {current_phase}",
                        icon=folium.Icon(color=marker_color, icon='cloud')
                    ).add_to(m_enso)
                bounds = gdf_filtered.total_bounds
                m_enso.fit_bounds([[bounds[1], bounds[0]], [bounds[3], bounds[2]]])
                for layer_config in selected_overlays_config:
                    folium.raster_layers.WmsTileLayer(url=layer_config["url"], layers=layer_config["layers"], fmt='image/png', transparent=layer_config.get("transparent", False), overlay=True, control=True, name=layer_config["attr"]).add_to(m_enso)
                folium.LayerControl().add_to(m_enso)
                folium_static(m_enso, height=700, width="100%")
    
    with indices_multiples_tab:
        st.subheader("Análisis de Índices Climáticos Múltiples")

        index_cols = [Config.ENSO_ONI_COL, Config.SOI_COL, Config.IOD_COL]
        available_cols_in_df = [col for col in index_cols if col in st.session_state.df_long.columns]

        if not available_cols_in_df:
            st.warning("No se encontraron columnas de índices climáticos (ONI, SOI, IOD) en los datos cargados.")
            return

        df_indices = st.session_state.df_long[[Config.DATE_COL] + available_cols_in_df].drop_duplicates().reset_index(drop=True)

        df_indices_filtered = df_indices[
            (df_indices[Config.DATE_COL].dt.year >= st.session_state.year_range[0]) & 
            (df_indices[Config.DATE_COL].dt.year <= st.session_state.year_range[1]) &
            (df_indices[Config.DATE_COL].dt.month.isin(st.session_state.meses_numeros))
        ].copy()

        if df_indices_filtered.empty:
            st.warning("No hay datos de índices climáticos para el período seleccionado.")
            return

        available_indices_to_plot = [
            col for col in available_cols_in_df 
            if not df_indices_filtered[col].isnull().all()
        ]

        if not available_indices_to_plot:
            st.warning("No hay índices climáticos con datos válidos en el período seleccionado para graficar.")
            return
        
        selected_indices = st.multiselect("Seleccione los índices a graficar:",
                                          options=available_indices_to_plot,
                                          default=available_indices_to_plot,
                                          key='multiselect_indices_plot')
        
        if selected_indices:
            fig = go.Figure()
            for index in selected_indices:
                fig.add_trace(go.Scatter(x=df_indices_filtered[Config.DATE_COL], y=df_indices_filtered[index], mode='lines', name=index.upper()))
            
            fig.update_layout(
                title="Series de Tiempo de Índices Climáticos",
                xaxis_title="Fecha",
                yaxis_title="Valor del Índice",
                height=600,
                legend_title="Índices"
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Seleccione uno o más índices para graficar.")

def display_trends_and_forecast_tab(df_anual_melted, df_monthly_to_process, stations_for_analysis):
    st.header("Análisis de Tendencias y Pronósticos")
    selected_stations_str = f"{len(stations_for_analysis)} estaciones" if len(stations_for_analysis) > 1 else f"1 estación: {stations_for_analysis[0]}"
    st.info(f"Mostrando análisis para {selected_stations_str} en el período {st.session_state.year_range[0]} - {st.session_state.year_range[1]}.")

    if len(stations_for_analysis) == 0:
        st.warning("Por favor, seleccione al menos una estación para ver esta sección.")
        return
    tendencia_individual_tab, tendencia_tabla_tab, pronostico_sarima_tab, pronostico_prophet_tab = st.tabs(["Análisis Lineal", "Tabla Comparativa", "Pronóstico SARIMA", "Pronóstico Prophet"])

    with tendencia_individual_tab:
        st.subheader("Tendencia de Precipitación Anual")
        analysis_type = st.radio("Tipo de Análisis de Tendencia:", ["Promedio de la selección", "Estación individual"], horizontal=True)
        df_to_analyze = None
        if analysis_type == "Promedio de la selección":
            df_to_analyze = df_anual_melted.groupby(Config.YEAR_COL)[Config.PRECIPITATION_COL].mean().reset_index()
        else:
            station_to_analyze = st.selectbox("Seleccione una estación para analizar:", options=stations_for_analysis)
            if station_to_analyze: df_to_analyze = df_anual_melted[df_anual_melted[Config.STATION_NAME_COL] == station_to_analyze]
        if df_to_analyze is not None and len(df_to_analyze) > 2:
            df_to_analyze['año_num'] = pd.to_numeric(df_to_analyze[Config.YEAR_COL])
            slope, intercept, r_value, p_value, std_err = stats.linregress(df_to_analyze['año_num'], df_to_analyze[Config.PRECIPITATION_COL])
            tendencia_texto = "aumentando" if slope > 0 else "disminuyendo"
            significancia_texto = "**estadísticamente significativa**" if p_value < 0.05 else "no es estadísticamente significativa"
            st.markdown(f"La tendencia de la precipitación es de **{slope:.2f} mm/año** (es decir, está {tendencia_texto}). Con un valor p de **{p_value:.3f}**, esta tendencia **{significancia_texto}**.")
            df_to_analyze['tendencia'] = slope * df_to_analyze['año_num'] + intercept
            fig_tendencia = px.scatter(df_to_analyze, x='año_num', y=Config.PRECIPITATION_COL, title=f'Tendencia de la Precipitación Anual ({st.session_state.year_range[0]} - {st.session_state.year_range[1]})')
            fig_tendencia.add_trace(go.Scatter(x=df_to_analyze['año_num'], y=df_to_analyze['tendencia'], mode='lines', name='Línea de Tendencia', line=dict(color='red')))
            fig_tendencia.update_layout(xaxis_title="Año", yaxis_title="Precipitación Anual (mm)")
            st.plotly_chart(fig_tendencia, use_container_width=True)
            
            csv_data = df_to_analyze.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="Descargar datos de Tendencia Anual",
                data=csv_data,
                file_name=f'tendencia_anual_{analysis_type.replace(" ", "_")}.csv',
                mime='text/csv',
                key='download-anual-tendencia'
            )
        else:
            st.warning("No hay suficientes datos en el período seleccionado para calcular una tendencia.")

    with tendencia_tabla_tab:
        st.subheader("Tabla Comparativa de Tendencias de Precipitación Anual")
        if st.button("Calcular Tendencias para Todas las Estaciones Seleccionadas"):
            with st.spinner("Calculando tendencias..."):
                results = []
                df_anual_calc = df_anual_melted.copy()
                df_anual_calc.dropna(subset=[Config.PRECIPITATION_COL], inplace=True)
                if st.session_state.exclude_zeros: df_anual_calc = df_anual_calc[df_anual_calc[Config.PRECIPITATION_COL] > 0]
                for station in stations_for_analysis:
                    station_data = df_anual_calc[df_anual_calc[Config.STATION_NAME_COL] == station]
                    if len(station_data) > 2:
                        station_data['año_num'] = pd.to_numeric(station_data[Config.YEAR_COL])
                        slope, _, _, p_value, _ = stats.linregress(station_data['año_num'], station_data[Config.PRECIPITATION_COL])
                        interpretation = "Significativa (p < 0.05)" if p_value < 0.05 else "No Significativa (p ≥ 0.05)"
                        results.append({
                            "Estación": station, "Tendencia (mm/año)": slope, "Valor p": p_value,
                            "Interpretación": interpretation, "Años Analizados": len(station_data)
                        })
                    else:
                        results.append({
                            "Estación": station, "Tendencia (mm/año)": np.nan, "Valor p": np.nan,
                            "Interpretación": "Datos insuficientes", "Años Analizados": len(station_data)
                        })
                if results:
                    results_df = pd.DataFrame(results)
                    def style_p_value(val):
                        color = 'lightgreen' if val < 0.05 else 'lightcoral'
                        return f'background-color: {color}'
                    st.dataframe(results_df.style.format({
                        "Tendencia (mm/año)": "{:.2f}",
                        "Valor p": "{:.4f}"
                    }).applymap(style_p_value, subset=['Valor p']), use_container_width=True)
                    
                    csv_data = results_df.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        label="Descargar tabla de tendencias en CSV",
                        data=csv_data,
                        file_name='tabla_tendencias.csv',
                        mime='text/csv',
                        key='download-tabla-tendencias'
                    )
                else:
                    st.warning("No se pudieron calcular tendencias para las estaciones seleccionadas.")

    with pronostico_sarima_tab:
        st.subheader("Pronóstico de Precipitación Mensual (Modelo SARIMA)")
        with st.expander("¿Cómo funciona SARIMA?"):
            st.markdown("""
                El modelo **SARIMA** (Seasonal Auto-Regressive Integrated Moving Average) es un método estadístico que utiliza datos históricos para predecir valores futuros.
                - **Autoregresivo (AR):** La predicción depende de valores pasados de la serie.
                - **Integrado (I):** Utiliza la diferencia entre valores para hacer la serie estacionaria (eliminar tendencias).
                - **Media Móvil (MA):** La predicción se basa en errores de pronósticos pasados.
                - **Estacional (S):** Captura patrones que se repiten en ciclos, como la variación anual de las lluvias.
            """)
        
        station_to_forecast = st.selectbox("Seleccione una estación para el pronóstico:", options=stations_for_analysis, key="sarima_station_select")
        forecast_horizon = st.slider("Meses a pronosticar:", 12, 36, 12, step=12, key="sarima_forecast_horizon_slider")

        if not df_monthly_to_process.empty and len(df_monthly_to_process[df_monthly_to_process[Config.STATION_NAME_COL] == station_to_forecast]) < 24:
            st.warning("Se necesitan al menos 24 puntos de datos para el pronóstico SARIMA. Por favor, ajuste la selección de años.")
        elif not df_monthly_to_process.empty:
            with st.spinner(f"Entrenando modelo y generando pronóstico para {station_to_forecast}..."):
                try:
                    ts_data = df_monthly_to_process[df_monthly_to_process[Config.STATION_NAME_COL] == station_to_forecast][[Config.DATE_COL, Config.PRECIPITATION_COL]].copy()
                    ts_data = ts_data.set_index(Config.DATE_COL).sort_index()
                    ts_data = ts_data[Config.PRECIPITATION_COL].asfreq('MS')

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
                    
                    forecast_df = pd.DataFrame({
                        'fecha': forecast_mean.index,
                        'pronostico': forecast_mean.values,
                        'limite_inferior': forecast_ci.iloc[:, 0].values,
                        'limite_superior': forecast_ci.iloc[:, 1].values
                    })
                    csv_data = forecast_df.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        label="Descargar Pronóstico SARIMA en CSV",
                        data=csv_data,
                        file_name=f'pronostico_sarima_{station_to_forecast.replace(" ", "_")}.csv',
                        mime='text/csv',
                        key='download-sarima'
                    )

                except Exception as e:
                    st.error(f"Ocurrió un error al generar el pronóstico con Prophet. Esto puede deberse a que la serie de datos es demasiado corta o inestable. Error: {e}")
        else:
            st.info("Por favor, cargue datos para generar un pronóstico.")

    with pronostico_prophet_tab:
        st.subheader("Pronóstico de Precipitación Mensual (Modelo Prophet)")
        with st.expander("¿Cómo funciona Prophet?"):
            st.markdown("""
                **Prophet**, desarrollado por Facebook, es un procedimiento para pronosticar series de tiempo.
                - Se basa en un modelo aditivo en el que se ajustan las tendencias no lineales con la estacionalidad anual y semanal, además de los efectos de festivos.
                - Es especialmente útil para series de tiempo que tienen una fuerte estacionalidad y múltiples ciclos.
                - Es más robusto que otros modelos a datos faltantes o atípicos.
            """)
        
        station_to_forecast_prophet = st.selectbox("Seleccione una estación para el pronóstico:", options=stations_for_analysis, key="prophet_station_select", help="El pronóstico se realiza para una única serie de tiempo con Prophet.")
        forecast_horizon_prophet = st.slider("Meses a pronosticar:", 12, 36, 12, step=12, key="prophet_forecast_horizon_slider")

        if not df_monthly_to_process.empty and len(df_monthly_to_process[df_monthly_to_process[Config.STATION_NAME_COL] == station_to_forecast_prophet]) < 24:
            st.warning("Se necesitan al menos 24 puntos de datos para que Prophet funcione correctamente. Por favor, ajuste la selección de años.")
        elif not df_monthly_to_process.empty:
            with st.spinner(f"Entrenando modelo Prophet y generando pronóstico para {station_to_forecast_prophet}..."):
                try:
                    ts_data_prophet = df_monthly_to_process[df_monthly_to_process[Config.STATION_NAME_COL] == station_to_forecast_prophet][[Config.DATE_COL, Config.PRECIPITATION_COL]].copy()
                    ts_data_prophet.rename(columns={Config.DATE_COL: 'ds', Config.PRECIPITATION_COL: 'y'}, inplace=True)
                    model_prophet = Prophet()
                    model_prophet.fit(ts_data_prophet)
                    future = model_prophet.make_future_dataframe(periods=forecast_horizon_prophet, freq='MS')
                    forecast_prophet = model_prophet.predict(future)
                    
                    st.success("Pronóstico generado exitosamente.")
                    fig_prophet = plot_plotly(model_prophet, forecast_prophet)
                    fig_prophet.update_layout(title=f"Pronóstico de Precipitación con Prophet para {station_to_forecast_prophet}", yaxis_title="Precipitación (mm)")
                    st.plotly_chart(fig_prophet, use_container_width=True)

                    csv_data = forecast_prophet[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].to_csv(index=False).encode('utf-8')
                    st.download_button(
                        label="Descargar Pronóstico Prophet en CSV",
                        data=csv_data,
                        file_name=f'pronostico_prophet_{station_to_forecast_prophet.replace(" ", "_")}.csv',
                        mime='text/csv',
                        key='download-prophet'
                    )

                except Exception as e:
                    st.error(f"Ocurrió un error al generar el pronóstico con Prophet. Esto puede deberse a que la serie de datos es demasiado corta o inestable. Error: {e}")
        else:
            st.info("Por favor, cargue datos para generar un pronóstico.")

def display_downloads_tab(df_anual_melted, df_monthly_filtered, stations_for_analysis):
    st.header("Opciones de Descarga")
    if len(stations_for_analysis) == 0:
        st.warning("Por favor, seleccione al menos una estación para activar las descargas.")
        return
    @st.cache_data
    def convert_df_to_csv(df):
        return df.to_csv(index=False).encode('utf-8')

    st.markdown("**Datos de Precipitación Anual (Filtrados)**")
    csv_anual = convert_df_to_csv(df_anual_melted)
    st.download_button("Descargar CSV Anual", csv_anual, 'precipitacion_anual.csv', 'text/csv', key='download-anual')

    st.markdown("**Datos de Precipitación Mensual (Filtrados)**")
    csv_mensual = convert_df_to_csv(df_monthly_filtered)
    st.download_button("Descargar CSV Mensual", csv_mensual, 'precipitacion_mensual.csv', 'text/csv', key='download-mensual')

    if st.session_state.analysis_mode == "Completar series (interpolación)":
        st.markdown("**Datos de Precipitación Mensual (Series Completadas y Filtradas)**")
        csv_completado = convert_df_to_csv(st.session_state.df_monthly_processed)
        st.download_button("Descargar CSV con Series Completadas", csv_completado, 'precipitacion_mensual_completada.csv', 'text/csv', key='download-completado')
    else:
        st.info("Para descargar las series completadas, seleccione la opción 'Completar series (interpolación)' en el panel lateral.")

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

    with st.sidebar.expander("**Cargar Archivos**", expanded=True):
        uploaded_file_mapa = st.file_uploader("1. Cargar archivo de estaciones (mapaCVENSO.csv)", type="csv")
        uploaded_file_precip = st.file_uploader("2. Cargar archivo de precipitación mensual y ENSO (DatosPptnmes_ENSO.csv)", type="csv")
        uploaded_zip_shapefile = st.file_uploader("3. Cargar shapefile de municipios (.zip)", type="zip")
        if st.button("Recargar Datos"):
            st.session_state.data_loaded = False
            st.cache_data.clear()
            st.rerun()

    if not st.session_state.data_loaded:
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
        filtered_stations_temp = st.session_state.gdf_stations[st.session_state.gdf_stations[Config.PERCENTAGE_COL] >= min_data_perc]
        if selected_altitudes:
            conditions = []
            for r in selected_altitudes:
                if r == '0-500': conditions.append((filtered_stations_temp[Config.ALTITUDE_COL] >= 0) & (filtered_stations_temp[Config.ALTITUDE_COL] <= 500))
                elif r == '500-1000': conditions.append((filtered_stations_temp[Config.ALTITUDE_COL] > 500) & (filtered_stations_temp[Config.ALTITUDE_COL] <= 1000))
                elif r == '1000-2000': conditions.append((filtered_stations_temp[Config.ALTITUDE_COL] > 1000) & (filtered_stations_temp[Config.ALTITUDE_COL] <= 2000))
                elif r == '>3000': conditions.append(filtered_stations_temp[Config.ALTITUDE_COL] > 3000)
            if conditions: stations_filtered = stations_filtered[pd.concat(conditions, axis=1).any(axis=1)]
        if selected_regions: filtered_stations_temp = filtered_stations_temp[filtered_stations_temp[Config.REGION_COL].isin(selected_regions)]
        municipios_list = sorted(filtered_stations_temp[Config.MUNICIPALITY_COL].dropna().unique())
        selected_municipios = st.multiselect('Filtrar por Municipio', options=municipios_list, default=st.session_state.get('municipios_multiselect', []), key='municipios_multiselect')
        if selected_municipios: filtered_stations_temp = filtered_stations_temp[filtered_stations_temp[Config.MUNICIPALITY_COL].isin(selected_municipios)]
        celdas_list = sorted(filtered_stations_temp[Config.CELL_COL].dropna().unique())
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
        if st.checkbox("Seleccionar/Deseleccionar todas las estaciones", value=st.session_state.select_all_stations_state, key='select_all_checkbox'):
            selected_stations = stations_options
        else:
            selected_stations = st.multiselect('Seleccionar Estaciones', options=stations_options, default=st.session_state.get('station_multiselect', []), key='station_multiselect')

        if set(selected_stations) == set(stations_options) and not st.session_state.select_all_stations_state:
            st.session_state.select_all_stations_state = True
        elif set(selected_stations) != set(stations_options) and st.session_state.select_all_stations_state:
            st.session_state.select_all_stations_state = False

        if selected_stations:
            years_with_data_in_selection = sorted([int(col) for col in st.session_state.gdf_stations.loc[st.session_state.gdf_stations[Config.STATION_NAME_COL].isin(selected_stations)].columns if str(col).isdigit()])
        else:
            years_with_data_in_selection = sorted([int(col) for col in st.session_state.gdf_stations.columns if str(col).isdigit()])
        if not years_with_data_in_selection:
            st.error("No se encontraron años disponibles para la selección actual.")
            st.stop()

        year_range = st.slider("Seleccionar Rango de Años", min(years_with_data_in_selection), max(years_with_data_in_selection), (min(years_with_data_in_selection), max(years_with_data_in_selection)))
        meses_dict = {'Enero': 1, 'Febrero': 2, 'Marzo': 3, 'Abril': 4, 'Mayo': 5, 'Junio': 6, 'Julio': 7, 'Agosto': 8, 'Septiembre': 9, 'Octubre': 10, 'Noviembre': 11, 'Diciembre': 12}
        meses_nombres = st.multiselect("Seleccionar Meses", list(meses_dict.keys()), default=list(meses_dict.keys()))
        meses_numeros = [meses_dict[m] for m in meses_nombres]

    with st.sidebar.expander("Opciones de Preprocesamiento de Datos", expanded=False):
        analysis_mode = st.radio("Análisis de Series Mensuales", ("Usar datos originales", "Completar series (interpolación)"))
        
        exclude_na = st.checkbox("Excluir datos nulos (NaN)", value=st.session_state.exclude_na, key='exclude_na_checkbox')
        exclude_zeros = st.checkbox("Excluir valores cero (0)", value=st.session_state.exclude_zeros, key='exclude_zeros_checkbox')

        if analysis_mode != st.session_state.analysis_mode or exclude_na != st.session_state.exclude_na or exclude_zeros != st.session_state.exclude_zeros:
            st.session_state.analysis_mode = analysis_mode
            st.session_state.exclude_na = exclude_na
            st.session_state.exclude_zeros = exclude_zeros
            st.rerun()

    # --- Lógica de filtrado de datos principal ---
    st.session_state.gdf_filtered = apply_filters_to_stations(st.session_state.gdf_stations, min_data_perc, selected_altitudes, selected_regions, selected_municipios, selected_celdas)

    if not selected_stations:
        stations_for_analysis = st.session_state.gdf_filtered[Config.STATION_NAME_COL].unique()
    else:
        stations_for_analysis = selected_stations
        st.session_state.gdf_filtered = st.session_state.gdf_filtered[st.session_state.gdf_filtered[Config.STATION_NAME_COL].isin(stations_for_analysis)]

    if st.session_state.gdf_stations is not None and not st.session_state.gdf_stations.empty:
        df_anual_melted_temp = st.session_state.gdf_stations.melt(
            id_vars=[Config.STATION_NAME_COL, Config.MUNICIPALITY_COL, Config.LONGITUDE_COL, Config.LATITUDE_COL, Config.ALTITUDE_COL],
            value_vars=[str(y) for y in range(year_range[0], year_range[1] + 1) if str(y) in st.session_state.gdf_stations.columns],
            var_name=Config.YEAR_COL,
            value_name=Config.PRECIPITATION_COL
        )
        st.session_state.df_anual_melted = df_anual_melted_temp[df_anual_melted_temp[Config.STATION_NAME_COL].isin(stations_for_analysis)].copy()
    else:
        st.session_state.df_anual_melted = pd.DataFrame()
    
    if st.session_state.exclude_na and not st.session_state.df_anual_melted.empty:
        st.session_state.df_anual_melted.dropna(subset=[Config.PRECIPITATION_COL], inplace=True)
    if st.session_state.exclude_zeros and not st.session_state.df_anual_melted.empty:
        st.session_state.df_anual_melted = st.session_state.df_anual_melted[st.session_state.df_anual_melted[Config.PRECIPITATION_COL] > 0]
    
    if st.session_state.df_long is not None and not st.session_state.df_long.empty:
        if st.session_state.analysis_mode == "Completar series (interpolación)":
            st.session_state.df_monthly_processed = complete_series(st.session_state.df_long.copy())
        else:
            st.session_state.df_monthly_processed = st.session_state.df_long.copy()
        
        df_monthly_processed_filtered = st.session_state.df_monthly_processed.copy()
        
        if st.session_state.exclude_na:
            df_monthly_processed_filtered.dropna(subset=[Config.PRECIPITATION_COL], inplace=True)
        if st.session_state.exclude_zeros:
            df_monthly_processed_filtered = df_monthly_processed_filtered[df_monthly_processed_filtered[Config.PRECIPITATION_COL] > 0]
        
        if Config.STATION_NAME_COL in df_monthly_processed_filtered.columns and not df_monthly_processed_filtered.empty:
            st.session_state.df_monthly_filtered = df_monthly_processed_filtered[
                (df_monthly_processed_filtered[Config.STATION_NAME_COL].isin(stations_for_analysis)) &
                (df_monthly_processed_filtered[Config.DATE_COL].dt.year >= year_range[0]) &
                (df_monthly_processed_filtered[Config.DATE_COL].dt.year <= year_range[1]) &
                (df_monthly_processed_filtered[Config.DATE_COL].dt.month.isin(meses_numeros))
            ].copy()
        else:
            st.session_state.df_monthly_filtered = pd.DataFrame(columns=[Config.STATION_NAME_COL, Config.DATE_COL, Config.PRECIPITATION_COL])
    else:
        st.session_state.df_monthly_filtered = pd.DataFrame(columns=[Config.STATION_NAME_COL, Config.DATE_COL, Config.PRECIPITATION_COL])


    st.session_state.year_range = year_range
    st.session_state.meses_numeros = meses_numeros

    # Pestañas Principales
    tab_names = ["🏠 Bienvenida", "Distribución Espacial", "Gráficos", "Mapas Avanzados", "Tabla de Estaciones", "Análisis de Anomalías", "Estadísticas", "Análisis de Correlación", "Análisis ENSO", "Tendencias y Pronósticos", "Descargas"]
    (bienvenida_tab, mapa_tab, graficos_tab, mapas_avanzados_tab, tabla_estaciones_tab, anomalias_tab, estadisticas_tab, correlacion_tab, enso_tab, tendencias_tab, descargas_tab) = st.tabs(tab_names)

    with bienvenida_tab:
        display_welcome_tab()
    with mapa_tab:
        display_spatial_distribution_tab(st.session_state.gdf_filtered, st.session_state.df_anual_melted, stations_for_analysis)
    with graficos_tab:
        display_graphs_tab(st.session_state.df_anual_melted, st.session_state.df_monthly_filtered, stations_for_analysis)
    with mapas_avanzados_tab:
        display_advanced_maps_tab(st.session_state.gdf_filtered, st.session_state.df_anual_melted, stations_for_analysis)
    with tabla_estaciones_tab:
        display_station_table_tab(st.session_state.gdf_filtered, st.session_state.df_anual_melted, stations_for_analysis)
    with anomalias_tab:
        display_anomalies_tab(st.session_state.df_long, st.session_state.df_monthly_filtered, st.session_state.df_enso, stations_for_analysis)
    with estadisticas_tab:
        display_stats_tab(st.session_state.df_long, st.session_state.df_anual_melted, st.session_state.df_monthly_filtered, stations_for_analysis)
    with correlacion_tab:
        display_correlation_tab(st.session_state.df_monthly_filtered, stations_for_analysis)
    with enso_tab:
        display_enso_tab(st.session_state.df_monthly_filtered, st.session_state.df_enso, st.session_state.gdf_filtered, stations_for_analysis)
    with tendencias_tab:
        display_trends_and_forecast_tab(st.session_state.df_anual_melted, st.session_state.df_monthly_processed, stations_for_analysis)
    with descargas_tab:
        display_downloads_tab(st.session_state.df_anual_melted, st.session_state.df_monthly_filtered, stations_for_analysis)

if __name__ == "__main__":
    main()
