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
import base64
from scipy import stats
import statsmodels.api as sm
from prophet import Prophet
from prophet.plot import plot_plotly
import branca.colormap as cm

# --- Clase de Configuración ---
# Centraliza los nombres de las columnas y constantes para fácil mantenimiento.
class Config:
    # Nombres de columnas en los archivos de entrada
    class C:
        # Generales
        FECHA_MES_ANO = 'fecha_mes_año'
        ANO = 'año'
        MES = 'mes'
        NOM_ESTACION = 'nom_est'
        ID_ESTACION_STR = 'id_estacion'
        ID_ESTACION_INT = 'id_estacio'
        PRECIPITACION = 'precipitation'

        # Archivo de estaciones (mapaCVENSO.csv)
        LON_BUSCAR = ['longitud', 'lon']
        LAT_BUSCAR = ['latitud', 'lat']
        ALTITUD = 'alt_est'
        PORC_DATOS = 'porc_datos'
        DEPTO_REGION = 'depto_region'
        MUNICIPIO = 'municipio'
        CELDA_XY = 'celda_xy'
        LON_GEO = 'longitud_geo'
        LAT_GEO = 'latitud_geo'

        # Archivo de precipitación (DatosPptnmes_ENSO.csv)
        ORIGEN = 'origen'
        ANOMALIA_ONI = 'anomalia_oni'
        TEMP_SST = 'temp_sst'
        TEMP_MEDIA = 'temp_media'

        # Archivo Shapefile
        MUNICIPIO_SHP_BUSCAR = ['mcnpio', 'municipio', 'nombre_mpio', 'mpio_cnmbr']

    # Valores para lógica interna
    class V:
        ORIGEN_ORIGINAL = 'Original'
        ORIGEN_COMPLETADO = 'Completado'
        FASE_NINO = 'El Niño'
        FASE_NINA = 'La Niña'
        FASE_NEUTRAL = 'Neutral'

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
                gdf.set_crs("EPSG:9377", inplace=True) # Asumir CRS si no está definido
            return gdf.to_crs("EPSG:4326")
    except Exception as e:
        st.error(f"Error al procesar el shapefile: {e}")
        return None

@st.cache_data
def complete_series(_df):
    """Completa las series de tiempo de precipitación usando interpolación."""
    all_completed_dfs = []
    station_list = _df[Config.C.NOM_ESTACION].unique()
    progress_bar = st.progress(0, text="Completando todas las series...")
    for i, station in enumerate(station_list):
        df_station = _df[_df[Config.C.NOM_ESTACION] == station].copy()
        df_station[Config.C.FECHA_MES_ANO] = pd.to_datetime(df_station[Config.C.FECHA_MES_ANO], format='%b-%y', errors='coerce')
        df_station.set_index(Config.C.FECHA_MES_ANO, inplace=True)

        if not df_station.index.is_unique:
            df_station = df_station[~df_station.index.duplicated(keep='first')]

        original_data = df_station[[Config.C.PRECIPITACION, Config.C.ORIGEN]].copy()
        df_resampled = df_station.resample('MS').asfreq()
        df_resampled[Config.C.PRECIPITACION] = original_data[Config.C.PRECIPITACION]
        df_resampled[Config.C.ORIGEN] = original_data[Config.C.ORIGEN]
        if Config.C.ANOMALIA_ONI in df_station.columns:
            df_resampled[Config.C.ANOMALIA_ONI] = df_station[Config.C.ANOMALIA_ONI]
        df_resampled[Config.C.ORIGEN] = df_resampled[Config.C.ORIGEN].fillna(Config.V.ORIGEN_COMPLETADO)
        df_resampled[Config.C.PRECIPITACION] = df_resampled[Config.C.PRECIPITACION].interpolate(method='time')

        df_resampled[Config.C.NOM_ESTACION] = station
        df_resampled[Config.C.ANO] = df_resampled.index.year
        df_resampled[Config.C.MES] = df_resampled.index.month
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
    lon_col = next((col for col in df_precip_anual.columns if any(sub in col.lower() for sub in Config.C.LON_BUSCAR)), None)
    lat_col = next((col for col in df_precip_anual.columns if any(sub in col.lower() for sub in Config.C.LAT_BUSCAR)), None)
    if not all([lon_col, lat_col]):
        st.error("No se encontraron las columnas de longitud y/o latitud en el archivo de estaciones.")
        return None, None, None, None, None
    df_precip_anual[lon_col] = pd.to_numeric(df_precip_anual[lon_col].astype(str).str.replace(',', '.'), errors='coerce')
    df_precip_anual[lat_col] = pd.to_numeric(df_precip_anual[lat_col].astype(str).str.replace(',', '.'), errors='coerce')
    if Config.C.ALTITUD in df_precip_anual.columns:
        df_precip_anual[Config.C.ALTITUD] = pd.to_numeric(df_precip_anual[Config.C.ALTITUD].astype(str).str.replace(',', '.'), errors='coerce')
    df_precip_anual.dropna(subset=[lon_col, lat_col], inplace=True)
    gdf_temp = gpd.GeoDataFrame(df_precip_anual, geometry=gpd.points_from_xy(df_precip_anual[lon_col], df_precip_anual[lat_col]), crs="EPSG:9377")
    gdf_stations = gdf_temp.to_crs("EPSG:4326")
    gdf_stations[Config.C.LON_GEO] = gdf_stations.geometry.x
    gdf_stations[Config.C.LAT_GEO] = gdf_stations.geometry.y

    # Preprocesamiento de datos mensuales y ENSO
    df_precip_mensual = df_precip_mensual_raw.copy()
    station_cols = [col for col in df_precip_mensual.columns if col.isdigit()]
    if not station_cols:
        st.error("No se encontraron columnas de estación (ej: '12345') en el archivo de precipitación mensual.")
        return None, None, None, None, None

    id_vars_base = ['id', Config.C.FECHA_MES_ANO, Config.C.ANO, Config.C.MES, 'enso_año', 'enso_mes']
    id_vars_enso = [Config.C.ANOMALIA_ONI, Config.C.TEMP_SST, Config.C.TEMP_MEDIA]
    id_vars = id_vars_base + id_vars_enso

    for col in id_vars_enso:
        if col in df_precip_mensual.columns:
            df_precip_mensual[col] = df_precip_mensual[col].astype(str).str.replace(',', '.')

    df_long = df_precip_mensual.melt(id_vars=[col for col in id_vars if col in df_precip_mensual.columns],
                                     value_vars=station_cols, var_name=Config.C.ID_ESTACION_STR, value_name=Config.C.PRECIPITACION)

    df_long[Config.C.PRECIPITACION] = pd.to_numeric(df_long[Config.C.PRECIPITACION].astype(str).str.replace(',', '.'), errors='coerce')
    for col in id_vars_enso:
        if col in df_long.columns:
            df_long[col] = pd.to_numeric(df_long[col], errors='coerce')

    df_long.dropna(subset=[Config.C.PRECIPITACION], inplace=True)
    df_long[Config.C.FECHA_MES_ANO] = parse_spanish_dates(df_long[Config.C.FECHA_MES_ANO])
    df_long[Config.C.FECHA_MES_ANO] = pd.to_datetime(df_long[Config.C.FECHA_MES_ANO], format='%b-%y', errors='coerce')
    df_long.dropna(subset=[Config.C.FECHA_MES_ANO], inplace=True)
    df_long[Config.C.ORIGEN] = Config.V.ORIGEN_ORIGINAL

    gdf_stations[Config.C.ID_ESTACION_INT] = gdf_stations[Config.C.ID_ESTACION_INT].astype(str).str.strip()
    df_long[Config.C.ID_ESTACION_STR] = df_long[Config.C.ID_ESTACION_STR].astype(str).str.strip()
    station_mapping = gdf_stations.set_index(Config.C.ID_ESTACION_INT)[Config.C.NOM_ESTACION].to_dict()
    df_long[Config.C.NOM_ESTACION] = df_long[Config.C.ID_ESTACION_STR].map(station_mapping)
    df_long.dropna(subset=[Config.C.NOM_ESTACION], inplace=True)

    enso_cols = ['id', Config.C.FECHA_MES_ANO, Config.C.ANOMALIA_ONI, Config.C.TEMP_SST, Config.C.TEMP_MEDIA]
    existing_enso_cols = [col for col in enso_cols if col in df_precip_mensual.columns]
    df_enso = df_precip_mensual[existing_enso_cols].drop_duplicates().copy()

    for col in [c for c in [Config.C.ANOMALIA_ONI, Config.C.TEMP_SST, Config.C.TEMP_MEDIA] if c in df_enso.columns]:
        df_enso[col] = pd.to_numeric(df_enso[col], errors='coerce')

    if Config.C.FECHA_MES_ANO in df_enso.columns:
        df_enso[Config.C.FECHA_MES_ANO] = parse_spanish_dates(df_enso[Config.C.FECHA_MES_ANO])
        df_enso[Config.C.FECHA_MES_ANO] = pd.to_datetime(df_enso[Config.C.FECHA_MES_ANO], format='%b-%y', errors='coerce')
        df_enso.dropna(subset=[Config.C.FECHA_MES_ANO], inplace=True)

    return gdf_stations, df_precip_anual, gdf_municipios, df_long, df_enso

# --- Funciones de Gráficos Reutilizables ---
def create_enso_chart(enso_data):
    if enso_data.empty or Config.C.ANOMALIA_ONI not in enso_data.columns:
        return go.Figure()

    data = enso_data.copy().sort_values(Config.C.FECHA_MES_ANO)
    data.dropna(subset=[Config.C.ANOMALIA_ONI], inplace=True)

    conditions = [data[Config.C.ANOMALIA_ONI] >= 0.5, data[Config.C.ANOMALIA_ONI] <= -0.5]
    phases = [Config.V.FASE_NINO, Config.V.FASE_NINA]
    colors = ['red', 'blue']
    data['phase'] = np.select(conditions, phases, default=Config.V.FASE_NEUTRAL)
    data['color'] = np.select(conditions, colors, default='grey')

    y_range = [data[Config.C.ANOMALIA_ONI].min() - 0.5, data[Config.C.ANOMALIA_ONI].max() + 0.5]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=data[Config.C.FECHA_MES_ANO], y=[y_range[1] - y_range[0]] * len(data),
        base=y_range[0], marker_color=data['color'], width=30*24*60*60*1000,
        opacity=0.3, hoverinfo='none', showlegend=False
    ))

    legend_map = {Config.V.FASE_NINO: 'red', Config.V.FASE_NINA: 'blue', Config.V.FASE_NEUTRAL: 'grey'}
    for phase, color in legend_map.items():
        fig.add_trace(go.Scatter(
            x=[None], y=[None], mode='markers',
            marker=dict(size=15, color=color, symbol='square', opacity=0.5),
            name=phase, showlegend=True
        ))

    fig.add_trace(go.Scatter(
        x=data[Config.C.FECHA_MES_ANO], y=data[Config.C.ANOMALIA_ONI],
        mode='lines', name='Anomalía ONI', line=dict(color='black', width=2), showlegend=True
    ))

    fig.add_hline(y=0.5, line_dash="dash", line_color="red")
    fig.add_hline(y=-0.5, line_dash="dash", line_color="blue")

    fig.update_layout(
        height=600, title="Fases del Fenómeno ENSO y Anomalía ONI",
        yaxis_title="Anomalía ONI (°C)", xaxis_title="Fecha",
        showlegend=True, legend_title_text='Fase', yaxis_range=y_range
    )
    return fig

def create_anomaly_chart(df_plot):
    if df_plot.empty:
        return go.Figure()

    df_plot['color'] = np.where(df_plot['anomalia'] < 0, 'red', 'blue')
    fig = go.Figure()

    fig.add_trace(go.Bar(
        x=df_plot[Config.C.FECHA_MES_ANO], y=df_plot['anomalia'],
        marker_color=df_plot['color'], name='Anomalía de Precipitación'
    ))

    if Config.C.ANOMALIA_ONI in df_plot.columns:
        df_plot_enso = df_plot.dropna(subset=[Config.C.ANOMALIA_ONI])
        
        nino_periods = df_plot_enso[df_plot_enso[Config.C.ANOMALIA_ONI] >= 0.5]
        for _, row in nino_periods.iterrows():
            fig.add_vrect(x0=row[Config.C.FECHA_MES_ANO] - pd.DateOffset(days=15), x1=row[Config.C.FECHA_MES_ANO] + pd.DateOffset(days=15), 
                          fillcolor="red", opacity=0.15, layer="below", line_width=0)

        nina_periods = df_plot_enso[df_plot_enso[Config.C.ANOMALIA_ONI] <= -0.5]
        for _, row in nina_periods.iterrows():
            fig.add_vrect(x0=row[Config.C.FECHA_MES_ANO] - pd.DateOffset(days=15), x1=row[Config.C.FECHA_MES_ANO] + pd.DateOffset(days=15), 
                          fillcolor="blue", opacity=0.15, layer="below", line_width=0)

        fig.add_trace(go.Scatter(x=[None], y=[None], mode='markers', marker=dict(symbol='square', color='rgba(255, 0, 0, 0.3)'), name=f'Fase {Config.V.FASE_NINO}'))
        fig.add_trace(go.Scatter(x=[None], y=[None], mode='markers', marker=dict(symbol='square', color='rgba(0, 0, 255, 0.3)'), name=f'Fase {Config.V.FASE_NINA}'))

    fig.update_layout(
        height=600, title="Anomalías Mensuales de Precipitación y Fases ENSO",
        yaxis_title="Anomalía de Precipitación (mm)", xaxis_title="Fecha", showlegend=True
    )
    return fig

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

# --- Funciones de Renderizado de Pestañas ---
def render_bienvenida_tab():
    logo_path = "CuencaVerdeLogo_V1.JPG"
    st.header("Bienvenido al Sistema de Información de Lluvias y Clima")
    st.markdown("""
    Esta plataforma interactiva está diseñada para la visualización y análisis de datos históricos de precipitación y su relación con el fenómeno ENSO en el norte de la región Andina.
    
    **¿Cómo empezar?**
    1. **Cargue sus archivos**: Si es la primera vez que usa la aplicación, el panel de la izquierda le solicitará cargar los archivos de estaciones, precipitación y el shapefile de municipios. La aplicación recordará estos archivos en su sesión.
    2. **Filtre los datos**: Utilice el **Panel de Control** en la barra lateral para filtrar las estaciones por ubicación (región, municipio), altitud, porcentaje de datos disponibles, y para seleccionar el período de análisis (años y meses).
    3. **Explore las pestañas**: Cada pestaña ofrece una perspectiva diferente de los datos. Navegue a través de ellas para descubrir:
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

def render_mapa_tab(stations_for_analysis, df_anual_melted, gdf_filtered, gdf_municipios, analysis_mode, df_monthly_filtered, year_range):
    st.header(f"Distribución espacial de las Estaciones de Lluvia ({year_range[0]} - {year_range[1]})")
    logo_gota_path = "CuencaVerdeGoticaLogo.JPG"
    
    if len(stations_for_analysis) == 0:
        st.warning("Por favor, seleccione al menos una estación para ver esta sección.")
        return

    if not df_anual_melted.empty:
        df_mean_precip = df_anual_melted.groupby(Config.C.NOM_ESTACION)[Config.C.PRECIPITACION].mean().reset_index()
        gdf_filtered_map = gdf_filtered.merge(df_mean_precip.rename(columns={Config.C.PRECIPITACION: 'precip_media_anual'}), on=Config.C.NOM_ESTACION, how='left')
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
                    if os.path.exists(logo_gota_path):
                        st.image(logo_gota_path, width=50)
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

                folium.GeoJson(gdf_municipios.to_json(), name='Municipios').add_to(m)

                for layer_config in selected_overlays_config:
                    WmsTileLayer(
                        url=layer_config["url"], layers=layer_config["layers"], fmt='image/png',
                        transparent=layer_config.get("transparent", False), overlay=True,
                        control=True, name=layer_config["attr"]
                    ).add_to(m)
                
                marker_cluster = MarkerCluster(name='Estaciones').add_to(m)
                for _, row in gdf_filtered_map.iterrows():
                    html = f"""
                    <b>Estación:</b> {row[Config.C.NOM_ESTACION]}<br>
                    <b>Municipio:</b> {row[Config.C.MUNICIPIO]}<br>
                    <b>Celda:</b> {row[Config.C.CELDA_XY]}<br>
                    <b>% Datos Disponibles:</b> {row[Config.C.PORC_DATOS]:.0f}%<br>
                    <b>Ppt. Media Anual (mm):</b> {row['precip_media_anual']:.0f}
                    """
                    folium.Marker(
                        location=[row[Config.C.LAT_GEO], row[Config.C.LON_GEO]],
                        tooltip=html
                    ).add_to(marker_cluster)

                folium.LayerControl().add_to(m)
                MiniMap(toggle_display=True).add_to(m)
                
                folium_static(m, height=700, width="100%")
            else:
                st.warning("No hay estaciones seleccionadas para mostrar en el mapa.")

    with sub_tab_grafico:
        st.subheader("Disponibilidad y Composición de Datos por Estación")
        if not gdf_filtered.empty:
            if analysis_mode == "Completar series (interpolación)":
                st.info("Mostrando la composición de datos originales vs. completados para el período seleccionado.")
                
                data_composition = df_monthly_filtered.groupby([Config.C.NOM_ESTACION, Config.C.ORIGEN]).size().unstack(fill_value=0)
                if Config.V.ORIGEN_ORIGINAL not in data_composition:
                    data_composition[Config.V.ORIGEN_ORIGINAL] = 0
                if Config.V.ORIGEN_COMPLETADO not in data_composition:
                    data_composition[Config.V.ORIGEN_COMPLETADO] = 0
                
                data_composition['total'] = data_composition[Config.V.ORIGEN_ORIGINAL] + data_composition[Config.V.ORIGEN_COMPLETADO]
                data_composition['% Original'] = (data_composition[Config.V.ORIGEN_ORIGINAL] / data_composition['total']) * 100
                data_composition['% Completado'] = (data_composition[Config.V.ORIGEN_COMPLETADO] / data_composition['total']) * 100

                sort_order_comp = st.radio(
                    "Ordenar por:", ["% Datos Originales (Mayor a Menor)", "% Datos Originales (Menor a Mayor)", "Alfabético"],
                    horizontal=True, key="sort_comp"
                )
                if "Mayor a Menor" in sort_order_comp:
                    data_composition = data_composition.sort_values("% Original", ascending=False)
                elif "Menor a Mayor" in sort_order_comp:
                    data_composition = data_composition.sort_values("% Original", ascending=True)
                else:
                    data_composition = data_composition.sort_index(ascending=True)

                df_plot = data_composition.reset_index().melt(
                    id_vars=Config.C.NOM_ESTACION, value_vars=['% Original', '% Completado'],
                    var_name='Tipo de Dato', value_name='Porcentaje'
                )

                fig_comp = px.bar(df_plot, x=Config.C.NOM_ESTACION, y='Porcentaje', color='Tipo de Dato',
                                  title='Composición de Datos por Estación',
                                  labels={Config.C.NOM_ESTACION: 'Estación', 'Porcentaje': '% del Período'},
                                  color_discrete_map={'% Original': '#1f77b4', '% Completado': '#ff7f0e'},
                                  text_auto='.1f')
                fig_comp.update_layout(height=600, xaxis={'categoryorder': 'trace'})
                st.plotly_chart(fig_comp, use_container_width=True)
            else:
                st.info("Mostrando el porcentaje de disponibilidad de datos según el archivo de estaciones.")
                sort_order_disp = st.radio(
                    "Ordenar estaciones por:", ["% Datos (Mayor a Menor)", "% Datos (Menor a Mayor)", "Alfabético"],
                    horizontal=True, key="sort_disp"
                )
                df_chart = gdf_filtered.copy()
                if "% Datos (Mayor a Menor)" in sort_order_disp:
                    df_chart = df_chart.sort_values(Config.C.PORC_DATOS, ascending=False)
                elif "% Datos (Menor a Mayor)" in sort_order_disp:
                    df_chart = df_chart.sort_values(Config.C.PORC_DATOS, ascending=True)
                else:
                    df_chart = df_chart.sort_values(Config.C.NOM_ESTACION, ascending=True)
                
                fig_disp = px.bar(df_chart, x=Config.C.NOM_ESTACION, y=Config.C.PORC_DATOS, 
                                  title='Porcentaje de Disponibilidad de Datos Históricos',
                                  labels={Config.C.NOM_ESTACION: 'Estación', Config.C.PORC_DATOS: '% de Datos Disponibles'},
                                  color=Config.C.PORC_DATOS, color_continuous_scale=px.colors.sequential.Viridis)
                fig_disp.update_layout(height=600, xaxis={'categoryorder':'trace'})
                st.plotly_chart(fig_disp, use_container_width=True)
        else:
            st.warning("No hay estaciones seleccionadas para mostrar el gráfico.")

# --- INICIO DE LA APLICACIÓN ---
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
        st.info("Por favor, suba los 3 archivos requeridos para habilitar la aplicación. Una vez cargados, la plataforma se iniciará automáticamente.")
        st.stop()
    else:
        with st.spinner("Procesando archivos y cargando datos... Esto puede tomar un momento."):
            gdf_stations, df_precip_anual, gdf_municipios, df_long, df_enso = preprocess_data(uploaded_file_mapa, uploaded_file_precip, uploaded_zip_shapefile)
            if gdf_stations is not None:
                st.session_state.gdf_stations = gdf_stations
                st.session_state.df_precip_anual = df_precip_anual
                st.session_state.gdf_municipios = gdf_municipios
                st.session_state.df_long = df_long
                st.session_state.df_enso = df_enso
                st.session_state.data_loaded = True
                st.rerun()
            else:
                st.error("Hubo un error al procesar los archivos. Por favor, verifique que los archivos sean correctos y vuelva a intentarlo.")
                st.stop()

# Si los datos no se cargaron, detener la ejecución de la app
if 'gdf_stations' not in st.session_state or st.session_state.gdf_stations is None:
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
    if Config.C.PORC_DATOS in stations_filtered.columns:
        stations_filtered[Config.C.PORC_DATOS] = pd.to_numeric(stations_filtered[Config.C.PORC_DATOS].astype(str).str.replace(',', '.'), errors='coerce').fillna(0)
        stations_filtered = stations_filtered[stations_filtered[Config.C.PORC_DATOS] >= min_data_perc]
    
    if selected_altitudes:
        conditions = []
        for r in selected_altitudes:
            if r == '0-500': conditions.append((stations_filtered[Config.C.ALTITUD] >= 0) & (stations_filtered[Config.C.ALTITUD] <= 500))
            elif r == '500-1000': conditions.append((stations_filtered[Config.C.ALTITUD] > 500) & (stations_filtered[Config.C.ALTITUD] <= 1000))
            elif r == '1000-2000': conditions.append((stations_filtered[Config.C.ALTITUD] > 1000) & (stations_filtered[Config.C.ALTITUD] <= 2000))
            elif r == '2000-3000': conditions.append((stations_filtered[Config.C.ALTITUD] > 2000) & (stations_filtered[Config.C.ALTITUD] <= 3000))
            elif r == '>3000': conditions.append(stations_filtered[Config.C.ALTITUD] > 3000)
        if conditions:
            combined_condition = pd.concat(conditions, axis=1).any(axis=1)
            stations_filtered = stations_filtered[combined_condition]

    if selected_regions:
        stations_filtered = stations_filtered[stations_filtered[Config.C.DEPTO_REGION].isin(selected_regions)]
    if selected_municipios:
        stations_filtered = stations_filtered[stations_filtered[Config.C.MUNICIPIO].isin(selected_municipios)]
    if selected_celdas:
        stations_filtered = stations_filtered[stations_filtered[Config.C.CELDA_XY].isin(selected_celdas)]
    
    return stations_filtered

# --- Controles en la Barra Lateral ---
with st.sidebar.expander("**1. Filtros Geográficos y de Datos**", expanded=True):
    min_data_perc = st.slider("Filtrar por % de datos mínimo:", 0, 100, 0, key='min_data_perc_slider')
    altitude_ranges = ['0-500', '500-1000', '1000-2000', '2000-3000', '>3000']
    selected_altitudes = st.multiselect('Filtrar por Altitud (m)', options=altitude_ranges, key='altitude_multiselect')
    regions_list = sorted(gdf_stations[Config.C.DEPTO_REGION].dropna().unique())
    selected_regions = st.multiselect('Filtrar por Depto/Región', options=regions_list, key='regions_multiselect')
    
    filtered_stations_temp = apply_filters_to_stations(gdf_stations, min_data_perc, selected_altitudes, selected_regions, [], [])
    municipios_list = sorted(filtered_stations_temp[Config.C.MUNICIPIO].dropna().unique())
    selected_municipios = st.multiselect('Filtrar por Municipio', options=municipios_list, key='municipios_multiselect')

    filtered_stations_temp_2 = apply_filters_to_stations(gdf_stations, min_data_perc, selected_altitudes, selected_regions, selected_municipios, [])
    celdas_list = sorted(filtered_stations_temp_2[Config.C.CELDA_XY].dropna().unique())
    selected_celdas = st.multiselect('Filtrar por Celda_XY', options=celdas_list, key='celdas_multiselect')
    
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
    stations_master_list = apply_filters_to_stations(gdf_stations, min_data_perc, selected_altitudes, selected_regions, selected_municipios, selected_celdas)
    stations_options = sorted(stations_master_list[Config.C.NOM_ESTACION].unique())

    if 'select_all_stations_state' not in st.session_state:
        st.session_state.select_all_stations_state = False

    if st.checkbox("Seleccionar/Deseleccionar todas las estaciones", value=st.session_state.select_all_stations_state, key='select_all_checkbox'):
        selected_stations = stations_options
    else:
        selected_stations = st.multiselect(
            'Seleccionar Estaciones', options=stations_options,
            default=stations_options if st.session_state.select_all_stations_state else [],
            key='station_multiselect'
        )
    
    if set(selected_stations) == set(stations_options) and not st.session_state.select_all_stations_state:
        st.session_state.select_all_stations_state = True
    elif set(selected_stations) != set(stations_options) and st.session_state.select_all_stations_state:
        st.session_state.select_all_stations_state = False

    years_with_data_in_selection = sorted([int(col) for col in gdf_stations.columns if str(col).isdigit()])
    if not years_with_data_in_selection:
        st.error("No se encontraron años disponibles en los datos.")
        st.stop()
    
    year_range = st.slider(
        "Seleccionar Rango de Años", 
        min(years_with_data_in_selection), max(years_with_data_in_selection),
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
    gdf_filtered = apply_filters_to_stations(gdf_stations, min_data_perc, selected_altitudes, selected_regions, selected_municipios, selected_celdas)

    if not selected_stations:
        stations_for_analysis = gdf_filtered[Config.C.NOM_ESTACION].unique()
    else:
        stations_for_analysis = selected_stations
        gdf_filtered = gdf_filtered[gdf_filtered[Config.C.NOM_ESTACION].isin(stations_for_analysis)]

    df_anual_melted = gdf_stations.melt(
        id_vars=[Config.C.NOM_ESTACION, Config.C.MUNICIPIO, Config.C.LON_GEO, Config.C.LAT_GEO, Config.C.ALTITUD],
        value_vars=[str(y) for y in range(year_range[0], year_range[1] + 1) if str(y) in gdf_stations.columns],
        var_name=Config.C.ANO, value_name=Config.C.PRECIPITACION)
    df_anual_melted = df_anual_melted[df_anual_melted[Config.C.NOM_ESTACION].isin(stations_for_analysis)]
    df_anual_melted.dropna(subset=[Config.C.PRECIPITACION], inplace=True)

    df_monthly_filtered = df_monthly_to_process[
        (df_monthly_to_process[Config.C.NOM_ESTACION].isin(stations_for_analysis)) &
        (df_monthly_to_process[Config.C.FECHA_MES_ANO].dt.year >= year_range[0]) &
        (df_monthly_to_process[Config.C.FECHA_MES_ANO].dt.year <= year_range[1]) &
        (df_monthly_to_process[Config.C.FECHA_MES_ANO].dt.month.isin(meses_numeros))
    ].copy()

# --- Pestañas Principales ---
tab_names = ["🏠 Bienvenida", "Distribución Espacial", "Gráficos", "Mapas Avanzados", "Tabla de Estaciones", "Análisis de Anomalías", "Estadísticas", "Análisis de Correlación", "Análisis ENSO", "Tendencias y Pronósticos", "Descargas"]
(bienvenida_tab, mapa_tab, graficos_tab, mapas_avanzados_tab, tabla_estaciones_tab, anomalias_tab, estadisticas_tab, correlacion_tab, enso_tab, tendencias_tab, descargas_tab) = st.tabs(tab_names)

with bienvenida_tab:
    render_bienvenida_tab()

with mapa_tab:
    render_mapa_tab(stations_for_analysis, df_anual_melted, gdf_filtered, gdf_municipios, analysis_mode, df_monthly_filtered, year_range)

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
                        x=alt.X(f'{Config.C.ANO}:O', title='Año'),
                        y=alt.Y(f'{Config.C.PRECIPITACION}:Q', title='Precipitación (mm)'),
                        color=f'{Config.C.NOM_ESTACION}:N',
                        tooltip=[alt.Tooltip(Config.C.NOM_ESTACION), alt.Tooltip(Config.C.ANO), alt.Tooltip(f'{Config.C.PRECIPITACION}:Q', format='.0f')]
                    ).properties(height=600).interactive()
                    st.altair_chart(chart_anual, use_container_width=True)

            with anual_analisis_tab:
                if not df_anual_melted.empty:
                    st.subheader("Precipitación Media Multianual")
                    st.caption(f"Período de análisis: {year_range[0]} - {year_range[1]}")
                    chart_type_annual = st.radio("Seleccionar tipo de gráfico:", ("Gráfico de Barras (Promedio)", "Gráfico de Cajas (Distribución)"), key="avg_chart_type_annual", horizontal=True)
                    if chart_type_annual == "Gráfico de Barras (Promedio)":
                        df_summary = df_anual_melted.groupby(Config.C.NOM_ESTACION, as_index=False)[Config.C.PRECIPITACION].mean().round(0)
                        sort_order = st.radio("Ordenar estaciones por:", ["Promedio (Mayor a Menor)", "Promedio (Menor a Mayor)", "Alfabético"], horizontal=True, key="sort_annual_avg")
                        if "Mayor a Menor" in sort_order: df_summary = df_summary.sort_values(Config.C.PRECIPITACION, ascending=False)
                        elif "Menor a Mayor" in sort_order: df_summary = df_summary.sort_values(Config.C.PRECIPITACION, ascending=True)
                        else: df_summary = df_summary.sort_values(Config.C.NOM_ESTACION, ascending=True)

                        fig_avg = px.bar(df_summary, x=Config.C.NOM_ESTACION, y=Config.C.PRECIPITACION, title='Promedio de Precipitación Anual', labels={Config.C.NOM_ESTACION: 'Estación', Config.C.PRECIPITACION: 'Precipitación Media Anual (mm)'}, color=Config.C.PRECIPITACION, color_continuous_scale=px.colors.sequential.Blues_r)
                        fig_avg.update_layout(height=600, xaxis={'categoryorder':'total descending' if "Mayor a Menor" in sort_order else ('total ascending' if "Menor a Mayor" in sort_order else 'trace')})
                        st.plotly_chart(fig_avg, use_container_width=True)
                    else:
                        df_anual_filtered_for_box = df_anual_melted[df_anual_melted[Config.C.NOM_ESTACION].isin(stations_for_analysis)]
                        fig_box_annual = px.box(df_anual_filtered_for_box, x=Config.C.NOM_ESTACION, y=Config.C.PRECIPITACION, color=Config.C.NOM_ESTACION, points='all', title='Distribución de la Precipitación Anual por Estación', labels={Config.C.NOM_ESTACION: 'Estación', Config.C.PRECIPITACION: 'Precipitación Anual (mm)'})
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
                        base_chart = alt.Chart(df_monthly_filtered).encode(x=alt.X(f'{Config.C.FECHA_MES_ANO}:T', title='Fecha'), y=alt.Y(f'{Config.C.PRECIPITACION}:Q', title='Precipitación (mm)'), tooltip=[alt.Tooltip(Config.C.FECHA_MES_ANO, format='%Y-%m'), alt.Tooltip(Config.C.PRECIPITACION, format='.0f'), Config.C.NOM_ESTACION, Config.C.ORIGEN, alt.Tooltip(f'{Config.C.MES}:N', title="Mes")])
                        if color_by == "Estación":
                            color_encoding = alt.Color(f'{Config.C.NOM_ESTACION}:N', legend=alt.Legend(title="Estaciones"))
                        else:
                            color_encoding = alt.Color(f'month({Config.C.FECHA_MES_ANO}):N', legend=alt.Legend(title="Meses"), scale=alt.Scale(scheme='tableau20'))
                        
                        if chart_type == "Líneas y Puntos":
                            line_chart = base_chart.mark_line(opacity=0.4, color='lightgray').encode(detail=f'{Config.C.NOM_ESTACION}:N')
                            point_chart = base_chart.mark_point(filled=True, size=60).encode(color=color_encoding)
                            final_chart = (line_chart + point_chart)
                        else:
                            point_chart = base_chart.mark_point(filled=True, size=60).encode(color=color_encoding)
                            final_chart = point_chart
                        st.altair_chart(final_chart.properties(height=600).interactive(), use_container_width=True)
                    else:
                        st.subheader("Distribución de la Precipitación Mensual")
                        fig_box_monthly = px.box(df_monthly_filtered, x=Config.C.MES, y=Config.C.PRECIPITACION, color=Config.C.NOM_ESTACION, title='Distribución de la Precipitación por Mes', labels={Config.C.MES: 'Mes', Config.C.PRECIPITACION: 'Precipitación Mensual (mm)', Config.C.NOM_ESTACION: 'Estación'})
                        fig_box_monthly.update_layout(height=600)
                        st.plotly_chart(fig_box_monthly, use_container_width=True)
            
            with mensual_enso_tab:
                enso_filtered = df_enso[(df_enso[Config.C.FECHA_MES_ANO].dt.year >= year_range[0]) & (df_enso[Config.C.FECHA_MES_ANO].dt.year <= year_range[1]) & (df_enso[Config.C.FECHA_MES_ANO].dt.month.isin(meses_numeros))]
                fig_enso_mensual = create_enso_chart(enso_filtered)
                st.plotly_chart(fig_enso_mensual, use_container_width=True, key="enso_chart_mensual")

            with mensual_datos_tab:
                st.subheader("Datos de Precipitación Mensual Detallados")
                if not df_monthly_filtered.empty:
                    df_values = df_monthly_filtered.pivot_table(index=Config.C.FECHA_MES_ANO, columns=Config.C.NOM_ESTACION, values=Config.C.PRECIPITACION).round(0)
                    st.dataframe(df_values)
            
        with sub_tab_comparacion:
            st.subheader("Comparación de Precipitación entre Estaciones")
            if len(stations_for_analysis) < 2:
                st.info("Seleccione al menos dos estaciones para comparar.")
            else:
                st.markdown("##### Precipitación Mensual Promedio")
                df_monthly_avg = df_monthly_filtered.groupby([Config.C.NOM_ESTACION, Config.C.MES])[Config.C.PRECIPITACION].mean().reset_index()
                fig_avg_monthly = px.line(df_monthly_avg, x=Config.C.MES, y=Config.C.PRECIPITACION, color=Config.C.NOM_ESTACION, labels={Config.C.MES: 'Mes', Config.C.PRECIPITACION: 'Precipitación Promedio (mm)'}, title='Promedio de Precipitación Mensual por Estación')
                fig_avg_monthly.update_layout(height=600, xaxis = dict(tickmode = 'array', tickvals = list(meses_dict.values()), ticktext = list(meses_dict.keys())))
                st.plotly_chart(fig_avg_monthly, use_container_width=True)

                st.markdown("##### Distribución de Precipitación Anual")
                df_anual_filtered_for_box = df_anual_melted[df_anual_melted[Config.C.NOM_ESTACION].isin(stations_for_analysis)]
                fig_box_annual = px.box(df_anual_filtered_for_box, x=Config.C.NOM_ESTACION, y=Config.C.PRECIPITACION, color=Config.C.NOM_ESTACION, points='all', title='Distribución de la Precipitación Anual por Estación', labels={Config.C.NOM_ESTACION: 'Estación', Config.C.PRECIPITACION: 'Precipitación Anual (mm)'})
                fig_box_annual.update_layout(height=600)
                st.plotly_chart(fig_box_annual, use_container_width=True)

        with sub_tab_distribucion:
            st.subheader("Distribución de la Precipitación")
            distribucion_tipo = st.radio("Seleccionar tipo de distribución:", ("Anual", "Mensual"), horizontal=True)
            if distribucion_tipo == "Anual":
                if not df_anual_melted.empty:
                    fig_hist_anual = px.histogram(df_anual_melted, x=Config.C.PRECIPITACION, color=Config.C.NOM_ESTACION, title='Distribución Anual de Precipitación', labels={Config.C.PRECIPITACION: 'Precipitación Anual (mm)', 'count': 'Frecuencia'})
                    fig_hist_anual.update_layout(height=600)
                    st.plotly_chart(fig_hist_anual, use_container_width=True)
                else:
                    st.info("No hay datos anuales para mostrar la distribución.")
            else:
                if not df_monthly_filtered.empty:
                    fig_hist_mensual = px.histogram(df_monthly_filtered, x=Config.C.PRECIPITACION, color=Config.C.NOM_ESTACION, title='Distribución Mensual de Precipitación', labels={Config.C.PRECIPITACION: 'Precipitación Mensual (mm)', 'count': 'Frecuencia'})
                    fig_hist_mensual.update_layout(height=600)
                    st.plotly_chart(fig_hist_mensual, use_container_width=True)
                else:
                    st.info("No hay datos mensuales para mostrar la distribución.")

        with sub_tab_acumulada:
            st.subheader("Precipitación Acumulada Anual")
            if not df_anual_melted.empty:
                df_acumulada = df_anual_melted.groupby([Config.C.ANO, Config.C.NOM_ESTACION])[Config.C.PRECIPITACION].sum().reset_index()
                fig_acumulada = px.bar(df_acumulada, x=Config.C.ANO, y=Config.C.PRECIPITACION, color=Config.C.NOM_ESTACION, title='Precipitación Acumulada por Año', labels={Config.C.ANO: 'Año', Config.C.PRECIPITACION: 'Precipitación Acumulada (mm)'})
                fig_acumulada.update_layout(barmode='group', height=600)
                st.plotly_chart(fig_acumulada, use_container_width=True)
            else:
                st.info("No hay datos para calcular la precipitación acumulada.")

        with sub_tab_altitud:
            st.subheader("Relación entre Altitud y Precipitación")
            if not df_anual_melted.empty and not gdf_filtered[Config.C.ALTITUD].isnull().all():
                df_relacion = df_anual_melted.groupby(Config.C.NOM_ESTACION)[Config.C.PRECIPITACION].mean().reset_index()
                df_relacion = df_relacion.merge(gdf_filtered[[Config.C.NOM_ESTACION, Config.C.ALTITUD]].drop_duplicates(), on=Config.C.NOM_ESTACION)
                fig_relacion = px.scatter(df_relacion, x=Config.C.ALTITUD, y=Config.C.PRECIPITACION, color=Config.C.NOM_ESTACION, title='Relación entre Precipitación Media Anual y Altitud', labels={Config.C.ALTITUD: 'Altitud (m)', Config.C.PRECIPITACION: 'Precipitación Media Anual (mm)'})
                fig_relacion.update_layout(height=600)
                st.plotly_chart(fig_relacion, use_container_width=True)
            else:
                st.info("No hay datos de altitud o precipitación disponibles para analizar la relación.")

with mapas_avanzados_tab:
    # Esta pestaña es muy compleja, por lo que su lógica se mantiene aquí, pero podría ser refactorizada en el futuro.
    st.header("Mapas Avanzados")
    gif_tab, temporal_tab, compare_tab, kriging_tab, coropletico_tab = st.tabs(["Animación GIF (Antioquia)", "Visualización Temporal", "Comparación de Mapas", "Interpolación Kriging", "Mapa Coroplético"])

    with gif_tab:
        st.subheader("Distribución Espacio-Temporal de la Lluvia en Antioquia")
        gif_path = "PPAM.gif"
        if os.path.exists(gif_path):
            with open(gif_path, "rb") as file:
                contents = file.read()
                data_url = base64.b64encode(contents).decode("utf-8")
            st.markdown(f'<img src="data:image/gif;base64,{data_url}" alt="Animación PPAM" style="width:100%;">', unsafe_allow_html=True)
            st.caption("Precipitación Promedio Anual Multianual en Antioquia")
        else:
            st.warning("No se encontró el archivo GIF 'PPAM.gif'.")

    # (El resto de las subpestañas de Mapas Avanzados, Anomalías, etc., seguirían aquí)
    # ... por brevedad, se omite el código que no fue modificado, pero en tu archivo final debes incluirlo todo.
    # A continuación, se muestra cómo continuar con el resto de las pestañas

with tabla_estaciones_tab:
    st.header("Información Detallada de las Estaciones")
    if len(stations_for_analysis) == 0:
        st.warning("Por favor, seleccione al menos una estación para ver esta sección.")
    elif not df_anual_melted.empty:
        df_info_table = gdf_filtered[[Config.C.NOM_ESTACION, Config.C.ALTITUD, Config.C.MUNICIPIO, Config.C.DEPTO_REGION, Config.C.PORC_DATOS]].copy()
        df_mean_precip = df_anual_melted.groupby(Config.C.NOM_ESTACION)[Config.C.PRECIPITACION].mean().round(0).reset_index()
        df_mean_precip.rename(columns={Config.C.PRECIPITACION: 'Precipitación media anual (mm)'}, inplace=True)
        df_info_table = df_info_table.merge(df_mean_precip, on=Config.C.NOM_ESTACION, how='left')
        st.dataframe(df_info_table)
    else:
        st.info("No hay datos de precipitación anual para mostrar en la selección actual.")

with anomalias_tab:
    st.header("Análisis de Anomalías de Precipitación")
    if len(stations_for_analysis) == 0:
        st.warning("Por favor, seleccione al menos una estación para ver esta sección.")
    else:
        # Calcular climatología solo con datos originales para una base consistente
        df_long_original = df_long[df_long[Config.C.NOM_ESTACION].isin(stations_for_analysis)]
        df_climatology = df_long_original.groupby([Config.C.NOM_ESTACION, Config.C.MES])[Config.C.PRECIPITACION].mean().reset_index().rename(columns={Config.C.PRECIPITACION: 'precip_promedio_mes'})
        
        df_anomalias = pd.merge(df_monthly_filtered, df_climatology, on=[Config.C.NOM_ESTACION, Config.C.MES], how='left')
        df_anomalias['anomalia'] = df_anomalias[Config.C.PRECIPITACION] - df_anomalias['precip_promedio_mes']

        if df_anomalias.empty or df_anomalias['anomalia'].isnull().all():
            st.warning("No hay suficientes datos históricos para las estaciones y el período seleccionado para calcular y mostrar las anomalías.")
        else:
            anom_graf_tab, anom_mapa_tab, anom_fase_tab, anom_extremos_tab = st.tabs(["Gráfico de Anomalías", "Mapa de Anomalías Anuales", "Anomalías por Fase ENSO", "Tabla de Eventos Extremos"])

            with anom_graf_tab:
                avg_monthly_anom = df_anomalias.groupby([Config.C.FECHA_MES_ANO, Config.C.MES])['anomalia'].mean().reset_index()
                df_plot = pd.merge(avg_monthly_anom, df_enso[[Config.C.FECHA_MES_ANO, Config.C.ANOMALIA_ONI]], on=Config.C.FECHA_MES_ANO, how='left')
                fig = create_anomaly_chart(df_plot)
                st.plotly_chart(fig, use_container_width=True)
            
            with anom_mapa_tab:
                st.subheader("Mapa Interactivo de Anomalías Anuales")
                df_anomalias_anual = df_anomalias.groupby(['nom_est', 'año'])['anomalia'].sum().reset_index()
                df_anomalias_anual = pd.merge(df_anomalias_anual, gdf_stations.loc[gdf_stations['nom_est'].isin(stations_for_analysis)][['nom_est', 'latitud_geo', 'longitud_geo']], on='nom_est')

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
                if 'anomalia_oni' in df_anomalias.columns:
                    df_anomalias_enso = df_anomalias.dropna(subset=['anomalia_oni']).copy()
                    conditions = [df_anomalias_enso['anomalia_oni'] >= 0.5, df_anomalias_enso['anomalia_oni'] <= -0.5]
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
    if len(stations_for_analysis) == 0:
        st.warning("Por favor, seleccione al menos una estación para ver esta sección.")
    else:
        matriz_tab, resumen_mensual_tab, sintesis_tab = st.tabs(["Matriz de Disponibilidad", "Resumen Mensual", "Síntesis General"])

        with matriz_tab:
            st.subheader("Matriz de Disponibilidad de Datos Anual")
            original_data_counts = df_long[df_long['nom_est'].isin(stations_for_analysis)]
            original_data_counts = original_data_counts.groupby(['nom_est', 'año']).size().reset_index(name='count')
            original_data_counts['porc_original'] = (original_data_counts['count'] / 12) * 100
            heatmap_original_df = original_data_counts.pivot(index='nom_est', columns='año', values='porc_original')

            heatmap_df = heatmap_original_df
            color_scale = "Greens"
            title_text = "Disponibilidad Promedio de Datos Originales"

            if analysis_mode == "Completar series (interpolación)":
                view_mode = st.radio("Seleccione la vista de la matriz:", ("Porcentaje de Datos Originales", "Porcentaje de Datos Completados"), horizontal=True)

                if view_mode == "Porcentaje de Datos Completados":
                    completed_data = df_monthly_to_process[(df_monthly_to_process['nom_est'].isin(stations_for_analysis)) & (df_monthly_to_process['origen'] == 'Completado')]
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

with correlacion_tab:
    st.header("Correlación entre Precipitación y ENSO")
    st.markdown("Esta sección cuantifica la relación lineal entre la precipitación mensual y la anomalía ONI utilizando el coeficiente de correlación de Pearson.")

    if 'anomalia_oni' not in df_monthly_filtered.columns:
        st.warning("No se puede realizar el análisis de correlación porque la columna 'anomalia_oni' no fue encontrada en el archivo de datos cargado o para la selección actual.")
    elif len(stations_for_analysis) == 0:
        st.warning("Por favor, seleccione al menos una estación para realizar el análisis de correlación.")
    else:
        df_corr_analysis = df_monthly_filtered[['fecha_mes_año', 'nom_est', 'precipitation', 'anomalia_oni']].dropna(subset=['precipitation', 'anomalia_oni'])
        
        if df_corr_analysis.empty:
            st.warning("No hay datos coincidentes entre la precipitación y el ENSO para la selección actual.")
        else:
            analysis_level = st.radio("Nivel de Análisis de Correlación", ["Promedio de la selección", "Por Estación Individual"], key="corr_level")

            df_plot_corr = pd.DataFrame()
            title_text = ""

            if analysis_level == "Por Estación Individual":
                station_to_corr = st.selectbox("Seleccione Estación", options=sorted(df_corr_analysis['nom_est'].unique()), key="corr_station")
                if station_to_corr:
                    df_plot_corr = df_corr_analysis[df_corr_analysis['nom_est'] == station_to_corr]
                    title_text = f"Correlación para la estación: {station_to_corr}"
            else:
                df_plot_corr = df_corr_analysis.groupby('fecha_mes_año').agg(
                    precipitation=('precipitation', 'mean'),
                    anomalia_oni=('anomalia_oni', 'first')
                ).reset_index()
                title_text = "Correlación para el promedio de las estaciones seleccionadas"

            if len(df_plot_corr) > 2:
                if 'anomalia_oni' in df_plot_corr.columns and 'precipitation' in df_plot_corr.columns:
                    corr, p_value = stats.pearsonr(df_plot_corr['anomalia_oni'], df_plot_corr['precipitation'])

                    st.subheader(title_text)

                    col1, col2 = st.columns(2)
                    col1.metric("Coeficiente de Correlación (r)", f"{corr:.3f}")
                    col2.metric("Significancia (valor p)", f"{p_value:.4f}")

                    if p_value < 0.05:
                        st.success("La correlación es estadísticamente significativa, lo que sugiere una relación lineal entre las variables.")
                    else:
                        st.warning("La correlación no es estadísticamente significativa. No hay evidencia de una relación lineal fuerte.")

                    fig_corr = px.scatter(
                        df_plot_corr,
                        x='anomalia_oni',
                        y='precipitation',
                        trendline="ols",
                        title="Gráfico de Dispersión: Precipitación vs. Anomalía ONI",
                        labels={'anomalia_oni': 'Anomalía ONI (°C)', 'precipitation': 'Precipitación Mensual (mm)'}
                    )
                    st.plotly_chart(fig_corr, use_container_width=True)
                else:
                    st.warning("Las columnas necesarias para el gráfico no están disponibles.")
            else:
                st.warning("No hay suficientes datos superpuestos para calcular la correlación para la selección actual.")
    
with enso_tab:
    st.header("Análisis de Precipitación y el Fenómeno ENSO")
    enso_series_tab, enso_anim_tab = st.tabs(["Series de Tiempo ENSO", "Mapa Interactivo ENSO"])

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
        st.subheader("Explorador Mensual del Fenómeno ENSO")
        if df_enso.empty or gdf_stations.empty:
            st.warning("No hay datos disponibles para generar esta visualización.")
        elif 'anomalia_oni' not in df_enso.columns:
            st.warning("La columna 'anomalia_oni' es necesaria para esta visualización y no se encontró.")
        else:
            controls_col, map_col = st.columns([1, 3])
            enso_anim_data = df_enso[['fecha_mes_año', 'anomalia_oni']].copy()
            enso_anim_data.dropna(subset=['anomalia_oni'], inplace=True)
            conditions = [enso_anim_data['anomalia_oni'] >= 0.5, enso_anim_data['anomalia_oni'] <= -0.5]
            phases = ['El Niño', 'La Niña']
            enso_anim_data['fase'] = np.select(conditions, phases, default='Neutral')
            enso_anim_data_filtered = enso_anim_data[(enso_anim_data['fecha_mes_año'].dt.year >= year_range[0]) & (enso_anim_data['fecha_mes_año'].dt.year <= year_range[1])]
            
            with controls_col:
                st.markdown("##### Controles de Mapa")
                selected_base_map_config, selected_overlays_config = display_map_controls(st, "enso_anim")

                st.markdown("##### Selección de Fecha")
                available_dates = sorted(enso_anim_data_filtered['fecha_mes_año'].unique())
                if available_dates:
                    selected_date = st.select_slider("Seleccione una fecha (Año-Mes)", options=available_dates, format_func=lambda date: date.strftime('%Y-%m'))
                    
                    phase_info = enso_anim_data_filtered[enso_anim_data_filtered['fecha_mes_año'] == selected_date]
                    if not phase_info.empty:
                        current_phase = phase_info['fase'].iloc[0]
                        current_oni = phase_info['anomalia_oni'].iloc[0]
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
                            location=[station['latitud_geo'], station['longitud_geo']],
                            tooltip=f"{station['nom_est']}<br>Fase: {current_phase}",
                            icon=folium.Icon(color=marker_color, icon='cloud')
                        ).add_to(m_enso)
                    
                    bounds = gdf_filtered.total_bounds
                    m_enso.fit_bounds([[bounds[1], bounds[0]], [bounds[3], bounds[2]]])

                    for layer_config in selected_overlays_config:
                        folium.raster_layers.WmsTileLayer(url=layer_config["url"], layers=layer_config["layers"], fmt='image/png', transparent=layer_config.get("transparent", False), overlay=True, control=True, name=layer_config["attr"]).add_to(m_enso)
                    
                    folium.LayerControl().add_to(m_enso)
                    st_folium(m_enso, height=700, width="100%")

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
