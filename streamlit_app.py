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
 
# Manejo de la importación de ScaleControl
try:
    from folium.plugins import ScaleControl
except ImportError:
    class ScaleControl:
        def __init__(self, *args, **kwargs):
            pass
        def add_to(self, m):
            pass
 
# --- Configuración de la página y Estilos ---
st.set_page_config(layout="wide", page_title="Visor de Precipitación y ENSO")
st.markdown("""
<style>
div.block-container {padding-top: 2rem;}
.sidebar .sidebar-content {font-size: 13px; }
h1 { margin-top: 0px; padding-top: 0px; }
[data-testid="stMetricValue"] { font-size: 1.8rem; }
[data-testid="stMetricLabel"] { font-size: 1rem; padding-bottom: 5px; }
</style>
""", unsafe_allow_html=True)
 
# --- Funciones de Carga y Preprocesamiento ---
@st.cache_data
def load_data(file_path, sep=';', date_cols=None, lower_case=True):
    """
    Carga un archivo CSV, maneja múltiples codificaciones y errores.
    """
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
            df.columns = df.columns.str.strip()
            if lower_case:
                df.columns = df.columns.str.lower()
            return df
        except Exception:
            continue
    st.error("No se pudo decodificar el archivo.")
    return None
 
@st.cache_data
def load_shapefile(file_path):
    """Carga un shapefile desde un archivo .zip."""
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
                gdf.set_crs("EPSG:4326", inplace=True)
            return gdf.to_crs("EPSG:4326")
    except Exception as e:
        st.error(f"Error al procesar el shapefile: {e}")
        return None
 
@st.cache_data
def complete_series(_df):
    """
    Completa las series de tiempo de precipitación para cada estación
    usando interpolación lineal temporal.
    """
    all_completed_dfs = []
    station_list = _df['nom_est'].unique()
    progress_bar = st.progress(0, text="Completando todas las series...")
 
    for i, station in enumerate(station_list):
        df_station = _df[_df['nom_est'] == station].copy()
        # Convertir y manejar duplicados antes de set_index
        df_station['fecha_mes_año'] = pd.to_datetime(df_station['fecha_mes_año'], format='%b-%y', errors='coerce')
        df_station.dropna(subset=['fecha_mes_año'], inplace=True)
        df_station = df_station[~df_station.index.duplicated(keep='first')]
 
        # Crear un rango de fechas completo
        if df_station.empty:
            continue
        
        start_date = df_station['fecha_mes_año'].min()
        end_date = df_station['fecha_mes_año'].max()
        date_range = pd.date_range(start=start_date, end=end_date, freq='MS')
 
        df_full = pd.DataFrame(date_range, columns=['fecha_mes_año'])
        df_full = pd.merge(df_full, df_station, on='fecha_mes_año', how='left')
 
        # Marcar y rellenar los datos
        df_full['origen'] = np.where(df_full['precipitation'].isnull(), 'Completado', 'Original')
        df_full['precipitation'] = df_full['precipitation'].interpolate(method='time')
        df_full['nom_est'] = station
        df_full['año'] = df_full['fecha_mes_año'].dt.year
        df_full['mes'] = df_full['fecha_mes_año'].dt.month
        all_completed_dfs.append(df_full)
        progress_bar.progress((i + 1) / len(station_list), text=f"Completando series... Estación: {station}")
 
    progress_bar.empty()
    if not all_completed_dfs:
        return pd.DataFrame()
    return pd.concat(all_completed_dfs, ignore_index=True)
 
def create_enso_chart(enso_data):
    """Crea un gráfico de línea y barras para visualizar la anomalía ONI y las fases ENSO."""
    if enso_data.empty or 'anomalia_oni' not in enso_data.columns:
        return go.Figure().add_annotation(
            text="No hay datos de ENSO disponibles para el periodo seleccionado.",
            xref="paper", yref="paper", showarrow=False, font=dict(size=16)
        )
 
    data = enso_data.copy().sort_values('fecha_mes_año')
    conditions = [data['anomalia_oni'] >= 0.5, data['anomalia_oni'] <= -0.5]
    phases = ['El Niño', 'La Niña']
    colors = ['red', 'blue']
    data['phase'] = np.select(conditions, phases, default='Neutral')
    data['color'] = np.select(conditions, colors, default='grey')
 
    y_range = [data['anomalia_oni'].min() - 0.5, data['anomalia_oni'].max() + 0.5]
    fig = go.Figure()
 
    # Barras de fase ENSO
    for phase in ['El Niño', 'La Niña', 'Neutral']:
        subset = data[data['phase'] == phase]
        if not subset.empty:
            fig.add_trace(go.Bar(
                x=subset['fecha_mes_año'],
                y=[y_range[1] - y_range[0]] * len(subset),
                base=y_range[0],
                marker_color=subset['color'].iloc[0],
                opacity=0.3,
                name=f'Fase: {phase}',
                legendgroup='phases'
            ))
 
    # Línea de anomalía ONI
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
        legend_title_text='Leyenda',
        yaxis_range=y_range,
        xaxis_rangeslider_visible=True
    )
    return fig
 
@st.cache_data
def preprocess_data(uploaded_file_mapa, uploaded_file_precip, uploaded_zip_shapefile):
    """Función principal para cargar y preprocesar todos los archivos."""
    df_precip_anual = load_data(uploaded_file_mapa)
    df_precip_mensual_raw = load_data(uploaded_file_precip)
    gdf_municipios = load_shapefile(uploaded_zip_shapefile)
    
    if any(df is None for df in [df_precip_anual, df_precip_mensual_raw, gdf_municipios]):
        return None, None, None, None, None
 
    # Preprocesamiento de datos anuales y de estaciones
    lon_col = next((col for col in df_precip_anual.columns if 'longitud' in col or 'lon' in col), None)
    lat_col = next((col for col in df_precip_anual.columns if 'latitud' in col or 'lat' in col), None)
    if not all([lon_col, lat_col]):
        st.error("No se encontraron las columnas de longitud y/o latitud en el archivo de estaciones.")
        return None, None, None, None, None
    
    for col in [lon_col, lat_col]:
        df_precip_anual[col] = pd.to_numeric(df_precip_anual[col].astype(str).str.replace(',', '.'), errors='coerce')
    
    df_precip_anual.dropna(subset=[lon_col, lat_col], inplace=True)
    gdf_temp = gpd.GeoDataFrame(df_precip_anual, geometry=gpd.points_from_xy(df_precip_anual[lon_col], df_precip_anual[lat_col]), crs="EPSG:4326")
    gdf_stations = gdf_temp.copy()
    gdf_stations['longitud_geo'] = gdf_stations.geometry.x
    gdf_stations['latitud_geo'] = gdf_stations.geometry.y
 
    # Preprocesamiento de datos mensuales y ENSO
    df_precip_mensual = df_precip_mensual_raw.copy()
    station_cols = [col for col in df_precip_mensual.columns if col.isdigit()]
    if not station_cols:
        st.error("No se encontraron columnas de estación (ej: '12345') en el archivo de precipitación mensual.")
        return None, None, None, None, None
 
    id_vars = [col for col in ['id', 'fecha_mes_año', 'año', 'mes', 'enso_año', 'enso_mes', 'anomalia_oni', 'temp_sst', 'temp_media'] if col in df_precip_mensual.columns]
    df_long = df_precip_mensual.melt(id_vars=id_vars, value_vars=station_cols, var_name='id_estacion', value_name='precipitation')
    df_long['precipitation'] = pd.to_numeric(df_long['precipitation'].astype(str).str.replace(',', '.'), errors='coerce')
    df_long.dropna(subset=['precipitation'], inplace=True)
    df_long['fecha_mes_año'] = pd.to_datetime(df_long['fecha_mes_año'], format='%b-%y', errors='coerce')
    df_long.dropna(subset=['fecha_mes_año'], inplace=True)
    df_long['origen'] = 'Original'
 
    gdf_stations['id_estacio'] = gdf_stations['id_estacio'].astype(str).str.strip()
    df_long['id_estacion'] = df_long['id_estacion'].astype(str).str.strip()
    station_mapping = gdf_stations.set_index('id_estacio')['nom_est'].to_dict()
    df_long['nom_est'] = df_long['id_estacion'].map(station_mapping)
    df_long.dropna(subset=['nom_est'], inplace=True)
 
    enso_cols = [col for col in ['id', 'fecha_mes_año', 'año', 'mes', 'anomalia_oni', 'temp_sst', 'temp_media'] if col in df_precip_mensual.columns]
    df_enso = df_precip_mensual[enso_cols].drop_duplicates().copy()
    for col in ['anomalia_oni', 'temp_sst', 'temp_media']:
        if col in df_enso.columns:
            df_enso[col] = pd.to_numeric(df_enso[col].astype(str).str.replace(',', '.'), errors='coerce')
    df_enso['fecha_mes_año'] = pd.to_datetime(df_enso['fecha_mes_año'], format='%b-%y', errors='coerce')
    df_enso.dropna(subset=['fecha_mes_año'], inplace=True)
    
    return gdf_stations, df_precip_anual, gdf_municipios, df_long, df_enso
 
def display_map(gdf_filtered, gdf_municipios, map_centering, logo_gota_path):
    """Función para renderizar el mapa de estaciones de Folium."""
    controls_col, map_col = st.columns([1, 4])
 
    with controls_col:
        st.subheader("Controles del Mapa")
        m1, m2 = st.columns([1, 3])
        with m1:
            if os.path.exists(logo_gota_path):
                st.image(logo_gota_path, width=50)
        with m2:
            st.metric("Estaciones en Vista", len(gdf_filtered))
 
        st.markdown("---")
        st.markdown("### Vistas del Mapa")
        if st.button("Ajustar a Selección"):
            if not gdf_filtered.empty:
                bounds = gdf_filtered.total_bounds
                center_lat = (bounds[1] + bounds[3]) / 2
                center_lon = (bounds[0] + bounds[2]) / 2
                st.session_state.map_view = {"location": [center_lat, center_lon], "zoom": 9}
            else:
                st.warning("No hay estaciones seleccionadas para ajustar el mapa.")
 
    with map_col:
        if not gdf_filtered.empty:
            m = folium.Map(location=st.session_state.map_view["location"], zoom_start=st.session_state.map_view["zoom"], tiles="cartodbpositron")
            
            if map_centering == "Automático":
                bounds = gdf_filtered.total_bounds
                m.fit_bounds([[bounds[1], bounds[0]], [bounds[3], bounds[2]]])
 
            folium.GeoJson(gdf_municipios.to_json(), name='Municipios').add_to(m)
            for _, row in gdf_filtered.iterrows():
                html = f"<b>Estación:</b> {row['nom_est']}<br><b>Municipio:</b> {row['municipio']}"
                folium.Marker([row['latitud_geo'], row['longitud_geo']], tooltip=html).add_to(m)
            
            folium_static(m, width=1100, height=700)
        else:
            st.warning("No hay estaciones seleccionadas para mostrar en el mapa.")
 
# --- Interfaz de Usuario y Lógica Principal ---
 
# Título y logo
logo_path = "CuencaVerdeLogo_V1.JPG"
logo_gota_path = "CuencaVerdeGoticaLogo.JPG"
title_col1, title_col2 = st.columns([1, 5])
with title_col1:
    if os.path.exists(logo_path):
        st.image(logo_path, use_column_width='auto')
with title_col2:
    st.title('Visor de Precipitación y Fenómeno ENSO')
 
# Barra lateral de control
st.sidebar.header("Panel de Control")
with st.sidebar.expander("**Cargar Archivos**", expanded=True):
    uploaded_file_mapa = st.file_uploader("1. Estaciones (mapaCVENSO.csv)", type="csv")
    uploaded_file_precip = st.file_uploader("2. Precipitación y ENSO (DatosPptnmes_ENSO.csv)", type="csv")
    uploaded_zip_shapefile = st.file_uploader("3. Shapefile de municipios (.zip)", type="zip")
 
# Lógica de carga y preprocesamiento de datos
if not all([uploaded_file_mapa, uploaded_file_precip, uploaded_zip_shapefile]):
    st.info("Por favor, suba los 3 archivos requeridos para habilitar la aplicación.")
    st.stop()
 
if 'data_loaded' not in st.session_state:
    st.session_state.data_loaded = False
 
# Llama a la función de preprocesamiento solo si los archivos han cambiado o no se han cargado
if not st.session_state.data_loaded or any(
    file.id != st.session_state.get(f'{file.name}_id') for file in [uploaded_file_mapa, uploaded_file_precip, uploaded_zip_shapefile]
):
    st.session_state.gdf_stations, st.session_state.df_precip_anual, st.session_state.gdf_municipios, st.session_state.df_long, st.session_state.df_enso = preprocess_data(uploaded_file_mapa, uploaded_file_precip, uploaded_zip_shapefile)
    st.session_state.data_loaded = True
    for file in [uploaded_file_mapa, uploaded_file_precip, uploaded_zip_shapefile]:
        st.session_state[f'{file.name}_id'] = file.id
    st.rerun()
 
if st.session_state.gdf_stations is None:
    st.stop()
 
# Asignar variables desde el estado de la sesión
gdf_stations = st.session_state.gdf_stations
df_precip_anual = st.session_state.df_precip_anual
gdf_municipios = st.session_state.gdf_municipios
df_long = st.session_state.df_long
df_enso = st.session_state.df_enso
 
# --- Filtros en la Barra Lateral ---
st.sidebar.markdown("### Filtros de Visualización")
 
# Slider para filtrar por porcentaje de datos
if 'porc_datos' in gdf_stations.columns:
    gdf_stations['porc_datos'] = pd.to_numeric(gdf_stations['porc_datos'], errors='coerce').fillna(0)
    min_data_perc = st.sidebar.slider("Filtrar por % de datos mínimo:", 0, 100, 0)
    stations_master_list = gdf_stations[gdf_stations['porc_datos'] >= min_data_perc]
else:
    st.sidebar.text("Advertencia: Columna 'porc_datos' no encontrada.")
    stations_master_list = gdf_stations.copy()
 
# Selección en cascada
selected_municipios = st.sidebar.multiselect('1. Filtrar por Municipio', sorted(stations_master_list['municipio'].unique()))
stations_available = stations_master_list[stations_master_list['municipio'].isin(selected_municipios)] if selected_municipios else stations_master_list
selected_celdas = st.sidebar.multiselect('2. Filtrar por Celda_XY', sorted(stations_available['celda_xy'].unique()))
stations_available = stations_available[stations_available['celda_xy'].isin(selected_celdas)] if selected_celdas else stations_available
stations_options = sorted(stations_available['nom_est'].unique())
 
st.sidebar.markdown("---")
st.sidebar.markdown("### Selección de Estaciones")
 
# Lógica para seleccionar todas las estaciones
select_all = st.sidebar.checkbox("Seleccionar/Deseleccionar todas las estaciones", key='select_all_checkbox')
if select_all:
    selected_stations = st.sidebar.multiselect('3. Seleccionar Estaciones', options=stations_options, default=stations_options)
else:
    selected_stations = st.sidebar.multiselect('3. Seleccionar Estaciones', options=stations_options)
 
if not selected_stations:
    st.warning("Por favor, seleccione al menos una estación.")
    st.stop()
 
# Rango de años y meses
años_disponibles = sorted([int(col) for col in gdf_stations.columns if str(col).isdigit()])
if not años_disponibles:
    st.error("No se encontraron columnas de años (ej: '2020') en el archivo de estaciones.")
    st.stop()
year_range = st.sidebar.slider("4. Seleccionar Rango de Años", min(años_disponibles), max(años_disponibles), (min(años_disponibles), max(años_disponibles)))
meses_dict = {'Enero': 1, 'Febrero': 2, 'Marzo': 3, 'Abril': 4, 'Mayo': 5, 'Junio': 6, 'Julio': 7, 'Agosto': 8, 'Septiembre': 9, 'Octubre': 10, 'Noviembre': 11, 'Diciembre': 12}
meses_nombres = st.sidebar.multiselect("5. Seleccionar Meses", list(meses_dict.keys()), default=list(meses_dict.keys()))
meses_numeros = [meses_dict[m] for m in meses_nombres]
 
# Opción de análisis avanzado
st.sidebar.markdown("### Opciones de Análisis Avanzado")
analysis_mode = st.sidebar.radio("Análisis de Series Mensuales", ("Usar datos originales", "Completar series (interpolación)"))
 
# Cachear el resultado de la interpolación
if 'df_monthly_processed' not in st.session_state or st.session_state.analysis_mode != analysis_mode:
    st.session_state.analysis_mode = analysis_mode
    if analysis_mode == "Completar series (interpolación)":
        st.session_state.df_monthly_processed = complete_series(df_long)
    else:
        st.session_state.df_monthly_processed = df_long.copy()
 
df_monthly_to_process = st.session_state.df_monthly_processed
 
# --- Preparación de datos filtrados ---
df_anual_melted = gdf_stations[gdf_stations['nom_est'].isin(selected_stations)].melt(
    id_vars=['nom_est', 'longitud_geo', 'latitud_geo'],
    value_vars=[str(y) for y in range(year_range[0], year_range[1] + 1) if str(y) in gdf_stations.columns],
    var_name='año', value_name='precipitacion')
 
df_monthly_filtered = df_monthly_to_process[
    (df_monthly_to_process['nom_est'].isin(selected_stations)) &
    (df_monthly_to_process['fecha_mes_año'].dt.year >= year_range[0]) &
    (df_monthly_to_process['fecha_mes_año'].dt.year <= year_range[1]) &
    (df_monthly_to_process['fecha_mes_año'].dt.month.isin(meses_numeros))
]
 
# --- Pestañas Principales ---
tab1, tab2, tab_anim, tab3, tab_stats, tab4, tab5 = st.tabs(["Gráficos", "Mapa de Estaciones", "Mapas Avanzados", "Tabla de Estaciones", "Estadísticas", "Análisis ENSO", "Descargas"])
 
with tab1:
    st.header("Visualizaciones de Precipitación")
    sub_tab_anual, sub_tab_mensual = st.tabs(["Serie Anual", "Serie Mensual"])
    
    with sub_tab_anual:
        with st.expander("Ver Gráfico de Precipitación Anual", expanded=True):
            if not df_anual_melted.empty:
                st.subheader("Precipitación Anual (mm)")
                chart_anual = alt.Chart(df_anual_melted).mark_line(point=True).encode(
                    x=alt.X('año:O', title='Año'),
                    y=alt.Y('precipitacion:Q', title='Precipitación (mm)'),
                    color='nom_est:N',
                    tooltip=['nom_est', 'año', alt.Tooltip('precipitacion', format=".2f")]
                ).properties(height=600).interactive()
                st.altair_chart(chart_anual, use_container_width=True)
 
        with st.expander("Ver Análisis de Precipitación Media Multianual"):
            if not df_anual_melted.empty:
                st.subheader("Análisis de Precipitación Media Multianual")
                chart_type_annual = st.radio("Seleccionar tipo de gráfico:", ("Gráfico de Barras (Promedio)", "Gráfico de Cajas (Distribución)"), key="avg_chart_type_annual", horizontal=True)
 
                if chart_type_annual == "Gráfico de Barras (Promedio)":
                    df_summary = df_anual_melted.groupby('nom_est', as_index=False)['precipitacion'].mean().round(2)
                    fig_avg = px.bar(df_summary, x='nom_est', y='precipitacion', title='Promedio de Precipitación Anual', labels={'nom_est': 'Estación', 'precipitacion': 'Precipitación Media Anual (mm)'}, color='precipitacion', color_continuous_scale=px.colors.sequential.YlGnBu)
                    st.plotly_chart(fig_avg, use_container_width=True)
                else:
                    fig_box = px.box(df_anual_melted, x='nom_est', y='precipitacion', color='nom_est', points='all', title='Distribución de la Precipitación Anual por Estación', labels={'nom_est': 'Estación', 'precipitacion': 'Precipitación Anual (mm)'})
                    st.plotly_chart(fig_box, use_container_width=True)
 
    with sub_tab_mensual:
        if not df_monthly_filtered.empty:
            with st.expander("Ver Gráfico de Precipitación Mensual", expanded=True):
                control_col1, control_col2 = st.columns(2)
                chart_type = control_col1.radio("Tipo de Gráfico:", ["Líneas y Puntos", "Nube de Puntos", "Gráfico de Cajas (Distribución Mensual)"], key="monthly_chart_type")
                color_by = control_col2.radio("Colorear por:", ["Estación", "Mes"], key="monthly_color_by", disabled=(chart_type == "Gráfico de Cajas (Distribución Mensual)"))
 
                if chart_type != "Gráfico de Cajas (Distribución Mensual)":
                    base_chart = alt.Chart(df_monthly_filtered).encode(
                        x=alt.X('fecha_mes_año:T', title='Fecha'),
                        y=alt.Y('precipitation:Q', title='Precipitación (mm)'),
                        tooltip=[alt.Tooltip('fecha_mes_año', format='%Y-%m'), 'precipitation', 'nom_est', 'origen', alt.Tooltip('mes:N', title="Mes")]
                    )
                    color_encoding = alt.Color('nom_est:N' if color_by == "Estación" else 'month(fecha_mes_año):N', legend=alt.Legend(title=color_by))
                    final_chart = (base_chart.mark_line(opacity=0.4, color='lightgray').encode(detail='nom_est:N') + base_chart.mark_point(filled=True, size=60).encode(color=color_encoding)) if chart_type == "Líneas y Puntos" else base_chart.mark_point(filled=True, size=60).encode(color=color_encoding)
                    st.altair_chart(final_chart.properties(height=600).interactive(), use_container_width=True)
                else:
                    fig_box_monthly = px.box(df_monthly_filtered, x='mes', y='precipitation', color='nom_est', title='Distribución de la Precipitación por Mes', labels={'mes': 'Mes', 'precipitation': 'Precipitación Mensual (mm)', 'nom_est': 'Estación'})
                    st.plotly_chart(fig_box_monthly, use_container_width=True)
 
        with st.expander("Ver Análisis del Fenómeno ENSO"):
            enso_filtered = df_enso[(df_enso['fecha_mes_año'].dt.year >= year_range[0]) & (df_enso['fecha_mes_año'].dt.year <= year_range[1]) & (df_enso['fecha_mes_año'].dt.month.isin(meses_numeros))]
            fig_enso_mensual = create_enso_chart(enso_filtered)
            st.plotly_chart(fig_enso_mensual, use_container_width=True, key="enso_chart_mensual")
 
with tab2:
    st.header("Mapa de Ubicación de Estaciones")
    gdf_filtered = gdf_stations[gdf_stations['nom_est'].isin(selected_stations)]
    display_map(gdf_filtered, gdf_municipios, "Automático", logo_gota_path)
 
with tab_anim:
    st.header("Mapas Avanzados")
    with st.expander("Ver Animación de Puntos", expanded=True):
        st.subheader("Mapa Animado de Precipitación Anual")
        if not df_anual_melted.empty:
            fig_mapa_animado = px.scatter_geo(df_anual_melted, lat='latitud_geo', lon='longitud_geo', color='precipitacion', size='precipitacion', hover_name='nom_est', animation_frame='año', projection='natural earth', title='Precipitación Anual por Estación', color_continuous_scale=px.colors.sequential.YlGnBu)
            fig_mapa_animado.update_geos(fitbounds="locations", visible=True)
            st.plotly_chart(fig_mapa_animado, use_container_width=True)
 
    with st.expander("Ver Comparación de Mapas anuales & Kriging", expanded=True):
        if not df_anual_melted.empty and len(df_anual_melted['año'].unique()) > 0:
            st.sidebar.markdown("### Opciones de Mapa Comparativo")
            min_precip, max_precip = int(df_anual_melted['precipitacion'].min()), int(df_anual_melted['precipitacion'].max())
            color_range = st.sidebar.slider("Rango de Escala de Color (mm)", min_precip, max_precip, (min_precip, max_precip))
            col1, col2 = st.columns(2)
            min_year, max_year = int(df_anual_melted['año'].min()), int(df_anual_melted['año'].max())
            year1 = col1.slider("Seleccione el año para el Mapa 1", min_year, max_year, max_year)
            year2 = col2.slider("Seleccione el año para el Mapa 2", min_year, max_year, max_year - 1 if max_year > min_year else max_year)
 
            if st.button("Generar Mapas de Comparación"):
                if year1 == year2:
                    with st.spinner("Generando mapas..."):
                        st.info("Años iguales: Mapa 1 muestra Puntos, Mapa 2 muestra Superficie Kriging.")
                        map_col1, map_col2 = st.columns(2)
                        data_year = df_anual_melted[df_anual_melted['año'].astype(int) == year1]
                        
                        with map_col1:
                            st.subheader(f"Estaciones - Año: {year1}")
                            if data_year.empty:
                                st.warning(f"No hay datos para el año {year1}.")
                            else:
                                fig1 = px.scatter_geo(data_year, lat='latitud_geo', lon='longitud_geo', color='precipitacion', size='precipitacion', hover_name='nom_est', color_continuous_scale='YlGnBu', projection='natural earth', range_color=color_range)
                                fig1.update_geos(fitbounds="locations", visible=True)
                                st.plotly_chart(fig1, use_container_width=True)
 
                        with map_col2:
                            st.subheader(f"Interpolación Kriging - Año: {year1}")
                            if len(data_year) < 3:
                                st.warning(f"Se necesitan al menos 3 estaciones para generar el mapa Kriging del año {year1}.")
                            else:
                                lons, lats, vals = data_year['longitud_geo'].values, data_year['latitud_geo'].values, data_year['precipitacion'].values
                                bounds = gpd.GeoDataFrame(data_year, geometry=gpd.points_from_xy(lons, lats), crs="EPSG:4326").total_bounds
                                lon_range = [bounds[0] - 0.1, bounds[2] + 0.1]
                                lat_range = [bounds[1] - 0.1, bounds[3] + 0.1]
                                grid_lon, grid_lat = np.linspace(lon_range[0], lon_range[1], 100), np.linspace(lat_range[0], lat_range[1], 100)
                                OK = OrdinaryKriging(lons, lats, vals, variogram_model='linear', verbose=False, enable_plotting=False)
                                z, ss = OK.execute('grid', grid_lon, grid_lat)
                                fig2 = go.Figure(data=go.Contour(z=z, x=grid_lon, y=grid_lat, colorscale='YlGnBu', zmin=color_range[0], zmax=color_range[1], contours=dict(showlabels=True, labelfont=dict(size=12, color='white'))))
                                fig2.add_trace(go.Scatter(x=lons, y=lats, mode='markers', marker=dict(color='red', size=4), name='Estaciones'))
                                fig2.update_xaxes(range=lon_range)
                                fig2.update_yaxes(range=lat_range, scaleanchor="x", scaleratio=1)
                                fig2.update_layout(height=600, xaxis_title="Longitud", yaxis_title="Latitud")
                                st.plotly_chart(fig2, use_container_width=True)
                else:
                    with st.spinner("Generando mapas..."):
                        st.info("Años diferentes: Se comparan los Puntos de Estaciones para cada año.")
                        map_col1, map_col2 = st.columns(2)
                        for i, (col, year) in enumerate(zip([map_col1, map_col2], [year1, year2])):
                            with col:
                                st.subheader(f"Estaciones - Año: {year}")
                                data_year = df_anual_melted[df_anual_melted['año'].astype(int) == year]
                                if data_year.empty:
                                    st.warning(f"No hay datos para el año {year}.")
                                    continue
                                fig = px.scatter_geo(data_year, lat='latitud_geo', lon='longitud_geo', color='precipitacion', size='precipitacion', hover_name='nom_est', color_continuous_scale='YlGnBu', range_color=color_range, projection='natural earth')
                                fig.update_geos(fitbounds="locations", visible=True)
                                st.plotly_chart(fig, use_container_width=True, key=f'map_diff_{i}')
 
    with st.expander("Mapa Animado del Fenómeno ENSO"):
        st.subheader("Evolución Mensual del Fenómeno ENSO")
        if not df_enso.empty and not gdf_stations.empty:
            st.info("El color de cada estación representa la fase del fenómeno ENSO a nivel global para cada mes.")
            stations_subset = gdf_stations[['nom_est', 'latitud_geo', 'longitud_geo']]
            enso_anim_data = df_enso[['fecha_mes_año', 'anomalia_oni']].copy()
            enso_anim_data.dropna(subset=['anomalia_oni'], inplace=True)
            conditions = [enso_anim_data['anomalia_oni'] >= 0.5, enso_anim_data['anomalia_oni'] <= -0.5]
            phases = ['El Niño', 'La Niña']
            enso_anim_data['fase'] = np.select(conditions, phases, default='Neutral')
            enso_anim_data['fecha_str'] = enso_anim_data['fecha_mes_año'].dt.strftime('%Y-%m')
            enso_anim_data = enso_anim_data[(enso_anim_data['fecha_mes_año'].dt.year >= year_range[0]) & (enso_anim_data['fecha_mes_año'].dt.year <= year_range[1])]
            animation_df = pd.merge(stations_subset.assign(key=1), enso_anim_data.assign(key=1), on='key').drop('key', axis=1)
            
            fig_enso_anim = px.scatter_geo(
                animation_df,
                lat='latitud_geo', lon='longitud_geo', color='fase', animation_frame='fecha_str',
                hover_name='nom_est', color_discrete_map={'El Niño': 'red', 'La Niña': 'blue', 'Neutral': 'lightgrey'},
                category_orders={"fase": ["El Niño", "La Niña", "Neutral"]}, projection='natural earth'
            )
            fig_enso_anim.update_geos(fitbounds="locations", visible=True)
            st.plotly_chart(fig_enso_anim, use_container_width=True)
 
with tab3:
    st.header("Información Detallada de las Estaciones")
    if not df_anual_melted.empty:
        display_cols = [col for col in gdf_stations.columns if col != 'geometry']
        df_info_table = gdf_stations[display_cols]
        df_mean_precip = df_anual_melted.groupby('nom_est')['precipitacion'].mean().round(2).reset_index().rename(columns={'precipitacion': 'Precipitación media anual (mm)'})
        df_info_table = df_info_table.merge(df_mean_precip, on='nom_est', how='left')
        st.dataframe(df_info_table[df_info_table['nom_est'].isin(selected_stations)])
    else:
        st.info("No hay datos de precipitación anual para mostrar en la selección actual.")
 
with tab_stats:
    st.header("Estadísticas de Precipitación")
    st.subheader("Matriz de Disponibilidad de Datos Anual")
    
    original_data_counts = df_long[df_long['nom_est'].isin(selected_stations)].groupby(['nom_est', 'año']).size().reset_index(name='count')
    original_data_counts['porc_original'] = (original_data_counts['count'] / 12) * 100
    heatmap_original_df = original_data_counts.pivot(index='nom_est', columns='año', values='porc_original')
 
    heatmap_df, color_scale, title_text = heatmap_original_df, "Greens", "Porcentaje de Datos Originales (%) por Estación y Año"
 
    if analysis_mode == "Completar series (interpolación)":
        view_mode = st.radio("Seleccione la vista de la matriz:", ("Porcentaje de Datos Originales", "Porcentaje de Datos Completados"), horizontal=True)
        if view_mode == "Porcentaje de Datos Completados":
            completed_data = df_monthly_to_process[(df_monthly_to_process['nom_est'].isin(selected_stations)) & (df_monthly_to_process['origen'] == 'Completado')]
            if not completed_data.empty:
                completed_counts = completed_data.groupby(['nom_est', 'año']).size().reset_index(name='count')
                completed_counts['porc_completado'] = (completed_counts['count'] / 12) * 100
                heatmap_df, color_scale, title_text = completed_counts.pivot(index='nom_est', columns='año', values='porc_completado'), "Reds", "Porcentaje de Datos Completados (%) por Estación y Año"
            else:
                heatmap_df = pd.DataFrame()
 
    if not heatmap_df.empty:
        fig_heatmap = px.imshow(heatmap_df, text_auto='.0f', aspect="auto", color_continuous_scale=color_scale, labels=dict(x="Año", y="Estación", color="% Datos"), title=title_text)
        fig_heatmap.update_layout(height=max(400, len(selected_stations) * 40))
        st.plotly_chart(fig_heatmap, use_container_width=True)
    else:
        st.info("No hay datos para mostrar en la matriz con la selección actual.")
 
    st.markdown("---")
    if not df_monthly_filtered.empty and not df_anual_melted.empty:
        st.subheader("Síntesis General")
        max_annual_row = df_anual_melted.loc[df_anual_melted['precipitacion'].idxmax()]
        max_monthly_row = df_monthly_filtered.loc[df_monthly_filtered['precipitation'].idxmax()]
        col1, col2 = st.columns(2)
        col1.metric("Máxima Ppt. Anual Registrada", f"{max_annual_row['precipitacion']:.1f} mm", f"{max_annual_row['nom_est']} (Año {max_annual_row['año']})")
        col2.metric("Máxima Ppt. Mensual Registrada", f"{max_monthly_row['precipitation']:.1f} mm", f"{max_monthly_row['nom_est']} ({max_monthly_row['fecha_mes_año'].strftime('%Y-%m')})")
        st.markdown("---")
        st.subheader("Resumen de Estadísticas Mensuales por Estación")
        summary_data = []
        for station_name, group in df_monthly_filtered.groupby('nom_est'):
            if group['precipitation'].empty: continue
            max_row = group.loc[group['precipitation'].idxmax()]
            min_row = group.loc[group['precipitation'].idxmin()]
            summary_data.append({"Estación": station_name, "Ppt. Máxima Mensual (mm)": max_row['precipitation'], "Fecha Máxima": max_row['fecha_mes_año'].strftime('%Y-%m'), "Ppt. Mínima Mensual (mm)": min_row['precipitation'], "Fecha Mínima": min_row['fecha_mes_año'].strftime('%Y-%m'), "Promedio Mensual (mm)": group['precipitation'].mean()})
        summary_df = pd.DataFrame(summary_data)
        st.dataframe(summary_df.round(2), use_container_width=True)
 
with tab4:
    st.header("Análisis de Precipitación y el Fenómeno ENSO")
    if df_enso.empty:
        st.warning("No se encontraron datos del fenómeno ENSO en el archivo de precipitación cargado. El análisis ENSO no está disponible.")
    else:
        enso_series_tab, enso_corr_tab = st.tabs(["Series de Tiempo ENSO", "Correlación Precipitación-ENSO"])
        with enso_series_tab:
            st.subheader("Visualización de Variables ENSO")
            enso_vars_available = [v for v in ['anomalia_oni', 'temp_sst', 'temp_media'] if v in df_enso.columns]
            if not enso_vars_available:
                st.warning("No hay variables ENSO disponibles en el archivo de datos para visualizar.")
            else:
                variable_enso = st.selectbox("Seleccione la variable ENSO a visualizar:", enso_vars_available)
                df_enso_filtered = df_enso[(df_enso['fecha_mes_año'].dt.year >= year_range[0]) & (df_enso['fecha_mes_año'].dt.year <= year_range[1]) & (df_enso['fecha_mes_año'].dt.month.isin(meses_numeros))]
                if not df_enso_filtered.empty and variable_enso in df_enso_filtered.columns and not df_enso_filtered[variable_enso].isnull().all():
                    fig_enso_series = px.line(df_enso_filtered, x='fecha_mes_año', y=variable_enso, title=f"Serie de Tiempo para {variable_enso}")
                    st.plotly_chart(fig_enso_series, use_container_width=True)
                else:
                    st.warning(f"No hay datos disponibles para '{variable_enso}' en el período seleccionado.")
 
        with enso_corr_tab:
            df_analisis = pd.merge(df_monthly_filtered.copy(), df_enso, on=['fecha_mes_año'], how='left', suffixes=('_precip', '_enso'))
            if 'anomalia_oni' in df_analisis.columns:
                df_analisis.dropna(subset=['anomalia_oni'], inplace=True)
                df_analisis['enso_fase'] = df_analisis['anomalia_oni'].apply(lambda oni: 'El Niño' if oni >= 0.5 else 'La Niña' if oni <= -0.5 else 'Neutral')
 
                if not df_analisis.empty:
                    st.subheader("Precipitación Media por Evento ENSO")
                    df_enso_group = df_analisis.groupby('enso_fase')['precipitation'].mean().reindex(['El Niño', 'La Niña', 'Neutral']).reset_index()
                    fig_enso = px.bar(df_enso_group, x='enso_fase', y='precipitation', color='enso_fase', labels={'precipitation': 'Precipitación Media (mm)'})
                    st.plotly_chart(fig_enso, use_container_width=True)
 
                    st.subheader("Correlación entre Anomalía ONI y Precipitación")
                    if df_analisis['anomalia_oni'].nunique() > 1 and df_analisis['precipitation'].nunique() > 1:
                        correlation = df_analisis['anomalia_oni'].corr(df_analisis['precipitation'])
                        st.metric("Coeficiente de Correlación de Pearson", f"{correlation:.2f}")
                    else:
                        st.warning("No hay suficientes datos variados para calcular la correlación.")
                else:
                    st.warning("No hay datos suficientes para realizar el análisis ENSO con la selección actual.")
            else:
                st.warning("Análisis no disponible. Falta la columna 'anomalia_oni' en el archivo de datos.")
 
with tab5:
    st.header("Opciones de Descarga")
    @st.cache_data
    def convert_df_to_csv(df):
        return df.to_csv(index=False).encode('utf-8')
 
    if not df_anual_melted.empty:
        st.markdown("**Datos de Precipitación Anual (Filtrados)**")
        st.download_button("Descargar CSV Anual", convert_df_to_csv(df_anual_melted), 'precipitacion_anual.csv', 'text/csv', key='download-anual')
        
    if not df_monthly_filtered.empty:
        st.markdown("**Datos de Precipitación Mensual (Filtrados)**")
        st.download_button("Descargar CSV Mensual", convert_df_to_csv(df_monthly_filtered), 'precipitacion_mensual.csv', 'text/csv', key='download-mensual')
 
    if analysis_mode == "Completar series (interpolación)" and not df_monthly_to_process.empty:
        st.markdown("**Datos de Precipitación Mensual (Series Completadas)**")
        st.download_button("Descargar CSV con Series Completadas", convert_df_to_csv(df_monthly_to_process), 'precipitacion_mensual_completada.csv', 'text/csv', key='download-completado')
    else:
        st.info("Para descargar las series completadas, seleccione la opción 'Completar series (interpolación)' en el panel lateral.")
