# --- Importaciones ---
import streamlit as st
import pandas as pd
import altair as alt
import folium
from streamlit_folium import st_folium
import plotly.express as px
import plotly.graph_objects as go
import geopandas as gpd
import zipfile
import tempfile
import os
import io
import numpy as np
from pykrige.ok import OrdinaryKriging
from folium.plugins import MarkerCluster

try:
    from folium.plugins import ScaleControl
except ImportError:
    class ScaleControl:
        def __init__(self, *args, **kwargs): pass
        def add_to(self, m): pass

# --- Configuración de la página ---
st.set_page_config(layout="wide", page_title="Visor de Precipitación y ENSO", page_icon="💧")

# --- CSS para optimizar el espacio y estilo de métricas ---
st.markdown("""
<style>
div.block-container {padding-top: 2rem;}
.sidebar .sidebar-content {font-size: 13px; }
h1 { margin-top: 0px; padding-top: 0px; }
[data-testid="stMetricValue"] {
    font-size: 1.8rem;
}
[data-testid="stMetricLabel"] {
    font-size: 1rem;
    padding-bottom: 5px;
}
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
            # Asignar un CRS si no existe
            if gdf.crs is None:
                lon_col = next((col for col in gdf.columns if 'longitud' in col.lower() or 'lon' in col.lower() or 'x' in col.lower()), None)
                lat_col = next((col for col in gdf.columns if 'latitud' in col.lower() or 'lat' in col.lower() or 'y' in col.lower()), None)
                if lon_col and lat_col:
                    # Convertir a numérico antes de la comparación
                    gdf[lon_col] = pd.to_numeric(gdf[lon_col], errors='coerce')
                    gdf[lat_col] = pd.to_numeric(gdf[lat_col], errors='coerce')
                    if (gdf[lon_col].dropna().max() < 180 and gdf[lon_col].dropna().min() > -180 and
                        gdf[lat_col].dropna().max() < 90 and gdf[lat_col].dropna().min() > -90):
                         gdf.set_crs("EPSG:4326", inplace=True)
                    else:
                         gdf.set_crs("EPSG:9377", inplace=True)
            return gdf.to_crs("EPSG:4326")
    except Exception as e:
       st.error(f"Error al procesar el shapefile: {e}")
       return None

@st.cache_data
def complete_series(_df):
    all_completed_dfs = []
    station_list = _df['Nom_Est'].unique()
    progress_bar = st.progress(0, text="Completando todas las series...")
    for i, station in enumerate(station_list):
        df_station = _df.loc[_df['Nom_Est'] == station].copy()
        df_station.set_index('Fecha', inplace=True)
        if not df_station.index.is_unique:
            df_station = df_station[~df_station.index.duplicated(keep='first')]
        
        # Crear un rango de fechas completo
        if not df_station.empty:
            start_date = df_station.index.min()
            end_date = df_station.index.max()
            full_date_range = pd.date_range(start=start_date, end=end_date, freq='MS')
            
            # Reindexar el DataFrame
            df_resampled = df_station.reindex(full_date_range)
            
            # Conservar las columnas originales y completar las faltantes
            df_resampled['Precipitation'] = df_resampled['Precipitation'].interpolate(method='time')
            df_resampled['Origen'] = df_resampled['Origen'].fillna('Completado')
            df_resampled['Nom_Est'] = station
            df_resampled['año'] = df_resampled.index.year
            df_resampled['mes'] = df_resampled.index.month
            df_resampled.reset_index(inplace=True)
            df_resampled.rename(columns={'index': 'Fecha'}, inplace=True)
            all_completed_dfs.append(df_resampled)
        progress_bar.progress((i + 1) / len(station_list), text=f"Completando series... Estación: {station}")
    progress_bar.empty()
    if not all_completed_dfs:
        return pd.DataFrame()
    return pd.concat(all_completed_dfs, ignore_index=True)

# --- Interfaz y Carga de Archivos ---
logo_path = "CuencaVerdeLogo_V1.JPG"
logo_gota_path = "CuencaVerdeGoticaLogo.JPG"

title_col1, title_col2 = st.columns([1, 5])
with title_col1:
    if os.path.exists(logo_gota_path):
       st.image(logo_gota_path, width=50)
with title_col2:
    st.title('Visor de Precipitación y Fenómeno ENSO')

st.sidebar.header("Panel de Control")
with st.sidebar.expander("Cargar Archivos", expanded=True):
    uploaded_file_mapa = st.file_uploader("1. Estaciones (mapaCVENSO.csv) 🗺️", type="csv")
    uploaded_file_precip = st.file_uploader("2. Precipitación y ENSO (DatosPptnmes_ENSO.csv) 📈", type="csv")
    uploaded_zip_shapefile = st.file_uploader("3. Shapefile de municipios (.zip) 🌎", type="zip")
    
    if st.button("Limpiar datos cargados"):
        st.cache_data.clear()
        st.rerun()

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

# Lógica de detección y renombrado de columnas mejorada
date_col_name_full = next((col for col in df_precip_mensual.columns if 'fecha_mes_aã±o' in col or 'fecha_mes_año' in col), None)

if date_col_name_full:
    df_precip_mensual['Fecha'] = pd.to_datetime(df_precip_mensual[date_col_name_full], format='%b-%y', errors='coerce')
    df_precip_mensual.dropna(subset=['Fecha'], inplace=True)
    df_precip_mensual['año'] = df_precip_mensual['Fecha'].dt.year
    df_precip_mensual['mes'] = df_precip_mensual['Fecha'].dt.month
else:
    # Si no se encuentra la columna de fecha completa, se recurre a las columnas 'año' y 'mes'
    year_col_name = next((col for col in df_precip_mensual.columns if 'año' in col or 'aã±o' in col), None)
    month_col_name = next((col for col in df_precip_mensual.columns if 'mes' in col), None)

    if not all([year_col_name, month_col_name]):
        st.error("No se encontraron las columnas de fecha necesarias ('fecha_mes_año' o 'año' y 'mes') en el archivo de precipitación mensual. Por favor, asegúrese de que existan.")
        st.stop()
    
    df_precip_mensual.rename(columns={year_col_name: 'año', month_col_name: 'mes'}, inplace=True)
    df_precip_mensual['año'] = pd.to_numeric(df_precip_mensual['año'], errors='coerce').fillna(-1).astype(int)
    df_precip_mensual['mes'] = pd.to_numeric(df_precip_mensual['mes'], errors='coerce').fillna(-1).astype(int)
    df_precip_mensual.dropna(subset=['año', 'mes'], inplace=True)
    df_precip_mensual['Fecha'] = pd.to_datetime(df_precip_mensual[['año', 'mes']].assign(day=1), errors='coerce')
    df_precip_mensual.dropna(subset=['Fecha'], inplace=True)

id_col_name = next((col for col in df_precip_mensual_raw.columns if col.lower() == 'id'), None)
if id_col_name:
    df_precip_mensual.rename(columns={id_col_name: 'Id'}, inplace=True)

precip_col_name = next((col for col in df_precip_mensual_raw.columns if 'precipitacion' in col.lower()), None)

# Corrección clave: Se define el df_enso con columnas específicas y se procesan sus tipos de datos
enso_cols = ['año', 'mes', 'anomalia_oni', 'temp_sst']
if all(col in df_precip_mensual.columns for col in enso_cols):
    df_enso = df_precip_mensual[enso_cols].drop_duplicates().copy()
    
    # Se asegura que las columnas de año y mes sean numéricas para crear la fecha
    df_enso['año'] = pd.to_numeric(df_enso['año'], errors='coerce').fillna(-1).astype(int)
    df_enso['mes'] = pd.to_numeric(df_enso['mes'], errors='coerce').fillna(-1).astype(int)
    df_enso.dropna(subset=['año', 'mes'], inplace=True)
    
    df_enso['Fecha'] = pd.to_datetime(df_enso[['año', 'mes']].assign(day=1), errors='coerce')
    df_enso['fecha_merge'] = df_enso['Fecha'].dt.strftime('%Y-%m')
    df_enso.dropna(subset=['Fecha'], inplace=True)
    
    for col in ['anomalia_oni', 'temp_sst']:
        if col in df_enso.columns:
            df_enso[col] = pd.to_numeric(df_enso[col].astype(str).str.replace(',', '.'), errors='coerce')
else:
    df_enso = pd.DataFrame()


lon_col = next((col for col in df_precip_anual.columns if 'longitud' in col.lower() or 'lon' in col.lower()), None)
lat_col = next((col for col in df_precip_anual.columns if 'latitud' in col.lower() or 'lat' in col.lower()), None)
if not all([lon_col, lat_col]):
    st.error(f"No se encontraron las columnas de longitud y/o latitud en el archivo de estaciones.")
    st.stop()
df_precip_anual.loc[:, lon_col] = pd.to_numeric(df_precip_anual.loc[:, lon_col].astype(str).str.replace(',', '.'), errors='coerce')
df_precip_anual.loc[:, lat_col] = pd.to_numeric(df_precip_anual.loc[:, lat_col].astype(str).str.replace(',', '.'), errors='coerce')
df_precip_anual.dropna(subset=[lon_col, lat_col], inplace=True)

gdf_temp = gpd.GeoDataFrame(df_precip_anual, geometry=gpd.points_from_xy(df_precip_anual[lon_col], df_precip_anual[lat_col]), crs="EPSG:4326")
gdf_stations = gdf_temp.copy()
gdf_stations.loc[:, 'Longitud_geo'] = gdf_stations.geometry.x
gdf_stations.loc[:, 'Latitud_geo'] = gdf_stations.geometry.y

station_cols = [col for col in df_precip_mensual.columns if col.isdigit()]
if not station_cols:
    st.error("No se encontraron columnas de estación (ej: '12345') en el archivo de precipitación mensual.")
    st.stop()

id_vars = ['año', 'mes', 'Fecha']
if 'Id' in df_precip_mensual.columns:
    id_vars.append('Id')

# Se renombra 'value_name' a 'Precipitation' si la columna original no se llamaba así
value_name_col = 'Precipitation' if not precip_col_name else 'Precipitation_temp'
df_long = df_precip_mensual.melt(id_vars=id_vars, value_vars=station_cols, var_name='Id_estacion', value_name=value_name_col)
df_long.loc[:, 'Precipitation'] = pd.to_numeric(df_long[value_name_col].astype(str).str.replace(',', '.'), errors='coerce')
if value_name_col != 'Precipitation':
    df_long.drop(columns=[value_name_col], inplace=True)
df_long.dropna(subset=['Precipitation'], inplace=True)
df_long.loc[:, 'Origen'] = 'Original'

gdf_stations.loc[:, 'Id_estacio'] = gdf_stations['Id_estacio'].astype(str).str.strip()
df_long.loc[:, 'Id_estacion'] = df_long['Id_estacion'].astype(str).str.strip()
station_mapping = gdf_stations.set_index('Id_estacio')['Nom_Est'].to_dict()
df_long.loc[:, 'Nom_Est'] = df_long['Id_estacion'].map(station_mapping)
df_long.dropna(subset=['Nom_Est'], inplace=True)

if df_long.empty:
    st.warning("El dataframe de precipitación mensual está vacío después del preprocesamiento.")
    st.stop()
    
# Se agrega esta línea para garantizar el tipo de dato de las columnas de año en `gdf_stations`
for col in gdf_stations.columns:
    try:
        if str(col).isdigit():
            gdf_stations[col] = pd.to_numeric(gdf_stations[col], errors='coerce')
    except (TypeError, ValueError):
        continue

# --- Controles en la Barra Lateral ---
st.sidebar.markdown("### Filtros de Visualización")
if 'Porc_datos' in gdf_stations.columns:
    gdf_stations.loc[:, 'Porc_datos'] = pd.to_numeric(gdf_stations['Porc_datos'], errors='coerce').fillna(0)
    min_data_perc = st.sidebar.slider("Filtrar por % de datos mínimo:", 0, 100, 0)
    stations_master_list = gdf_stations.loc[gdf_stations['Porc_datos'] >= min_data_perc]
else:
    st.sidebar.text("Advertencia: Columna 'Porc_datos' no encontrada.")
    stations_master_list = gdf_stations.copy()

municipios_list = sorted(stations_master_list['municipio'].unique())
celdas_list = sorted(stations_master_list['Celda_XY'].unique())
selected_municipios = st.sidebar.multiselect('1. Filtrar por Municipio', options=municipios_list)
selected_celdas = st.sidebar.multiselect('2. Filtrar por Celda_XY', options=celdas_list)
stations_available = stations_master_list.copy()
if selected_municipios:
    stations_available = stations_available.loc[stations_available['municipio'].isin(selected_municipios)]
if selected_celdas:
    stations_available = stations_available.loc[stations_available['Celda_XY'].isin(selected_celdas)]
stations_options = sorted(stations_available['Nom_Est'].unique())

if 'selected_stations' not in st.session_state or 'filter_changed' not in st.session_state:
    st.session_state.selected_stations = [stations_options[0]] if stations_options else []
    st.session_state.filter_changed = False

select_all = st.sidebar.checkbox("Seleccionar/Deseleccionar Todas las Estaciones", value=False)
if select_all:
    default_selection = stations_options
    st.session_state.selected_stations = stations_options
else:
    default_selection = st.session_state.selected_stations

selected_stations = st.sidebar.multiselect(
    '3. Seleccionar Estaciones',
   options=stations_options,
   default=default_selection
)
st.session_state.selected_stations = selected_stations

años_disponibles = sorted([int(col) for col in gdf_stations.columns if str(col).isdigit() and pd.api.types.is_numeric_dtype(gdf_stations[str(col)])])
if not años_disponibles:
    st.error("No se encontraron columnas de años (ej: '2020', '2021') en el archivo de estaciones.")
    st.stop()
year_range = st.sidebar.slider("4. Seleccionar Rango de Años", min(años_disponibles), max(años_disponibles), (min(años_disponibles), max(años_disponibles)))
meses_dict = {'Enero': 1, 'Febrero': 2, 'Marzo': 3, 'Abril': 4, 'Mayo': 5, 'Junio': 6, 'Julio': 7, 'Agosto': 8, 'Septiembre': 9, 'Octubre': 10, 'Noviembre': 11, 'Diciembre': 12}
meses_nombres = st.sidebar.multiselect("5. Seleccionar Meses", list(meses_dict.keys()), default=list(meses_dict.keys()))
meses_numeros = [meses_dict[m] for m in meses_nombres]
st.sidebar.markdown("### Opciones de Análisis Avanzado")
analysis_mode = st.sidebar.radio("Análisis de Series Mensuales", ("Usar datos originales", "Completar series (interpolación)"))

if analysis_mode == "Completar series (interpolación)":
    df_monthly_to_process = complete_series(df_long)
else:
    df_monthly_to_process = df_long.copy()

if not selected_stations or not meses_numeros:
    st.warning("Por favor, seleccione al menos una estación y un mes.")
    st.stop()

# --- Preparación de datos filtrados ---
df_anual_melted = gdf_stations.loc[gdf_stations['Nom_Est'].isin(selected_stations)].melt(
   id_vars=['Nom_Est', 'Longitud_geo', 'Latitud_geo'],
   value_vars=[str(y) for y in range(year_range[0], year_range[1] + 1) if str(y) in gdf_stations.columns and pd.api.types.is_numeric_dtype(gdf_stations[str(y)])],
   var_name='año', value_name='Precipitación')
df_anual_melted['año'] = pd.to_numeric(df_anual_melted['año'], errors='coerce')
df_anual_melted.dropna(subset=['año', 'Precipitación'], inplace=True)
df_anual_melted = df_anual_melted.astype({'año': int})

df_monthly_filtered = df_monthly_to_process.loc[
   (df_monthly_to_process['Nom_Est'].isin(selected_stations)) &
   (df_monthly_to_process['Fecha'].dt.year >= year_range[0]) &
   (df_monthly_to_process['Fecha'].dt.year <= year_range[1]) &
   (df_monthly_to_process['Fecha'].dt.month.isin(meses_numeros))
].copy()

# --- NUEVA FUNCIONALIDAD: Cálculo de Anomalías y Climatología ---
@st.cache_data
def calculate_anomalies(df_monthly_data, base_years):
    df_filtered_base = df_monthly_data.loc[(df_monthly_data['año'] >= base_years[0]) & (df_monthly_data['año'] <= base_years[1])]
    monthly_avg = df_filtered_base.groupby(['Nom_Est', 'mes'])['Precipitation'].mean().reset_index()
    monthly_avg.rename(columns={'Precipitation': 'Precipitation_long_term_avg'}, inplace=True)
    df_with_avg = pd.merge(df_monthly_data, monthly_avg, on=['Nom_Est', 'mes'], how='left')
    df_with_avg['Precipitation_Anomaly'] = df_with_avg['Precipitation'] - df_with_avg['Precipitation_long_term_avg']
    return df_with_avg

# --- Pestañas Principales ---
tab1, tab2, tab_anim, tab_stats, tab_anom, tab_enso, tab_descargas = st.tabs([
    "Gráficos 📈", "Mapa de Estaciones 🗺️", "Mapas Avanzados 🌍",
    "Estadísticas 📊", "Análisis de Anomalías 🔍", "Análisis ENSO 🌡️", "Descargas 📥"
])

# Resumen de filtros
st.info(f"Mostrando datos de **{len(selected_stations)}** estaciones para el período de **{year_range[0]}** a **{year_range[1]}** en los meses seleccionados: **{', '.join(meses_nombres)}**.")

with tab1:
    st.header("Visualizaciones de Precipitación")
    sub_tab_anual, sub_tab_mensual, sub_tab_monthly_avg = st.tabs(["Serie Anual", "Serie Mensual", "Climatología Mensual"])
    
    with sub_tab_anual:
        with st.expander("Ver Gráfico de Precipitación Anual", expanded=True):
            if not df_anual_melted.empty:
                st.subheader("Precipitación Anual (mm)")
                chart_anual = alt.Chart(df_anual_melted).mark_line(point=True).encode(
                    x=alt.X('año:O', title='Año'),
                    y=alt.Y('Precipitación:Q', title='Precipitación (mm)'),
                    color='Nom_Est:N',
                    tooltip=['Nom_Est', 'año', 'Precipitación']
                ).properties(height=600).interactive()
                st.altair_chart(chart_anual, use_container_width=True)
        
        with st.expander("Ver Análisis de Precipitación Media Multianual"):
            if not df_anual_melted.empty:
                st.subheader("Análisis de Precipitación Media Multianual")
                st.caption(f"Período de análisis: {year_range[0]} - {year_range[1]}")

                chart_type_annual = st.radio("Seleccionar tipo de gráfico:", 
                                             ("Gráfico de Barras (Promedio)", "Gráfico de Cajas (Distribución)"),
                                             key="avg_chart_type_annual", horizontal=True)

                if chart_type_annual == "Gráfico de Barras (Promedio)":
                    df_summary = df_anual_melted.groupby('Nom_Est', as_index=False)['Precipitación'].mean().round(2)
                    sort_order = st.radio(
                        "Ordenar estaciones por:",
                        ["Promedio (Mayor a Menor)", "Promedio (Menor a Mayor)", "Alfabético"],
                        horizontal=True, key="sort_annual_avg"
                    )

                    if "Mayor a Menor" in sort_order:
                        df_summary = df_summary.sort_values("Precipitación", ascending=False)
                    elif "Menor a Mayor" in sort_order:
                        df_summary = df_summary.sort_values("Precipitación", ascending=True)
                    else:
                        df_summary = df_summary.sort_values("Nom_Est", ascending=True)

                    fig_avg = px.bar(df_summary, x='Nom_Est', y='Precipitación', title='Promedio de Precipitación Anual',
                                     labels={'Nom_Est': 'Estación', 'Precipitación': 'Precipitación Media Anual (mm)'},
                                     color='Precipitación', color_continuous_scale=px.colors.sequential.Blues_r)
                    fig_avg.update_layout(
                        height=600,
                        xaxis={'categoryorder':'total descending' if "Mayor a Menor" in sort_order else ('total ascending' if "Menor a Mayor" in sort_order else 'trace')}
                    )
                    st.plotly_chart(fig_avg, use_container_width=True)
                else: 
                    fig_box = px.box(df_anual_melted, x='Nom_Est', y='Precipitación', color='Nom_Est',
                                     points='all',
                                     title='Distribución de la Precipitación Anual por Estación',
                                     labels={'Nom_Est': 'Estación', 'Precipitación': 'Precipitación Anual (mm)'})
                    fig_box.update_layout(height=600)
                    st.plotly_chart(fig_box, use_container_width=True)

    with sub_tab_mensual:
        if not df_monthly_filtered.empty:
            with st.expander("Ver Gráfico de Precipitación Mensual", expanded=True):
                control_col1, control_col2 = st.columns(2)
                chart_type = control_col1.radio("Tipo de Gráfico:", ["Líneas y Puntos", "Nube de Puntos", "Gráfico de Cajas (Distribución Mensual)"], key="monthly_chart_type")
                color_by = control_col2.radio("Colorear por:", ["Estación", "Mes"], key="monthly_color_by", disabled=(chart_type == "Gráfico de Cajas (Distribución Mensual)"))

                if chart_type != "Gráfico de Cajas (Distribución Mensual)":
                    base_chart = alt.Chart(df_monthly_filtered).encode(
                        x=alt.X('Fecha:T', title='Fecha'),
                        y=alt.Y('Precipitation:Q', title='Precipitación (mm)'),
                        tooltip=[alt.Tooltip('Fecha', format='%Y-%m'), 'Precipitation', 'Nom_Est', 'Origen', alt.Tooltip('mes:N', title="Mes")]
                    )
                    
                    if color_by == "Estación":
                        color_encoding = alt.Color('Nom_Est:N', legend=alt.Legend(title="Estaciones"))
                    else: 
                        color_encoding = alt.Color('month(Fecha):N', legend=alt.Legend(title="Meses"), scale=alt.Scale(scheme='tableau20'))

                    if chart_type == "Líneas y Puntos":
                        line_chart = base_chart.mark_line(opacity=0.4, color='lightgray').encode(detail='Nom_Est:N')
                        point_chart = base_chart.mark_point(filled=True, size=60).encode(color=color_encoding)
                        final_chart = (line_chart + point_chart)
                    else: # Nube de Puntos
                        point_chart = base_chart.mark_point(filled=True, size=60).encode(color=color_encoding)
                        final_chart = point_chart
                    
                    st.altair_chart(final_chart.properties(height=600).interactive(), use_container_width=True)
                else:
                    st.subheader("Distribución de la Precipitación Mensual")
                    fig_box_monthly = px.box(df_monthly_filtered, x='mes', y='Precipitation', color='Nom_Est',
                                             title='Distribución de la Precipitación por Mes',
                                             labels={'mes': 'Mes', 'Precipitation': 'Precipitación Mensual (mm)', 'Nom_Est': 'Estación'})
                    fig_box_monthly.update_layout(height=600)
                    st.plotly_chart(fig_box_monthly, use_container_width=True)
            
            with st.expander("Ver Tabla de Datos Detallados"):
                st.subheader("Datos de Precipitación Mensual Detallados")
                if not df_monthly_filtered.empty:
                    df_values = df_monthly_filtered.pivot_table(index='Fecha', columns='Nom_Est', values='Precipitation')
                    
                    st.dataframe(df_values, use_container_width=True)

    with sub_tab_monthly_avg:
        st.subheader("Climatología de Precipitación Mensual")
        st.caption("Promedio de precipitación por mes, calculado sobre el rango de años seleccionado.")
        df_monthly_avg = df_monthly_filtered.groupby(['Nom_Est', 'mes'])['Precipitation'].mean().reset_index()
        df_monthly_avg['mes_nombre'] = df_monthly_avg['mes'].map({v:k for k,v in meses_dict.items()})
        
        fig_clim = px.bar(df_monthly_avg, 
                          x='mes_nombre', 
                          y='Precipitation', 
                          color='Nom_Est',
                          barmode='group',
                          title='Precipitación Mensual Media Climatológica',
                          labels={'Precipitation': 'Precipitación Media (mm)', 'mes_nombre': 'Mes'})
        fig_clim.update_layout(height=600)
        st.plotly_chart(fig_clim, use_container_width=True)

# --- Mapa de Estaciones 🗺️
with tab2:
    st.header("Mapa de Ubicación de Estaciones")
    controls_col, map_col = st.columns([1, 4])
    gdf_filtered = gdf_stations.loc[gdf_stations['Nom_Est'].isin(selected_stations)].copy()

    with controls_col:
        st.subheader("Controles del Mapa")
        if not gdf_filtered.empty:
            
            m1, m2 = st.columns([1,3])
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
                    if not gdf_filtered.empty:
                        bounds = gdf_filtered.total_bounds
                        center_lat = (bounds[1] + bounds[3]) / 2
                        center_lon = (bounds[0] + bounds[2]) / 2
                        st.session_state.map_view = {"location": [center_lat, center_lon], "zoom": 9}
    
    with map_col:
        if not gdf_filtered.empty:
            m = folium.Map(location=st.session_state.map_view["location"], zoom_start=st.session_state.map_view["zoom"], tiles="cartodbpositron")
            
            if map_centering == "Automático":
                bounds = gdf_filtered.total_bounds
                m.fit_bounds([[bounds[1], bounds[0]], [bounds[3], bounds[2]]])

            folium.GeoJson(gdf_municipios.to_json(), name='Municipios').add_to(m)
            marker_cluster = MarkerCluster().add_to(m)
            for _, row in gdf_filtered.iterrows():
                html = f"<b>Estación:</b> {row['Nom_Est']}<br><b>Municipio:</b> {row['municipio']}"
                folium.Marker([row['Latitud_geo'], row['Longitud_geo']], tooltip=html).add_to(marker_cluster)
            
            st_folium(m, width=1100, height=700)
        else:
            st.warning("No hay estaciones seleccionadas para mostrar en el mapa.")

# --- Mapas Avanzados 🌍
with tab_anim:
    st.header("Mapas Avanzados")
    with st.expander("Ver Animación de Puntos", expanded=True):
        st.subheader("Mapa Animado de Precipitación Anual")
        if not df_anual_melted.empty:
            fig_mapa_animado = px.scatter_geo(df_anual_melted, lat='Latitud_geo', lon='Longitud_geo', color='Precipitación', size='Precipitación',
                                             hover_name='Nom_Est', animation_frame='año', projection='natural earth',
                                             title='Precipitación Anual por Estación', color_continuous_scale=px.colors.sequential.YlGnBu_r)
            fig_mapa_animado.update_geos(fitbounds="locations", visible=True)
            fig_mapa_animado.update_layout(height=700)
            st.plotly_chart(fig_mapa_animado, use_container_width=True)
            
    with st.expander("Ver Comparación de Mapas anuales & Kriging", expanded=True):
        if not df_anual_melted.empty and len(df_anual_melted['año'].unique()) > 0:
            
            m1, m2 = st.columns([1,3])
            with m1:
                if os.path.exists(logo_gota_path):
                    st.image(logo_gota_path, width=40)
            with m2:
                st.metric("Estaciones para Análisis", len(df_anual_melted['Nom_Est'].unique()))

            st.sidebar.markdown("### Opciones de Mapa Comparativo")
            min_precip, max_precip = int(df_anual_melted['Precipitación'].min()), int(df_anual_melted['Precipitación'].max())
            color_range = st.sidebar.slider("Rango de Escala de Color (mm)", min_precip, max_precip, (min_precip, max_precip))
            
            col1, col2 = st.columns(2)
            min_year, max_year = int(df_anual_melted['año'].min()), int(df_anual_melted['año'].max())
            
            year1 = col1.slider("Seleccione el año para el Mapa 1", min_year, max_year, max_year)
            year2 = col2.slider("Seleccione el año para el Mapa 2", min_year, max_year, max_year - 1 if max_year > min_year else max_year)

            if st.button("Generar Mapas de Comparación"):
                if year1 == year2:
                    with st.expander("Superficies de lluvia (Kriging)", expanded=True):
                        st.info("Años iguales: Mapa 1 muestra Puntos, Mapa 2 muestra Superficie Kriging.")
                        map_col1, map_col2 = st.columns(2)
                        data_year = df_anual_melted.loc[df_anual_melted['año'].astype(int) == year1]
                        
                        if len(data_year) < 3:
                            st.warning(f"Se necesitan al menos 3 estaciones para generar el mapa Kriging del año {year1}.")
                        else:
                            gdf_data_year = gpd.GeoDataFrame(
                                data_year, 
                                geometry=gpd.points_from_xy(data_year['Longitud_geo'], data_year['Latitud_geo']),
                                crs="EPSG:4326"
                            )
                            bounds = gdf_data_year.total_bounds
                            lon_range = [bounds[0] - 0.1, bounds[2] + 0.1]
                            lat_range = [bounds[1] - 0.1, bounds[3] + 0.1]
                            
                            with map_col1:
                                st.subheader(f"Estaciones - Año: {year1}")
                                fig1 = px.scatter_geo(data_year, lat='Latitud_geo', lon='Longitud_geo', color='Precipitación', 
                                                      size='Precipitación', hover_name='Nom_Est', color_continuous_scale=px.colors.sequential.YlGnBu, 
                                                      projection='natural earth', range_color=color_range)
                                fig1.update_geos(lonaxis_range=lon_range, lataxis_range=lat_range, visible=True, showcoastlines=True)
                                fig1.update_layout(height=600)
                                st.plotly_chart(fig1, use_container_width=True)

                            with map_col2, st.spinner("Generando mapa Kriging..."):
                                st.subheader(f"Interpolación Kriging - Año: {year1}")
                                lons, lats, vals = data_year['Longitud_geo'].values, data_year['Latitud_geo'].values, data_year['Precipitación'].values
                                grid_lon, grid_lat = np.linspace(lon_range[0], lon_range[1], 100), np.linspace(lat_range[0], lat_range[1], 100)
                                
                                try:
                                    OK = OrdinaryKriging(lons, lats, vals, variogram_model='linear', verbose=False, enable_plotting=False)
                                    z, ss = OK.execute('grid', grid_lon, grid_lat)
                                except Exception as e:
                                     st.warning(f"Error al ejecutar Kriging: {e}. Puede que los datos sean insuficientes para el modelo de variograma.")
                                     z = np.zeros((100, 100)) # Placeholder para evitar errores
                                
                                fig2 = go.Figure(data=go.Contour(
                                    z=z, x=grid_lon, y=grid_lat, colorscale='YlGnBu',
                                    zmin=color_range[0], zmax=color_range[1],
                                    contours=dict(showlabels=True, labelfont=dict(size=12, color='white'))
                                ))
                                fig2.add_trace(go.Scatter(x=lons, y=lats, mode='markers', marker=dict(color='red', size=4), name='Estaciones'))
                                fig2.update_xaxes(range=lon_range, showticklabels=True)
                                fig2.update_yaxes(range=lat_range, scaleanchor="x", scaleratio=1, showticklabels=True)
                                fig2.update_layout(height=600, xaxis_title="Longitud", yaxis_title="Latitud")
                                st.plotly_chart(fig2, use_container_width=True)
                else:
                    with st.expander("Comparación de Mapas de lluvia anual", expanded=True):
                        st.info("Años diferentes: Se comparan los Puntos de Estaciones para cada año.")
                        map_col1, map_col2 = st.columns(2)
                        for i, (col, year) in enumerate(zip([map_col1, map_col2], [year1, year2])):
                            with col:
                                st.subheader(f"Estaciones - Año: {year}")
                                data_year = df_anual_melted.loc[df_anual_melted['año'].astype(int) == year]
                                if data_year.empty:
                                    st.warning(f"No hay datos para el año {year}.")
                                    continue
                                fig = px.scatter_geo(data_year, lat='Latitud_geo', lon='Longitud_geo', color='Precipitación', size='Precipitación',
                                                     hover_name='Nom_Est', color_continuous_scale='YlGnBu', range_color=color_range, projection='natural earth')
                                fig.update_geos(fitbounds="locations", visible=True)
                                st.plotly_chart(fig, use_container_width=True, key=f'map_diff_{i}')
        else:
            st.warning("No hay años disponibles en la selección actual para la comparación.")
            
    with st.expander("Mapa Animado del Fenómeno ENSO"):
        st.subheader("Evolución Mensual del Fenómeno ENSO")
        if df_enso.empty or 'anomalia_oni' not in df_enso.columns:
            st.warning("No se puede generar el Mapa Animado del Fenómeno ENSO. Falta la columna 'anomalia_oni' o no hay datos disponibles.")
        else:
            enso_anim_data = df_enso.loc[(df_enso['Fecha'].dt.year >= year_range[0]) & (df_enso['Fecha'].dt.year <= year_range[1])].copy()
            enso_anim_data.dropna(subset=['anomalia_oni'], inplace=True)
            
            conditions = [enso_anim_data['anomalia_oni'] >= 0.5, enso_anim_data['anomalia_oni'] <= -0.5]
            phases = ['El Niño', 'La Niña']
            enso_anim_data.loc[:, 'Fase'] = np.select(conditions, phases, default='Neutral')
            enso_anim_data.loc[:, 'Fecha_str'] = enso_anim_data['Fecha'].dt.strftime('%Y-%m')

            # Para garantizar que el merge funcione, se añade un key temporal
            stations_subset = gdf_stations.loc[gdf_stations['Nom_Est'].isin(selected_stations), ['Nom_Est', 'Longitud_geo', 'Latitud_geo']].copy()
            
            if not stations_subset.empty:
                enso_anim_data.loc[:, 'key'] = 1
                stations_subset.loc[:, 'key'] = 1
                animation_df = pd.merge(stations_subset, enso_anim_data, on='key', how='inner').drop('key', axis=1)
            else:
                animation_df = pd.DataFrame()

            if not animation_df.empty:
                fig_enso_anim = px.scatter_geo(
                    animation_df,
                    lat='Latitud_geo',
                    lon='Longitud_geo',
                    color='Fase',
                    animation_frame='Fecha_str',
                    hover_name='Nom_Est',
                    color_discrete_map={'El Niño': 'red', 'La Niña': 'blue', 'Neutral': 'lightgrey'},
                    category_orders={"Fase": ["El Niño", "La Niña", "Neutral"]},
                    projection='natural earth'
                )
                fig_enso_anim.update_geos(fitbounds="locations", visible=True)
                fig_enso_anim.update_layout(height=700, title="Fase ENSO por Mes en las Estaciones Seleccionadas")
                st.plotly_chart(fig_enso_anim, use_container_width=True)
            else:
                st.warning("No se puede generar el mapa animado. Verifique que haya estaciones seleccionadas y datos ENSO.")


# --- Estadísticas 📊
with tab_stats:
    st.header("Estadísticas de Precipitación")
    
    st.subheader("Matriz de Disponibilidad de Datos Anual")
    
    original_data_counts = df_long.loc[df_long['Nom_Est'].isin(selected_stations)].copy()
    original_data_counts = original_data_counts.groupby(['Nom_Est', 'año']).size().reset_index(name='count')
    original_data_counts.loc[:, 'porc_original'] = (original_data_counts['count'] / 12) * 100
    heatmap_original_df = original_data_counts.pivot(index='Nom_Est', columns='año', values='porc_original')

    heatmap_df = heatmap_original_df
    color_scale = "Greens"
    title_text = "Porcentaje de Datos Originales (%) por Estación y año"
    
    if analysis_mode == "Completar series (interpolación)":
        view_mode = st.radio("Seleccione la vista de la matriz:", 
                             ("Porcentaje de Datos Originales", "Porcentaje de Datos Completados"), horizontal=True)

        if view_mode == "Porcentaje de Datos Completados":
            completed_data = df_monthly_to_process.loc[
                (df_monthly_to_process['Nom_Est'].isin(selected_stations)) &
                (df_monthly_to_process['Origen'] == 'Completado')
            ].copy()
            if not completed_data.empty:
                completed_counts = completed_data.groupby(['Nom_Est', 'año']).size().reset_index(name='count')
                completed_counts.loc[:, 'porc_completado'] = (completed_counts['count'] / 12) * 100
                heatmap_df = completed_counts.pivot(index='Nom_Est', columns='año', values='porc_completado')
                color_scale = "Reds"
                title_text = "Porcentaje de Datos Completados (%) por Estación y año"
            else:
                heatmap_df = pd.DataFrame()
    
    if not heatmap_df.empty:
        fig_heatmap = px.imshow(
            heatmap_df,
            text_auto='.0f',
            aspect="auto",
            color_continuous_scale=color_scale,
            labels=dict(x="Año", y="Estación", color="% Datos"),
            title=title_text
        )
        fig_heatmap.update_layout(height=max(400, len(selected_stations) * 40))
        st.plotly_chart(fig_heatmap, use_container_width=True)
    else:
        st.info("No hay datos para mostrar en la matriz con la selección actual.")
    
    st.markdown("---")
    if not df_monthly_filtered.empty and not df_anual_melted.empty:
        st.subheader("Síntesis General")
        max_annual_row = df_anual_melted.loc[df_anual_melted['Precipitación'].idxmax()]
        max_monthly_row = df_monthly_filtered.loc[df_monthly_filtered['Precipitation'].idxmax()]
        col1, col2 = st.columns(2)
        with col1:
            st.metric(
                "Máxima Ppt. Anual Registrada",
                f"{max_annual_row['Precipitación']:.1f} mm",
                f"{max_annual_row['Nom_Est']} (Año {max_annual_row['año']})"
            )
        with col2:
            st.metric(
                "Máxima Ppt. Mensual Registrada",
                f"{max_monthly_row['Precipitation']:.1f} mm",
                f"{max_monthly_row['Nom_Est']} ({max_monthly_row['Fecha'].strftime('%Y-%m')})"
            )
        st.markdown("---")
        st.subheader("Resumen de Estadísticas Mensuales por Estación")
        summary_data = []
        for station_name, group in df_monthly_filtered.groupby('Nom_Est'):
            max_row = group.loc[group['Precipitation'].idxmax()]
            min_row = group.loc[group['Precipitation'].idxmin()]
            summary_data.append({
                "Estación": station_name,
                "Ppt. Máxima Mensual (mm)": max_row['Precipitation'],
                "Fecha Máxima": max_row['Fecha'].strftime('%Y-%m'),
                "Ppt. Mínima Mensual (mm)": min_row['Precipitation'],
                "Fecha Mínima": min_row['Fecha'].strftime('%Y-%m'),
                "Promedio Mensual (mm)": group['Precipitation'].mean()
            })
        summary_df = pd.DataFrame(summary_data)
        st.dataframe(summary_df.round(2), use_container_width=True)


# --- Análisis de Anomalías 🔍
with tab_anom:
    st.header("Análisis de Anomalías de Precipitación")
    st.info("Las anomalías se calculan como la diferencia entre la precipitación mensual y el promedio mensual de un período base.")
    
    if not df_monthly_filtered.empty:
        
        st.markdown("#### **Definir Período Base Climatológico**")
        base_year_range = st.slider(
            "Seleccionar Rango de Años Base para el Promedio:", 
            min(años_disponibles), max(años_disponibles), 
            (min(años_disponibles), max(años_disponibles)),
            key="base_years_anom"
        )
        
        df_anomalies = calculate_anomalies(df_monthly_filtered, base_year_range)
        
        if not df_anomalies['Precipitation_long_term_avg'].isnull().all():
            st.markdown("---")
            st.subheader("Visualización de Anomalías de Precipitación")
            
            fig_anom = px.bar(df_anomalies, 
                              x='Fecha', 
                              y='Precipitation_Anomaly', 
                              color='Nom_Est',
                              title=f"Anomalía de Precipitación por Mes (Período base: {base_year_range[0]}-{base_year_range[1]})",
                              labels={'Precipitation_Anomaly': 'Anomalía de Precipitación (mm)'},
                              hover_data=['Precipitation', 'Precipitation_long_term_avg'])
            
            fig_anom.add_hline(y=0, line_dash="dash", line_color="black")
            fig_anom.update_layout(height=600, barmode='group')
            st.plotly_chart(fig_anom, use_container_width=True)
            
            st.markdown("---")
            st.subheader("Tabla de Anomalías de Precipitación")
            df_anomalies_pivot = df_anomalies.pivot_table(index='Fecha', columns='Nom_Est', values='Precipitation_Anomaly').round(2)
            st.dataframe(df_anomalies_pivot, use_container_width=True)
        else:
            st.warning("No se pudieron calcular las anomalías. Verifique que haya datos en el período base seleccionado.")
    else:
        st.info("No hay datos de precipitación mensual para realizar el análisis de anomalías.")


# --- Análisis ENSO 🌡️
with tab_enso:
    st.header("Análisis de Precipitación y el Fenómeno ENSO")
    if df_enso.empty or 'anomalia_oni' not in df_enso.columns:
        st.warning("No se encontraron datos del fenómeno ENSO en el archivo de precipitación cargado. El análisis ENSO no está disponible.")
    else:
        enso_series_tab, enso_corr_tab, enso_precip_combo = st.tabs(["Series de Tiempo ENSO", "Correlación Precipitación-ENSO", "ENSO y Precipitación"])
        
        with enso_series_tab:
            st.subheader("Visualización de Variables ENSO")
            enso_vars_available = [v for v in ['anomalia_oni', 'temp_sst'] if v in df_enso.columns]
            if not enso_vars_available:
                st.warning("No hay variables ENSO disponibles en el archivo de datos para visualizar.")
            else:
                variable_enso = st.selectbox("Seleccione la variable ENSO a visualizar:", enso_vars_available)
                df_enso_filtered = df_enso.loc[
                    (df_enso['Fecha'].dt.year >= year_range[0]) &
                    (df_enso['Fecha'].dt.year <= year_range[1])
                ].copy()
                if not df_enso_filtered.empty and variable_enso in df_enso_filtered.columns and not df_enso_filtered[variable_enso].isnull().all():
                    fig_enso_series = go.Figure(data=go.Scatter(x=df_enso_filtered['Fecha'], y=df_enso_filtered[variable_enso], mode='lines'))
                    fig_enso_series.update_layout(title=f"Serie de Tiempo para {variable_enso}", xaxis_title="Fecha", yaxis_title=variable_enso)
                    st.plotly_chart(fig_enso_series, use_container_width=True)
                else:
                    st.warning(f"No hay datos disponibles para '{variable_enso}' en el período seleccionado.")
        
        with enso_corr_tab:
            df_analisis = df_monthly_filtered.copy()
            df_analisis.loc[:, 'fecha_merge'] = df_analisis['Fecha'].dt.strftime('%Y-%m')
            df_analisis = pd.merge(df_analisis, df_enso, on='fecha_merge', how='left')
            
            if 'anomalia_oni' in df_analisis.columns:
                df_analisis.dropna(subset=['anomalia_oni'], inplace=True)
                
                def classify_enso(oni):
                    if oni >= 0.5: return 'El Niño'
                    elif oni <= -0.5: return 'La Niña'
                    else: return 'Neutral'
                
                df_analisis.loc[:, 'ENSO'] = df_analisis['anomalia_oni'].apply(classify_enso)
                
                if not df_analisis.empty:
                    st.subheader("Precipitación Media por Evento ENSO")
                    df_enso_group = df_analisis.groupby('ENSO')['Precipitation'].mean().reset_index()
                    fig_enso = px.bar(df_enso_group, x='ENSO', y='Precipitation', color='ENSO', labels={'Precipitation': 'Precipitación Media (mm)'})
                    st.plotly_chart(fig_enso, use_container_width=True)
                    
                    st.subheader("Correlación entre Anomalía ONI y Precipitación")
                    if df_analisis['anomalia_oni'].nunique() > 1 and df_analisis['Precipitation'].nunique() > 1:
                        correlation = df_analisis['anomalia_oni'].corr(df_analisis['Precipitation'])
                        st.metric("Coeficiente de Correlación de Pearson", f"{correlation:.2f}")
                    else:
                        st.warning("No hay suficientes datos variados para calcular la correlación.")
                else:
                    st.warning("No hay datos suficientes para realizar el análisis ENSO con la selección actual.")
            else:
                st.warning(f"Análisis no disponible. Falta la columna 'anomalia_oni' en el archivo de datos.")

        # --- GRÁFICO COMBINADO PRECIPITACIÓN-ENSO ---
        with enso_precip_combo:
            st.subheader("Serie de Tiempo: Precipitación y Anomalía ONI")
            st.info("Este gráfico combina la precipitación mensual (calculada como promedio para las estaciones seleccionadas) y la anomalía ONI.")
            
            df_combined = df_monthly_filtered.copy()
            df_combined = df_combined.groupby('Fecha', as_index=False)['Precipitation'].mean()
            df_combined.loc[:, 'fecha_merge'] = df_combined['Fecha'].dt.strftime('%Y-%m')
            df_combined = pd.merge(df_combined, df_enso, on='fecha_merge', how='left')
            df_combined.dropna(subset=['Precipitation', 'anomalia_oni'], inplace=True)
            
            if not df_combined.empty:
                fig_combined = go.Figure()
                
                # Gráfico de barras de precipitación
                fig_combined.add_trace(go.Bar(
                    x=df_combined['Fecha'],
                    y=df_combined['Precipitation'],
                    name='Precipitación Media (mm)',
                    marker_color='lightblue',
                    yaxis='y1'
                ))
                
                # Gráfico de línea de ONI
                fig_combined.add_trace(go.Scatter(
                    x=df_combined['Fecha'],
                    y=df_combined['anomalia_oni'],
                    mode='lines',
                    name='Anomalía ONI (°C)',
                    line=dict(color='black', width=2),
                    yaxis='y2'
                ))
                
                fig_combined.update_layout(
                    title="Precipitación Media Mensual y Anomalía ONI",
                    yaxis=dict(title='Precipitación (mm)', showgrid=False),
                    yaxis2=dict(title='Anomalía ONI (°C)', overlaying='y', side='right'),
                    legend=dict(x=0.01, y=0.99),
                    height=600
                )
                
                fig_combined.add_hline(y=0.5, line_dash="dash", line_color="red", annotation_text="El Niño", yaxis='y2', annotation_position="bottom right")
                fig_combined.add_hline(y=-0.5, line_dash="dash", line_color="blue", annotation_text="La Niña", yaxis='y2', annotation_position="top right")
                
                st.plotly_chart(fig_combined, use_container_width=True)
            else:
                st.warning("No hay datos de precipitación y anomalía ONI coincidentes en el período seleccionado.")

# --- Descargas 📥
with tab_descargas:
    st.header("Opciones de Descarga")
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
        if not df_monthly_to_process.empty:
            csv_completado = convert_df_to_csv(df_monthly_to_process.loc[df_monthly_to_process['Nom_Est'].isin(selected_stations)])
            st.download_button("Descargar CSV con Series Completadas", csv_completado, 'precipitacion_mensual_completada.csv', 'text/csv', key='download-completado')
        else:
            st.info("El DataFrame con series completadas está vacío.")
    else:
        st.info("Para descargar las series completadas, seleccione la opción 'Completar series (interpolación)' en el panel lateral.")
