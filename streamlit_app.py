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
            if gdf.crs is None:
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
        df_station = _df[_df['Nom_Est'] == station].copy()
        df_station['Fecha'] = pd.to_datetime(df_station['Fecha'])
        df_station.set_index('Fecha', inplace=True)
        if not df_station.index.is_unique:
            df_station = df_station[~df_station.index.duplicated(keep='first')]
        
        original_data = df_station[['Precipitation', 'Origen']].copy()
        df_resampled = df_station.resample('MS').asfreq()
        
        df_resampled['Precipitation'] = original_data['Precipitation']
        df_resampled['Origen'] = original_data['Origen']
        
        df_resampled['Origen'] = df_resampled['Origen'].fillna('Completado')
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

    data = enso_data.copy().sort_values('Fecha')
    
    conditions = [data['anomalia_oni'] >= 0.5, data['anomalia_oni'] <= -0.5]
    phases = ['El Niño', 'La Niña']
    colors = ['red', 'blue']
    data['phase'] = np.select(conditions, phases, default='Neutral')
    data['color'] = np.select(conditions, colors, default='grey')

    y_range = [data['anomalia_oni'].min() - 0.5, data['anomalia_oni'].max() + 0.5]

    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        x=data['Fecha'],
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
        x=data['Fecha'], y=data['anomalia_oni'],
        mode='lines', name='Anomalía ONI',
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

# --- Interfaz y Carga de Archivos ---
logo_gota_path = "CuencaVerdeGoticaLogo.JPG"

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
df_long['Origen'] = 'Original'

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
if 'Porc_datos' in gdf_stations.columns:
    gdf_stations['Porc_datos'] = pd.to_numeric(gdf_stations['Porc_datos'], errors='coerce').fillna(0)
    min_data_perc = st.sidebar.slider("Filtrar por % de datos mínimo:", 0, 100, 0)
    stations_master_list = gdf_stations[gdf_stations['Porc_datos'] >= min_data_perc]
else:
    st.sidebar.text("Advertencia: Columna 'Porc_datos' no encontrada.")
    stations_master_list = gdf_stations.copy()

municipios_list = sorted(stations_master_list['municipio'].unique())
celdas_list = sorted(stations_master_list['Celda_XY'].unique())
selected_municipios = st.sidebar.multiselect('1. Filtrar por Municipio', options=municipios_list)
selected_celdas = st.sidebar.multiselect('2. Filtrar por Celda_XY', options=celdas_list)
stations_available = stations_master_list.copy()
if selected_municipios:
    stations_available = stations_available[stations_available['municipio'].isin(selected_municipios)]
if selected_celdas:
    stations_available = stations_available[stations_available['Celda_XY'].isin(selected_celdas)]
stations_options = sorted(stations_available['Nom_Est'].unique())

select_all = st.sidebar.checkbox("Seleccionar/Deseleccionar Todas las Estaciones", value=True)
if select_all:
    default_selection = stations_options
else:
    default_selection = []
        
selected_stations = st.sidebar.multiselect('3. Seleccionar Estaciones', options=stations_options, default=default_selection)

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

df_monthly_to_process = df_long.copy()
if analysis_mode == "Completar series (interpolación)":
    with st.sidebar:
        with st.spinner("Completando series..."):
            df_monthly_to_process = complete_series(df_long)

if not selected_stations or not meses_numeros:
    st.warning("Por favor, seleccione al menos una estación y un mes.")
    st.stop()

# --- Preparación de datos filtrados ---
df_anual_melted = gdf_stations[gdf_stations['Nom_Est'].isin(selected_stations)].melt(
    id_vars=['Nom_Est', 'Longitud_geo', 'Latitud_geo'],
    value_vars=[str(y) for y in range(year_range[0], year_range[1] + 1) if str(y) in gdf_stations.columns],
    var_name='Año', value_name='Precipitación')
df_monthly_filtered = df_monthly_to_process[
    (df_monthly_to_process['Nom_Est'].isin(selected_stations)) &
    (df_monthly_to_process['Fecha'].dt.year >= year_range[0]) &
    (df_monthly_to_process['Fecha'].dt.year <= year_range[1]) &
    (df_monthly_to_process['Fecha'].dt.month.isin(meses_numeros))
]

# --- Pestañas Principales ---
tab1, tab2, tab_anim, tab3, tab_stats, tab4, tab5 = st.tabs(["Gráficos", "Mapa de Estaciones", "Mapas Avanzados", "Tabla de Estaciones", "Estadísticas", "Análisis ENSO", "Descargas"])

with tab1:
    st.header("Visualizaciones de Precipitación")
    sub_tab_anual, sub_tab_mensual, sub_tab_box = st.tabs(["Serie Anual", "Serie Mensual", "Box Plot Anual"])
    
    with sub_tab_anual:
        if not df_anual_melted.empty:
            st.subheader("Precipitación Anual (mm)")
            chart_anual = alt.Chart(df_anual_melted).mark_line(point=True).encode(x=alt.X('Año:O', title='Año'), y=alt.Y('Precipitación:Q', title='Precipitación (mm)'), color='Nom_Est:N', tooltip=['Nom_Est', 'Año', 'Precipitación']).interactive()
            st.altair_chart(chart_anual, use_container_width=True)

    with sub_tab_mensual:
        if not df_monthly_filtered.empty:
            st.subheader("Precipitación Mensual (mm)")
            chart_mensual = alt.Chart(df_monthly_filtered).mark_line().encode(x=alt.X('Fecha:T', title='Fecha'), y=alt.Y('Precipitation:Q', title='Precipitación (mm)'), color='Nom_Est:N', tooltip=[alt.Tooltip('Fecha', format='%Y-%m'), 'Precipitation', 'Nom_Est']).interactive()
            st.altair_chart(chart_mensual, use_container_width=True)

    with sub_tab_box:
        if not df_anual_melted.empty:
            st.subheader("Distribución de la Precipitación Anual por Estación")
            fig_box = px.box(df_anual_melted, x='Año', y='Precipitación', color='Nom_Est', title='Distribución Anual por Estación', labels={"Año": "Año", "Precipitación": "Precipitación Anual (mm)"})
            st.plotly_chart(fig_box, use_container_width=True)


with tab2:
    st.header("Mapa de Ubicación de Estaciones")
    gdf_filtered = gdf_stations[gdf_stations['Nom_Est'].isin(selected_stations)]
    if not gdf_filtered.empty:
        map_center_lat = gdf_filtered['Latitud_geo'].mean()
        map_center_lon = gdf_filtered['Longitud_geo'].mean()
        m = folium.Map(location=[map_center_lat, map_center_lon], tiles="cartodbpositron")
        bounds = gdf_filtered.total_bounds
        m.fit_bounds([[bounds[1], bounds[0]], [bounds[3], bounds[2]]])
        folium.GeoJson(gdf_municipios.to_json(), name='Municipios').add_to(m)
        for _, row in gdf_filtered.iterrows():
            html = f"<b>Estación:</b> {row['Nom_Est']}<br><b>Municipio:</b> {row['municipio']}"
            folium.Marker([row['Latitud_geo'], row['Longitud_geo']], tooltip=html).add_to(m)
        folium_static(m, width=900, height=600)
    else:
        st.warning("No hay estaciones seleccionadas para mostrar en el mapa.")

with tab_anim:
    st.header("Mapas Avanzados")
    with st.expander("Ver Animación de Puntos", expanded=True):
        st.subheader("Mapa Animado de Precipitación Anual")
        if not df_anual_melted.empty:
            fig_mapa_animado = px.scatter_geo(df_anual_melted, lat='Latitud_geo', lon='Longitud_geo', color='Precipitación', size='Precipitación', hover_name='Nom_Est', animation_frame='Año', projection='natural earth', title='Precipitación Anual por Estación', color_continuous_scale=px.colors.sequential.YlGnBu)
            fig_mapa_animado.update_geos(fitbounds="locations", visible=True)
            fig_mapa_animado.update_layout(height=700)
            st.plotly_chart(fig_mapa_animado, use_container_width=True)
    with st.expander("Ver Comparación de Mapas anuales & Kriging", expanded=True):
        st.subheader("Comparación de Mapas de Precipitación Anual (Kriging)")
        if not df_anual_melted.empty and len(df_anual_melted['Año'].unique()) > 0:
            min_year, max_year = int(df_anual_melted['Año'].min()), int(df_anual_melted['Año'].max())
            col1, col2 = st.columns(2)
            year1 = col1.slider("Seleccione el año para el Mapa 1", min_year, max_year, max_year)
            year2 = col2.slider("Seleccione el año para el Mapa 2", min_year, max_year, max_year - 1 if max_year > min_year else max_year)

            if st.button("Generar Mapas de Comparación"):
                if year1 == year2:
                    with st.expander("Superficies de lluvia (Kriging)", expanded=True):
                        # ... (código Kriging) ...
                        pass
                else:
                    with st.expander("Comparación de Mapas de lluvia anual", expanded=True):
                        # ... (código comparación) ...
                        pass
        else:
            st.warning("No hay años disponibles en la selección actual para la comparación.")

with tab3:
    st.header("Información Detallada de las Estaciones")
    if not df_anual_melted.empty:
        display_cols = [col for col in gdf_stations.columns if col != 'geometry']
        df_info_table = gdf_stations[display_cols]
        df_mean_precip = df_anual_melted.groupby('Nom_Est')['Precipitación'].mean().round(2).reset_index()
        df_mean_precip.rename(columns={'Precipitación': 'Precipitación media anual (mm)'}, inplace=True)
        df_info_table = df_info_table.merge(df_mean_precip, on='Nom_Est', how='left')
        st.dataframe(df_info_table[df_info_table['Nom_Est'].isin(selected_stations)])
    else:
        st.info("No hay datos de precipitación anual para mostrar en la selección actual.")

with tab_stats:
    st.header("Estadísticas de Precipitación")
    st.subheader("Matriz de Disponibilidad de Datos Anual")
    
    original_data_counts = df_long[df_long['Nom_Est'].isin(selected_stations)]
    original_data_counts = original_data_counts.groupby(['Nom_Est', 'Año']).size().reset_index(name='count')
    original_data_counts['porc_original'] = (original_data_counts['count'] / 12) * 100
    heatmap_original_df = original_data_counts.pivot(index='Nom_Est', columns='Año', values='porc_original')

    heatmap_df = heatmap_original_df
    color_scale = "Greens"
    title_text = "Porcentaje de Datos Originales (%) por Estación y Año"
    
    if analysis_mode == "Completar series (interpolación)":
        view_mode = st.radio("Seleccione la vista de la matriz:", 
                             ("Porcentaje de Datos Originales", "Porcentaje de Datos Completados"), horizontal=True)

        if view_mode == "Porcentaje de Datos Completados":
            completed_data = df_monthly_to_process[
                (df_monthly_to_process['Nom_Est'].isin(selected_stations)) &
                (df_monthly_to_process['Origen'] == 'Completado')
            ]
            if not completed_data.empty:
                completed_counts = completed_data.groupby(['Nom_Est', 'Año']).size().reset_index(name='count')
                completed_counts['porc_completado'] = (completed_counts['count'] / 12) * 100
                heatmap_df = completed_counts.pivot(index='Nom_Est', columns='Año', values='porc_completado')
                color_scale = "Reds"
                title_text = "Porcentaje de Datos Completados (%) por Estación y Año"
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
                f"{max_annual_row['Nom_Est']} (Año {max_annual_row['Año']})"
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

with tab4:
    st.header("Análisis de Precipitación y el Fenómeno ENSO")
    if df_enso.empty:
        st.warning("No se encontraron datos del fenómeno ENSO en el archivo de precipitación cargado. El análisis ENSO no está disponible.")
    else:
        enso_series_tab, enso_corr_tab, enso_anim_tab = st.tabs(["Series de Tiempo ENSO", "Correlación Precipitación-ENSO", "Mapa Animado ENSO"])
        
        with enso_series_tab:
            st.subheader("Visualización de Variables ENSO")
            enso_vars_available = [v for v in ['anomalia_oni', 'temp_sst', 'temp_media'] if v in df_enso.columns]
            if not enso_vars_available:
                st.warning("No hay variables ENSO disponibles en el archivo de datos para visualizar.")
            else:
                variable_enso = st.selectbox("Seleccione la variable ENSO a visualizar:", enso_vars_available)
                df_enso_filtered = df_enso[
                    (df_enso['Fecha'].dt.year >= year_range[0]) &
                    (df_enso['Fecha'].dt.year <= year_range[1]) &
                    (df_enso['Fecha'].dt.month.isin(meses_numeros))]
                
                if len(df_enso_filtered) > 1:
                    fig_enso_series = px.scatter(df_enso_filtered, x='Fecha', y=variable_enso, 
                                            title=f"Serie de Tiempo para {variable_enso}",
                                            trendline="ols", trendline_color_override="red")
                    fig_enso_series.add_trace(go.Scatter(x=df_enso_filtered['Fecha'], y=df_enso_filtered[variable_enso], mode='lines', name=variable_enso, line=dict(color='blue')))
                    st.plotly_chart(fig_enso_series, use_container_width=True)
                    
                    try:
                        results = px.get_trendline_results(fig_enso_series)
                        model = results.iloc[0]["px_fit_results"]
                        slope = model.params[1]
                        r_squared = model.rsquared
                        with st.expander("Ver Ecuación de la Línea de Tendencia"):
                            st.write(f"**Pendiente (cambio por día):** `{slope:.6f}`")
                            st.write(f"**Coeficiente de Determinación (R²):** `{r_squared:.4f}`")
                    except Exception as e:
                        st.warning(f"No se pudo calcular la línea de tendencia.")

                elif not df_enso_filtered.empty:
                    fig_enso_series = px.line(df_enso_filtered, x='Fecha', y=variable_enso, 
                                            title=f"Serie de Tiempo para {variable_enso}")
                    st.plotly_chart(fig_enso_series, use_container_width=True)
                else:
                    st.warning(f"No hay datos disponibles para '{variable_enso}' en el período seleccionado.")
        
        with enso_corr_tab:
            df_analisis = df_monthly_filtered.copy()
            df_analisis['fecha_merge'] = df_analisis['Fecha'].dt.strftime('%Y-%m')
            df_analisis = pd.merge(df_analisis, df_enso, on='fecha_merge', how='left')
            
            if 'anomalia_oni' in df_analisis.columns:
                df_analisis.dropna(subset=['anomalia_oni'], inplace=True)

                def classify_enso(oni):
                    if oni >= 0.5: return 'El Niño'
                    elif oni <= -0.5: return 'La Niña'
                    else: return 'Neutral'
                
                df_analisis['ENSO'] = df_analisis['anomalia_oni'].apply(classify_enso)
                
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

        with enso_anim_tab:
            st.subheader("Evolución Mensual del Fenómeno ENSO")
            if not df_enso.empty and not gdf_stations.empty:
                st.info("El color de cada estación representa la fase del fenómeno ENSO a nivel global para cada mes.")
                
                stations_subset = gdf_stations[gdf_stations['Nom_Est'].isin(selected_stations)][['Nom_Est', 'Latitud_geo', 'Longitud_geo']]
                
                if not stations_subset.empty:
                    enso_anim_data = df_enso[['Fecha', 'anomalia_oni']].copy()
                    enso_anim_data.dropna(subset=['anomalia_oni'], inplace=True)
                    
                    conditions = [enso_anim_data['anomalia_oni'] >= 0.5, enso_anim_data['anomalia_oni'] <= -0.5]
                    phases = ['El Niño', 'La Niña']
                    enso_anim_data['Fase'] = np.select(conditions, phases, default='Neutral')
                    enso_anim_data['FrameLabel'] = enso_anim_data['Fecha'].dt.strftime('%Y-%m') + ' - ' + enso_anim_data['Fase']

                    enso_anim_data = enso_anim_data[
                        (enso_anim_data['Fecha'].dt.year >= year_range[0]) &
                        (enso_anim_data['Fecha'].dt.year <= year_range[1])
                    ]

                    enso_anim_data['key'] = 1
                    stations_subset['key'] = 1

                    animation_df = pd.merge(stations_subset, enso_anim_data, on='key').drop('key', axis=1)
                    
                    fig_enso_anim = px.scatter_geo(
                        animation_df,
                        lat='Latitud_geo',
                        lon='Longitud_geo',
                        color='Fase',
                        animation_frame='FrameLabel',
                        hover_name='Nom_Est',
                        color_discrete_map={'El Niño': 'red', 'La Niña': 'blue', 'Neutral': 'lightgrey'},
                        category_orders={"Fase": ["El Niño", "La Niña", "Neutral"]},
                        projection='natural earth'
                    )
                    fig_enso_anim.update_geos(fitbounds="locations", visible=True)
                    fig_enso_anim.update_layout(height=700, title="Fase ENSO por Mes en las Estaciones Seleccionadas")
                    st.plotly_chart(fig_enso_anim, use_container_width=True)

with tab5:
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
        st.download_button("Descargar CSV con Series Completadas", csv_mensual, 'precipitacion_mensual_completada.csv', 'text/csv', key='download-completado')
    else:
        st.info("Para descargar las series completadas, seleccione la opción 'Completar series (interpolación)' en el panel lateral.")
