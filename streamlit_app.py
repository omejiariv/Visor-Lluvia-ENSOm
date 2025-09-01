# --- Importaciones ---
import streamlit as st
import pandas as pd
import altair as alt
import folium
from streamlit_folium import folium_static
import plotly.express as px
import plotly.graph_objects as go
import geopandas as gpd
from pykrige.ok import OrdinaryKriging
import numpy as np
import zipfile
import tempfile
import os
import io

# --- Configuración de la Página ---
st.set_page_config(
    layout="wide",
    page_title="Visor de Precipitación y ENSO",
    page_icon="💧"
)

# --- Estilos CSS Personalizados ---
st.markdown("""
<style>
    /* Reduce el padding superior para un contenido más compacto */
    div.block-container {padding-top: 2rem;}
    /* Títulos y métricas ajustadas para mejor visualización */
    h1, h2, h3 { margin-top: 0px !important; padding-top: 0px !important; }
    [data-testid="stMetricValue"] { font-size: 1.8rem; }
    [data-testid="stMetricLabel"] { font-size: 1rem; padding-bottom: 5px; }
</style>
""", unsafe_allow_html=True)


# --- Funciones de Carga y Preprocesamiento de Datos ---
@st.cache_data
def load_csv_data(file_uploader):
    """Carga un archivo CSV desde Streamlit File Uploader con manejo de errores y codificaciones."""
    if file_uploader is None:
        return None
    try:
        content = file_uploader.getvalue()
        if not content.strip():
            st.error(f"El archivo '{file_uploader.name}' parece estar vacío.")
            return None
        for encoding in ['utf-8', 'latin1', 'cp1252']:
            try:
                df = pd.read_csv(io.BytesIO(content), sep=';', encoding=encoding)
                df.columns = df.columns.str.strip().str.lower()
                return df
            except Exception:
                continue
        st.error(f"No se pudo decodificar el archivo '{file_uploader.name}'. Por favor, verifique que sea un CSV con codificación UTF-8 o Latin-1.")
        return None
    except Exception as e:
        st.error(f"Error al leer el archivo '{file_uploader.name}': {e}")
        return None

@st.cache_data
def load_shapefile(zip_file):
    """Carga un shapefile desde un archivo .zip subido a Streamlit."""
    if zip_file is None:
        return None
    try:
        with tempfile.TemporaryDirectory() as temp_dir:
            with zipfile.ZipFile(zip_file, 'r') as zip_ref:
                zip_ref.extractall(temp_dir)
            
            shp_files = [f for f in os.listdir(temp_dir) if f.endswith('.shp')]
            if not shp_files:
                st.error("No se encontró un archivo .shp en el .zip.")
                return None
            
            gdf = gpd.read_file(os.path.join(temp_dir, shp_files[0]))
            gdf.columns = gdf.columns.str.strip().str.lower()
            
            if gdf.crs is None:
                gdf = gdf.set_crs("EPSG:4326")
            if gdf.crs.to_epsg() != 4326:
                gdf = gdf.to_crs("EPSG:4326")
            return gdf
    except Exception as e:
        st.error(f"Error al procesar el shapefile: {e}")
        return None

def parse_spanish_dates(date_series):
    """Convierte fechas abreviadas en español a un formato estándar."""
    months_map = {'ene': 'Jan', 'abr': 'Apr', 'ago': 'Aug', 'dic': 'Dec'}
    for es, en in months_map.items():
        date_series = date_series.str.replace(es, en, regex=False, case=False)
    return pd.to_datetime(date_series, format='%b-%y', errors='coerce')

@st.cache_data
def preprocess_data(_df_stations_raw, _df_precip_raw, _gdf_municipios):
    """Función unificada para preprocesar todos los datos cargados."""
    if any(df is None for df in [_df_stations_raw, _df_precip_raw, _gdf_municipios]):
        return None, None, None

    # 1. Preprocesar datos de estaciones (geometría)
    df_stations = _df_stations_raw.copy()
    lon_col = next((c for c in df_stations.columns if 'lon' in c), None)
    lat_col = next((c for c in df_stations.columns if 'lat' in c), None)
    if not lon_col or not lat_col:
        st.error("Columnas de latitud/longitud no encontradas en el archivo de estaciones.")
        return None, None, None
        
    df_stations[lon_col] = pd.to_numeric(df_stations[lon_col].astype(str).str.replace(',', '.'), errors='coerce')
    df_stations[lat_col] = pd.to_numeric(df_stations[lat_col].astype(str).str.replace(',', '.'), errors='coerce')
    df_stations.dropna(subset=[lon_col, lat_col], inplace=True)
    
    gdf_stations = gpd.GeoDataFrame(
        df_stations, 
        geometry=gpd.points_from_xy(df_stations[lon_col], df_stations[lat_col]), 
        crs="EPSG:4326"
    )
    gdf_stations['longitud_geo'] = gdf_stations.geometry.x
    gdf_stations['latitud_geo'] = gdf_stations.geometry.y

    # 2. Preprocesar datos de precipitación mensual y ENSO
    df_precip = _df_precip_raw.copy()
    station_cols = [col for col in df_precip.columns if col.isdigit()]
    if not station_cols:
        st.error("No se encontraron columnas de estación (ej: '12345') en el archivo de precipitación.")
        return None, None, None

    id_vars = [col for col in ['id', 'fecha_mes_año', 'año', 'mes', 'anomalia_oni', 'temp_sst'] if col in df_precip.columns]
    
    numeric_cols = ['anomalia_oni', 'temp_sst'] + station_cols
    for col in numeric_cols:
        if col in df_precip.columns:
            df_precip[col] = pd.to_numeric(df_precip[col].astype(str).str.replace(',', '.'), errors='coerce')
    
    df_long = df_precip.melt(id_vars=id_vars, value_vars=station_cols, var_name='id_estacion', value_name='precipitation')
    df_long.dropna(subset=['precipitation'], inplace=True)
    
    df_long['fecha_mes_año'] = parse_spanish_dates(df_long['fecha_mes_año'])
    df_long.dropna(subset=['fecha_mes_año'], inplace=True)
    df_long['origen'] = 'Original'
    
    station_mapping = gdf_stations.set_index(gdf_stations['id_estacio'].astype(str).str.strip())['nom_est'].to_dict()
    df_long['nom_est'] = df_long['id_estacion'].astype(str).str.strip().map(station_mapping)
    df_long.dropna(subset=['nom_est'], inplace=True)

    return gdf_stations, df_long, _gdf_municipios

@st.cache_data
def complete_time_series(_df_long):
    """Rellena los datos faltantes en las series de tiempo de precipitación para cada estación."""
    completed_dfs = []
    station_list = _df_long['nom_est'].unique()
    progress_bar = st.progress(0, text="Completando series de tiempo...")

    for i, station in enumerate(station_list):
        df_station = _df_long[_df_long['nom_est'] == station].copy()
        df_station.set_index('fecha_mes_año', inplace=True)
        
        if not df_station.index.is_unique:
            df_station = df_station[~df_station.index.duplicated(keep='first')]

        date_range = pd.date_range(start=df_station.index.min(), end=df_station.index.max(), freq='MS')
        df_resampled = df_station.reindex(date_range)
        
        df_resampled['origen'] = np.where(df_resampled['precipitation'].isna(), 'Completado', 'Original')
        df_resampled['precipitation'] = df_resampled['precipitation'].interpolate(method='time')
        df_resampled['nom_est'] = station
        df_resampled['anomalia_oni'] = df_resampled['anomalia_oni'].ffill().bfill()
        
        completed_dfs.append(df_resampled.reset_index().rename(columns={'index': 'fecha_mes_año'}))
        progress_bar.progress((i + 1) / len(station_list), text=f"Procesando: {station}")
    
    progress_bar.empty()
    return pd.concat(completed_dfs, ignore_index=True)


# --- Funciones de Visualización ---
def create_enso_chart(enso_data):
    """Crea un gráfico de Plotly para visualizar las fases del ENSO y la anomalía ONI."""
    if enso_data.empty or 'anomalia_oni' not in enso_data.columns:
        return go.Figure()

    data = enso_data.copy().sort_values('fecha_mes_año').dropna(subset=['anomalia_oni'])
    conditions = [data['anomalia_oni'] >= 0.5, data['anomalia_oni'] <= -0.5]
    data['phase_color'] = np.select(conditions, ['red', 'blue'], default='lightgrey')
    y_range = [data['anomalia_oni'].min() - 0.5, data['anomalia_oni'].max() + 0.5]
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=data['fecha_mes_año'], y=[y_range[1] - y_range[0]] * len(data),
        base=y_range[0], marker_color=data['phase_color'],
        width=30*24*60*60*1000, opacity=0.3, hoverinfo='none', showlegend=False
    ))
    fig.add_trace(go.Scatter(
        x=data['fecha_mes_año'], y=data['anomalia_oni'], mode='lines',
        name='Anomalía ONI', line=dict(color='black', width=2)
    ))
    fig.add_hline(y=0.5, line_dash="dash", line_color="red", annotation_text="Umbral El Niño")
    fig.add_hline(y=-0.5, line_dash="dash", line_color="blue", annotation_text="Umbral La Niña")
    legend_items = {'El Niño': 'red', 'La Niña': 'blue', 'Neutral': 'lightgrey'}
    for name, color in legend_items.items():
        fig.add_trace(go.Scatter(
            x=[None], y=[None], mode='markers', name=name,
            marker=dict(size=10, color=color, symbol='square', opacity=0.5)
        ))
    fig.update_layout(
        height=500, title_text="<b>Fases del Fenómeno ENSO y Anomalía ONI</b>",
        yaxis_title="Anomalía ONI (°C)", xaxis_title="Fecha", legend_title_text='Fase',
        yaxis_range=y_range, margin=dict(t=50, b=10)
    )
    return fig

def generate_kriging_map(_df_year, color_range):
    """Genera un mapa de contorno con interpolación Kriging."""
    if len(_df_year) < 3:
        st.warning(f"Se necesitan al menos 3 estaciones con datos para generar el mapa Kriging.")
        return go.Figure()

    lons, lats, vals = _df_year['longitud_geo'].values, _df_year['latitud_geo'].values, _df_year['precipitacion'].values
    bounds = _df_year.total_bounds
    grid_lon = np.linspace(bounds[0] - 0.1, bounds[2] + 0.1, 100)
    grid_lat = np.linspace(bounds[1] - 0.1, bounds[3] + 0.1, 100)
    try:
        OK = OrdinaryKriging(lons, lats, vals, variogram_model='linear', verbose=False, enable_plotting=False)
        z, ss = OK.execute('grid', grid_lon, grid_lat)
        fig = go.Figure()
        fig.add_trace(go.Contour(
            z=z.data, x=grid_lon, y=grid_lat, colorscale='YlGnBu',
            zmin=color_range[0], zmax=color_range[1],
            contours=dict(showlabels=True, labelfont=dict(size=12, color='white')),
            colorbar=dict(title='Pptn (mm)')
        ))
        fig.add_trace(go.Scatter(
            x=lons, y=lats, text=_df_year['nom_est'],
            mode='markers', marker=dict(color='red', size=5, symbol='circle'), 
            name='Estaciones', hoverinfo='text'
        ))
        fig.update_layout(
            height=600, xaxis_title="Longitud", yaxis_title="Latitud",
            margin=dict(t=10, b=10, l=10, r=10)
        )
        return fig
    except Exception as e:
        st.error(f"Error durante la interpolación Kriging: {e}")
        return go.Figure()

# --- Interfaz Principal de la Aplicación ---

## CAMBIO 1: Se eliminó el logo del encabezado principal.
st.title('Visor Interactivo de Precipitación y ENSO')
st.markdown("Herramienta para el análisis de datos de lluvia y su relación con el fenómeno ENSO.")

# --- Barra Lateral (Panel de Control) ---

## CAMBIO 2: Se reemplazó 'use_column_width' por 'use_container_width' para corregir la advertencia.
## La imagen de la gotica se mantiene aquí.
st.sidebar.image("https://i.imgur.com/kdkE3b5.png", use_container_width=True)
st.sidebar.title("Panel de Control 🕹️")

with st.sidebar.expander("**1. Cargar Archivos**", expanded=True):
    uploaded_file_stations = st.file_uploader("Archivo de estaciones (CSV)", type="csv")
    uploaded_file_precip = st.file_uploader("Archivo de precipitación y ENSO (CSV)", type="csv")
    uploaded_file_shape = st.file_uploader("Shapefile de municipios (.zip)", type="zip")

# --- Lógica de Carga y Bienvenida ---
if not all([uploaded_file_stations, uploaded_file_precip, uploaded_file_shape]):
    st.info("👋 ¡Bienvenido! Por favor, carga los 3 archivos requeridos en el panel de la izquierda para comenzar.")
    st.image("https://i.imgur.com/qE4J34b.gif", caption="Sube tus archivos para activar el visor.")
    st.stop()

# Cargar y preprocesar los datos una sola vez
if 'data_loaded' not in st.session_state or not st.session_state.data_loaded:
    with st.spinner("Procesando archivos, por favor espera..."):
        gdf_stations, df_long, gdf_municipios = preprocess_data(
            load_csv_data(uploaded_file_stations),
            load_csv_data(uploaded_file_precip),
            load_shapefile(uploaded_file_shape)
        )
        if gdf_stations is not None and df_long is not None:
            st.session_state.gdf_stations = gdf_stations
            st.session_state.df_long_original = df_long
            st.session_state.gdf_municipios = gdf_municipios
            st.session_state.data_loaded = True
            st.rerun()
        else:
            st.error("Hubo un error al procesar los archivos. Por favor, revisa los formatos y vuelve a intentarlo.")
            st.stop()

## CAMBIO 3: Se añade esta validación para prevenir el AttributeError.
## El código no continuará hasta que los datos estén cargados en el estado de la sesión.
if 'data_loaded' not in st.session_state or not st.session_state.data_loaded:
    st.stop()

# Cargar datos desde el estado de la sesión a variables locales para un uso seguro
gdf_stations = st.session_state.gdf_stations
df_long_original = st.session_state.df_long_original
gdf_municipios = st.session_state.gdf_municipios

# --- Filtros de Visualización en la Barra Lateral ---
with st.sidebar.expander("**2. Filtros Geográficos** 🗺️", expanded=True):
    if 'porc_datos' in gdf_stations.columns:
        min_data_perc = st.slider("Filtrar por % mínimo de datos:", 0, 100, 0, help="Muestra solo estaciones con un porcentaje de datos superior al seleccionado.")
        stations_master_list = gdf_stations[gdf_stations['porc_datos'] >= min_data_perc]
    else:
        stations_master_list = gdf_stations

    municipios_list = sorted(stations_master_list['municipio'].unique())
    selected_municipios = st.multiselect('Filtrar por Municipio', options=municipios_list)
    
    stations_available = stations_master_list[stations_master_list['municipio'].isin(selected_municipios)] if selected_municipios else stations_master_list
    stations_options = sorted(stations_available['nom_est'].unique())

    select_all = st.checkbox("Seleccionar todas las estaciones filtradas")
    if select_all:
        selected_stations = st.multiselect('Estaciones Seleccionadas', options=stations_options, default=stations_options, disabled=True)
    else:
        selected_stations = st.multiselect('Seleccionar Estaciones', options=stations_options, default=stations_options[0] if stations_options else [])

with st.sidebar.expander("**3. Filtros Temporales** 🗓️", expanded=True):
    available_years = sorted([int(col) for col in gdf_stations.columns if str(col).isdigit()])
    if not available_years:
        st.error("No se encontraron columnas de años (ej: '2020') en el archivo de estaciones.")
        st.stop()
    
    start_year, end_year = min(available_years), max(available_years)
    year_range = st.slider("Seleccionar Rango de Años", start_year, end_year, (start_year, end_year))
    
    months_dict = {'Enero': 1, 'Febrero': 2, 'Marzo': 3, 'Abril': 4, 'Mayo': 5, 'Junio': 6, 'Julio': 7, 'Agosto': 8, 'Septiembre': 9, 'Octubre': 10, 'Noviembre': 11, 'Diciembre': 12}
    selected_months_names = st.multiselect("Seleccionar Meses", list(months_dict.keys()), default=list(months_dict.keys()))
    selected_months_nums = [months_dict[m] for m in selected_months_names]

with st.sidebar.expander("**4. Opciones de Análisis** 🔬", expanded=True):
    analysis_mode = st.radio(
        "Análisis de Series Mensuales",
        ("Usar datos originales", "Completar series (interpolación)"),
        help="La interpolación rellena meses sin datos para crear series continuas."
    )

if not selected_stations or not selected_months_nums:
    st.warning("Por favor, seleccione al menos una estación y un mes en el panel de la izquierda.")
    st.stop()

# --- Preparación de Datos Filtrados ---
if analysis_mode == "Completar series (interpolación)":
    if 'df_completed' not in st.session_state:
        st.session_state.df_completed = complete_time_series(df_long_original)
    df_monthly = st.session_state.df_completed
else:
    df_monthly = df_long_original

df_monthly_filtered = df_monthly[
    (df_monthly['nom_est'].isin(selected_stations)) &
    (df_monthly['fecha_mes_año'].dt.year.between(year_range[0], year_range[1])) &
    (df_monthly['fecha_mes_año'].dt.month.isin(selected_months_nums))
]

df_anual_melted = gdf_stations[gdf_stations['nom_est'].isin(selected_stations)].melt(
    id_vars=['nom_est', 'longitud_geo', 'latitud_geo'],
    value_vars=[str(y) for y in range(year_range[0], year_range[1] + 1) if str(y) in gdf_stations.columns],
    var_name='año', value_name='precipitacion'
)
df_anual_melted['precipitacion'] = pd.to_numeric(df_anual_melted['precipitacion'].astype(str).str.replace(',', '.'), errors='coerce')
df_anual_melted.dropna(subset=['precipitacion'], inplace=True)


# --- Pestañas de Visualización ---
tab_graphs, tab_maps, tab_stats, tab_enso, tab_downloads = st.tabs([
    "📊 Gráficos", "🗺️ Mapas", "📈 Estadísticas y Tablas", "🌀 Análisis ENSO", "📥 Descargas"
])

with tab_graphs:
    st.header("Visualizaciones de Precipitación")
    
    if df_anual_melted.empty and df_monthly_filtered.empty:
        st.warning("No hay datos disponibles para las selecciones actuales.")
    else:
        with st.expander("Precipitación Anual", expanded=True):
            if not df_anual_melted.empty:
                chart_anual = alt.Chart(df_anual_melted).mark_line(point=True).encode(
                    x=alt.X('año:O', title='Año'),
                    y=alt.Y('precipitacion:Q', title='Precipitación Anual (mm)'),
                    color='nom_est:N',
                    tooltip=['nom_est', 'año', 'precipitacion']
                ).properties(height=500).interactive()
                st.altair_chart(chart_anual, use_container_width=True)
            else:
                st.info("No hay datos anuales para mostrar.")

        with st.expander("Precipitación Mensual", expanded=True):
            if not df_monthly_filtered.empty:
                chart_monthly = alt.Chart(df_monthly_filtered).mark_line(point=True).encode(
                    x=alt.X('fecha_mes_año:T', title='Fecha'),
                    y=alt.Y('precipitation:Q', title='Precipitación Mensual (mm)'),
                    color='nom_est:N',
                    shape=alt.Shape('origen:N', title='Origen del Dato'),
                    tooltip=['nom_est', 'fecha_mes_año', 'precipitation', 'origen']
                ).properties(height=500).interactive()
                st.altair_chart(chart_monthly, use_container_width=True)
            else:
                st.info("No hay datos mensuales para mostrar.")

with tab_maps:
    st.header("Visualizaciones Geográficas")
    map_type = st.radio("Seleccionar tipo de mapa:", 
                        ("Ubicación de Estaciones", "Comparación Anual / Kriging", "Mapa Animado ENSO"),
                        horizontal=True, label_visibility="collapsed")
    
    gdf_filtered_stations = gdf_stations[gdf_stations['nom_est'].isin(selected_stations)]

    if map_type == "Ubicación de Estaciones":
        if not gdf_filtered_stations.empty:
            st.subheader("Mapa de Ubicación de Estaciones Seleccionadas")
            m = folium.Map(location=[6.24, -75.58], zoom_start=8, tiles="cartodbpositron")
            folium.GeoJson(gdf_municipios.to_json(), name='Municipios', style_function=lambda x: {'fillColor': 'transparent', 'color': 'gray'}).add_to(m)
            for _, row in gdf_filtered_stations.iterrows():
                html = f"<b>Estación:</b> {row['nom_est']}<br><b>Municipio:</b> {row['municipio']}"
                folium.Marker([row['latitud_geo'], row['longitud_geo']], tooltip=html).add_to(m)
            
            bounds = gdf_filtered_stations.total_bounds
            m.fit_bounds([[bounds[1], bounds[0]], [bounds[3], bounds[2]]])
            folium_static(m, width=None, height=600)
        else:
            st.warning("No hay estaciones seleccionadas para mostrar en el mapa.")

    elif map_type == "Comparación Anual / Kriging":
        st.subheader("Mapa Comparativo de Precipitación Anual")
        if not df_anual_melted.empty and df_anual_melted['año'].nunique() > 0:
            min_precip, max_precip = int(df_anual_melted['precipitacion'].min()), int(df_anual_melted['precipitacion'].max())
            color_range = st.slider("Rango de Escala de Color (mm)", min_precip, max_precip, (min_precip, max_precip))
            
            min_year, max_year = int(df_anual_melted['año'].min()), int(df_anual_melted['año'].max())
            year1 = st.slider("Seleccione un año", min_year, max_year, max_year)
            
            data_year1 = df_anual_melted[df_anual_melted['año'].astype(int) == year1]
            gdf_data_year1 = gpd.GeoDataFrame(
                data_year1, geometry=gpd.points_from_xy(data_year1['longitud_geo'], data_year1['latitud_geo']), crs="EPSG:4326"
            )

            col1, col2 = st.columns(2)
            with col1:
                st.markdown(f"**Mapa de Puntos - Año {year1}**")
                fig1 = px.scatter_geo(
                    data_year1, lat='latitud_geo', lon='longitud_geo', color='precipitacion',
                    size='precipitacion', hover_name='nom_est', color_continuous_scale='YlGnBu',
                    range_color=color_range, projection='natural earth'
                )
                fig1.update_geos(fitbounds="locations", visible=True, showcoastlines=True)
                fig1.update_layout(height=600, margin=dict(t=10, b=10, l=10, r=10))
                st.plotly_chart(fig1, use_container_width=True)
            
            with col2:
                st.markdown(f"**Superficie Interpolada (Kriging) - Año {year1}**")
                with st.spinner("Generando mapa Kriging..."):
                    fig2 = generate_kriging_map(gdf_data_year1, color_range)
                    st.plotly_chart(fig2, use_container_width=True)
        else:
            st.warning("No hay datos anuales para generar mapas comparativos.")

    elif map_type == "Mapa Animado ENSO":
        st.subheader("Evolución Mensual del Fenómeno ENSO")
        st.info("El color de cada estación representa la fase ENSO global para cada mes.")
        
        enso_anim_data = df_monthly[['fecha_mes_año', 'anomalia_oni']].drop_duplicates().dropna()
        enso_anim_data = enso_anim_data[enso_anim_data['fecha_mes_año'].dt.year.between(year_range[0], year_range[1])]
        conditions = [enso_anim_data['anomalia_oni'] >= 0.5, enso_anim_data['anomalia_oni'] <= -0.5]
        enso_anim_data['fase'] = np.select(conditions, ['El Niño', 'La Niña'], default='Neutral')
        enso_anim_data['fecha_str'] = enso_anim_data['fecha_mes_año'].dt.strftime('%Y-%m')
        
        animation_df = gdf_filtered_stations[['nom_est', 'geometry']].merge(enso_anim_data, how='cross')
        animation_df['latitud_geo'] = animation_df.geometry.y
        animation_df['longitud_geo'] = animation_df.geometry.x

        if not animation_df.empty:
            fig_enso_anim = px.scatter_geo(
                animation_df.sort_values('fecha_str'), lat='latitud_geo', lon='longitud_geo', color='fase',
                animation_frame='fecha_str', hover_name='nom_est',
                color_discrete_map={'El Niño': 'red', 'La Niña': 'blue', 'Neutral': 'lightgrey'},
                category_orders={"fase": ["El Niño", "La Niña", "Neutral"]},
                projection='natural earth'
            )
            fig_enso_anim.update_geos(fitbounds="locations", visible=True)
            fig_enso_anim.update_layout(height=700, title_text=None)
            st.plotly_chart(fig_enso_anim, use_container_width=True)
        else:
            st.warning("No hay datos ENSO para generar la animación en el rango seleccionado.")

with tab_stats:
    st.header("Estadísticas y Datos Tabulares")
    
    with st.expander("Resumen Estadístico", expanded=True):
        if not df_monthly_filtered.empty:
            col1, col2, col3 = st.columns(3)
            max_monthly_row = df_monthly_filtered.loc[df_monthly_filtered['precipitation'].idxmax()]
            col1.metric(
                "Máxima Ppt. Mensual",
                f"{max_monthly_row['precipitation']:.1f} mm",
                f"{max_monthly_row['nom_est']} ({max_monthly_row['fecha_mes_año'].strftime('%b %Y')})"
            )
            col2.metric("Promedio Ppt. Mensual", f"{df_monthly_filtered['precipitation'].mean():.1f} mm")
            col3.metric("Total de Registros", f"{len(df_monthly_filtered):,}")
            
            st.dataframe(df_monthly_filtered.groupby('nom_est')['precipitation'].describe().round(2))
        else:
            st.info("No hay datos para mostrar estadísticas.")

    with st.expander("Matriz de Disponibilidad de Datos", expanded=False):
        data_counts = df_long_original[df_long_original['nom_est'].isin(selected_stations)]
        data_counts['año'] = data_counts['fecha_mes_año'].dt.year
        heatmap_df = data_counts.groupby(['nom_est', 'año']).size().unstack(fill_value=0)
        
        if not heatmap_df.empty:
            fig_heatmap = px.imshow(
                heatmap_df, text_auto=True, aspect="auto", color_continuous_scale="Greens",
                labels=dict(x="Año", y="Estación", color="# Registros"),
                title="Número de Registros Mensuales Originales por Año"
            )
            fig_heatmap.update_layout(height=max(400, len(selected_stations) * 35))
            st.plotly_chart(fig_heatmap, use_container_width=True)
        else:
            st.info("No hay datos para generar la matriz.")

    with st.expander("Tabla de Datos Mensuales", expanded=False):
        if not df_monthly_filtered.empty:
            st.dataframe(df_monthly_filtered[['fecha_mes_año', 'nom_est', 'precipitation', 'origen']].round(2))
        else:
            st.info("No hay datos para mostrar.")

with tab_enso:
    st.header("Análisis de Precipitación vs. Fenómeno ENSO")
    
    df_analisis = df_monthly_filtered.copy()
    if 'anomalia_oni' not in df_analisis.columns or df_analisis['anomalia_oni'].isnull().all():
        st.warning("No se encontraron datos de 'anomalia_oni' para el período seleccionado.")
    else:
        df_analisis['enso_fase'] = df_analisis['anomalia_oni'].apply(
            lambda x: 'El Niño' if x >= 0.5 else ('La Niña' if x <= -0.5 else 'Neutral')
        )
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Precipitación por Fase ENSO")
            df_enso_group = df_analisis.groupby('enso_fase')['precipitation'].mean().reset_index()
            fig_enso = px.bar(
                df_enso_group, x='enso_fase', y='precipitation', color='enso_fase',
                labels={'precipitation': 'Precipitación Media (mm)', 'enso_fase': 'Fase ENSO'},
                color_discrete_map={'El Niño': 'red', 'La Niña': 'blue', 'Neutral': 'grey'}
            )
            st.plotly_chart(fig_enso, use_container_width=True)
        with col2:
            st.subheader("Correlación Precipitación vs. ONI")
            fig_corr = px.scatter(
                df_analisis, x='anomalia_oni', y='precipitation',
                trendline='ols',
                labels={'anomalia_oni': 'Anomalía ONI (°C)', 'precipitation': 'Precipitación (mm)'},
                title="Dispersión y Correlación"
            )
            st.plotly_chart(fig_corr, use_container_width=True)

        st.subheader("Gráfico General del Fenómeno ENSO")
        enso_chart_data = df_monthly[['fecha_mes_año', 'anomalia_oni']].drop_duplicates()
        enso_chart_data = enso_chart_data[enso_chart_data['fecha_mes_año'].dt.year.between(year_range[0], year_range[1])]
        st.plotly_chart(create_enso_chart(enso_chart_data), use_container_width=True)

with tab_downloads:
    st.header("Opciones de Descarga de Datos")
    st.info("Aquí puedes descargar los datos filtrados según tus selecciones en el panel de control.")

    @st.cache_data
    def convert_df_to_csv(df):
        return df.to_csv(index=False).encode('utf-8')

    if not df_monthly_filtered.empty:
        csv_mensual = convert_df_to_csv(df_monthly_filtered)
        st.download_button(
            label="📥 Descargar Datos Mensuales (.csv)",
            data=csv_mensual,
            file_name=f'precipitacion_mensual_{year_range[0]}_{year_range[1]}.csv',
            mime='text/csv'
        )
    else:
        st.warning("No hay datos mensuales para descargar.")
        
    if not df_anual_melted.empty:
        csv_anual = convert_df_to_csv(df_anual_melted)
        st.download_button(
            label="📥 Descargar Datos Anuales (.csv)",
            data=csv_anual,
            file_name=f'precipitacion_anual_{year_range[0]}_{year_range[1]}.csv',
            mime='text/csv'
        )
    else:
        st.warning("No hay datos anuales para descargar.")
