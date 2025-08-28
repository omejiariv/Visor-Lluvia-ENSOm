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

# 1. ESTANDARIZACIÓN DE COLUMNAS (A MINÚSCULAS)
df_precip_mensual.columns = df_precip_mensual.columns.str.strip().str.lower()
year_col_precip = next((col for col in df_precip_mensual.columns if ('año' in col or 'ano' in col) and 'enso' not in col), None)
if not year_col_precip:
    st.error(f"No se encontró columna de año principal ('año' o 'ano') en el archivo de precipitación mensual.")
    st.stop()
df_precip_mensual.rename(columns={year_col_precip: 'año'}, inplace=True) # Renombrar a 'año' minúscula

# 2. Procesar y crear tabla de referencia ENSO (usando nombres en minúscula)
# FIX: Use lowercase for all column names to match standardization
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

# 3. Procesar estaciones (mapaCVENSO)
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

# 4. Procesar datos de Precipitación Mensual
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

# 5. Mapeo y Fusión
gdf_stations['Id_estacio'] = gdf_stations['Id_estacio'].astype(str).str.strip()
df_long['Id_estacion'] = df_long['Id_estacion'].astype(str).str.strip()
station_mapping = gdf_stations.set_index('Id_estacio')['Nom_Est'].to_dict()
df_long['Nom_Est'] = df_long['Id_estacion'].map(station_mapping)
df_long.dropna(subset=['Nom_Est'], inplace=True)
df_long.rename(columns={'año': 'Año'}, inplace=True) # Devolver a Mayúscula para filtros y visualizaciones
if df_long.empty:
    st.warning("El dataframe de precipitación mensual está vacío después del preprocesamiento. Verifique el formato de los datos de precipitación.")
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
select_all = st.sidebar.checkbox("Seleccionar/Deseleccionar Todas las Estaciones", value=True)
default_selection = stations_options if select_all else []
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
    (df_monthly_for_analysis['Nom_Est'].isin(selected_stations)) &
    (df_monthly_for_analysis['Fecha'].dt.year >= year_range[0]) &
    (df_monthly_for_analysis['Fecha'].dt.year <= year_range[1]) &
    (df_monthly_for_analysis['Fecha'].dt.month.isin(meses_numeros))]

# --- Pestañas Principales ---
tab1, tab2, tab_anim, tab3, tab4, tab5 = st.tabs(["Gráficos", "Mapa de Estaciones", "Mapas Avanzados", "Tabla de Estaciones", "Análisis ENSO", "Descargas"])

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
        lat_center, lon_center = gdf_filtered['Latitud_geo'].mean(), gdf_filtered['Longitud_geo'].mean()
        m = folium.Map(location=[lat_center, lon_center], zoom_start=8)
        folium.GeoJson(gdf_municipios.to_json(), name='Municipios').add_to(m)
        for _, row in gdf_filtered.iterrows():
            folium.Marker([row['Latitud_geo'], row['Longitud_geo']], tooltip=f"<b>{row['Nom_Est']}</b><br>{row['municipio']}").add_to(m)
        bounds = gdf_filtered.total_bounds
        m.fit_bounds([[bounds[1], bounds[0]], [bounds[3], bounds[2]]])
        folium_static(m, width=900, height=600)
    else:
        st.warning("No hay estaciones seleccionadas para mostrar en el mapa.")

with tab_anim:
    st.header("Mapas Avanzados")
    anim_points_tab, anim_kriging_tab = st.tabs(["Animación de Puntos", "Análisis Kriging"])
    with anim_points_tab:
        st.subheader("Mapa Animado de Precipitación Anual")
        if not df_anual_melted.empty:
            fig_mapa_animado = px.scatter_geo(df_anual_melted, lat='Latitud_geo', lon='Longitud_geo', color='Precipitación', size='Precipitación', hover_name='Nom_Est', animation_frame='Año', projection='natural earth', title='Precipitación Anual por Estación', color_continuous_scale=px.colors.sequential.YlGnBu)
            fig_mapa_animado.update_geos(fitbounds="locations", visible=True)
            fig_mapa_animado.update_layout(height=700)
            st.plotly_chart(fig_mapa_animado, use_container_width=True)
    with anim_kriging_tab:
        st.subheader("Comparación de Mapas de Precipitación Anual (Kriging)")
        available_years = sorted(df_anual_melted['Año'].astype(int).unique())
        if available_years:
            st.sidebar.markdown("### Opciones de Mapa Comparativo")
            min_precip, max_precip = int(df_anual_melted['Precipitación'].min()), int(df_anual_melted['Precipitación'].max())
            color_range = st.sidebar.slider("Rango de Escala de Color (mm)", min_precip, max_precip, (min_precip, max_precip))
            col1, col2 = st.columns(2)
            year1 = col1.selectbox("Seleccione el año para el Mapa 1", available_years, index=len(available_years)-1, key='y1')
            year2 = col2.selectbox("Seleccione el año para el Mapa 2", available_years, index=len(available_years)-2 if len(available_years) > 1 else 0, key='y2')
            if st.button("Generar Mapas de Comparación"):
                if year1 == year2:
                    st.info("Años iguales: Mapa 1 muestra Puntos, Mapa 2 muestra Superficie Kriging.")
                    data_year = df_anual_melted[df_anual_melted['Año'].astype(int) == year1]
                    if len(data_year) < 3:
                        st.warning(f"Se necesitan al menos 3 estaciones para generar el mapa Kriging del año {year1}.")
                    else:
                        with col1:
                            st.subheader(f"Estaciones - Año: {year1}")
                            fig1 = px.scatter_geo(data_year, lat='Latitud_geo', lon='Longitud_geo', color='Precipitación', size='Precipitación', hover_name='Nom_Est', color_continuous_scale='YlGnBu', range_color=color_range, projection='natural earth')
                            fig1.update_geos(fitbounds="locations", visible=True)
                            st.plotly_chart(fig1, use_container_width=True)
                        with col2, st.spinner("Generando mapa Kriging..."):
                            st.subheader(f"Interpolación Kriging - Año: {year1}")
                            lons, lats, vals = data_year['Longitud_geo'].values, data_year['Latitud_geo'].values, data_year['Precipitación'].values
                            grid_lon, grid_lat = np.linspace(lons.min()-0.1, lons.max()+0.1, 100), np.linspace(lats.min()-0.1, lats.max()+0.1, 100)
                            OK = OrdinaryKriging(lons, lats, vals, variogram_model='linear', verbose=False, enable_plotting=False)
                            z, ss = OK.execute('grid', grid_lon, grid_lat)
                            fig2 = go.Figure(data=go.Contour(z=z, x=grid_lon, y=grid_lat, colorscale='YlGnBu', zmin=color_range[0], zmax=color_range[1]))
                            fig2.add_trace(go.Scatter(x=lons, y=lats, mode='markers', marker=dict(color='red', size=4), name='Estaciones'))
                            st.plotly_chart(fig2, use_container_width=True)
                else:
                    st.info("Años diferentes: Se comparan los Puntos de Estaciones para cada año.")
                    for i, (col, year) in enumerate(zip([col1, col2], [year1, year2])):
                        with col:
                            st.subheader(f"Estaciones Año: {year}")
                            data_year = df_anual_melted[df_anual_melted['Año'].astype(int) == year]
                            if data_year.empty:
                                st.warning(f"No hay datos para el año {year}.")
                                continue
                            fig = px.scatter_geo(data_year, lat='Latitud_geo', lon='Longitud_geo', color='Precipitación', size='Precipitación', hover_name='Nom_Est', color_continuous_scale='YlGnBu', range_color=color_range, projection='natural earth')
                            fig.update_geos(fitbounds="locations", visible=True)
                            st.plotly_chart(fig, use_container_width=True, key=f'map_diff_{i}')
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

with tab4:
    st.header("Análisis de Precipitación y el Fenómeno ENSO")
    enso_corr_tab, enso_series_tab = st.tabs(["Correlación Precipitación-ENSO", "Series de Tiempo ENSO"])
    with enso_corr_tab:
        df_analisis = df_monthly_filtered.copy()
        df_analisis['fecha_merge'] = df_analisis['Fecha'].dt.strftime('%Y-%m')
        df_analisis = pd.merge(df_analisis, df_enso, on='fecha_merge', how='left')
        
        # FIX: Use lowercase for all column name checks
        if 'anomalia_oni' in df_analisis.columns:
            df_analisis.dropna(subset=['anomalia_oni'], inplace=True)

            def classify_enso(oni):
                if oni >= 0.5:
                    return 'El Niño'
                elif oni <= -0.5:
                    return 'La Niña'
                else:
                    return 'Neutral'
            
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
            
    with enso_series_tab:
        st.subheader("Visualización de Variables ENSO")
        # FIX: Use lowercase for all column name checks
        enso_vars_available = [v for v in ['anomalia_oni', 'temp_sst', 'temp_media'] if v in df_enso.columns]
        if not enso_vars_available:
            st.warning("No hay variables ENSO disponibles en el archivo de datos para visualizar.")
        else:
            variable_enso = st.selectbox("Seleccione la variable ENSO a visualizar:", enso_vars_available)
            df_enso_filtered = df_enso[
                (df_enso['Fecha'].dt.year >= year_range[0]) &
                (df_enso['Fecha'].dt.year <= year_range[1]) &
                (df_enso['Fecha'].dt.month.isin(meses_numeros))]
            if not df_enso_filtered.empty and variable_enso in df_enso_filtered.columns and not df_enso_filtered[variable_enso].isnull().all():
                fig_enso_series = px.line(df_enso_filtered, x='Fecha', y=variable_enso, title=f"Serie de Tiempo para {variable_enso}")
                st.plotly_chart(fig_enso_series, use_container_width=True)
            else:
                st.warning(f"No hay datos disponibles para '{variable_enso}' en el período seleccionado.")

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
        st.info("Para descargar las series completadas, seleccione la opción 'Completar series (interpolación)' en el panel lateral.
