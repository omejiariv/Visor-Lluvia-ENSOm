# Visor de Información Geoespacial de Precipitación y el Fenómeno ENSO
# Versión final simplificada con datos pre-procesados
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
import re
import csv

try:
    from folium.plugins import ScaleControl
except ImportError:
    class ScaleControl:
        def __init__(self, *args, **kwargs): pass
        def add_to(self, m): pass

#--- Configuración de la página
st.set_page_config(layout="wide", page_title="Visor de Precipitación y ENSO")
st.markdown("""
<style>
div.block-container {padding-top: 2rem;}
.sidebar .sidebar-content { font-size: 13px; }
h1 { margin-top: 0px; padding-top: 0px; }
</style>
""", unsafe_allow_html=True)

#--- Funciones de Carga y Procesamiento ---
@st.cache_data
def load_data(file_path):
    if file_path is None: return None
    try:
        content = file_path.getvalue()
        if not content.strip():
            st.error(f"El archivo '{file_path.name}' parece estar vacío.")
            return None
    except Exception as e:
        st.error(f"No se pudo leer el contenido del archivo '{file_path.name}': {e}")
        return None
    encodings_to_try = ['utf-8', 'latin1', 'cp1252', 'iso-8859-1']
    for encoding in encodings_to_try:
        try:
            sample_str = content[:4096].decode(encoding)
            try:
                dialect = csv.Sniffer().sniff(sample_str, delimiters=';,')
                sep = dialect.delimiter
            except csv.Error:
                sep = ';' if sample_str.count(';') > sample_str.count(',') else ','
            df = pd.read_csv(io.BytesIO(content), sep=sep, encoding=encoding, low_memory=False)
            df.columns = df.columns.str.strip()
            return df
        except Exception:
            continue
    st.error(f"No se pudo decodificar el archivo '{file_path.name}'.")
    return None

@st.cache_data
def load_shapefile(file_path):
    try:
        with tempfile.TemporaryDirectory() as temp_dir:
            with zipfile.ZipFile(file_path, 'r') as zip_ref:
                zip_ref.extractall(temp_dir)
            shp_path = [f for f in os.listdir(temp_dir) if f.endswith('.shp')][0]
            gdf = gpd.read_file(os.path.join(temp_dir, shp_path))
            gdf.columns = gdf.columns.str.strip()
            if gdf.crs is None:
                gdf = gdf.set_crs("EPSG:9377")
            return gdf.to_crs("EPSG:4326")
    except Exception as e:
        st.error(f"Error al procesar el shapefile: {e}")
        return None

#--- Interfaz y Carga de Archivos ---
st.title('Visor de Precipitación y Fenómeno ENSO')
st.sidebar.header("Panel de Control")
with st.sidebar.expander("**Cargar Archivos**", expanded=True):
    st.info("Cargue los archivos de precipitación que ya contienen las columnas ENSO.")
    uploaded_file_mapa = st.file_uploader("1. Archivo de estaciones con ENSO Anual", type="csv")
    uploaded_file_precip = st.file_uploader("2. Archivo de precipitación con ENSO Mensual", type="csv")
    uploaded_zip_shapefile = st.file_uploader("3. Shapefile de municipios (.zip)", type="zip")

if not all([uploaded_file_mapa, uploaded_file_precip, uploaded_zip_shapefile]):
    st.info("Por favor, suba los 3 archivos requeridos para habilitar la aplicación.")
    st.stop()

#--- Carga y Preprocesamiento de Datos ---
df_precip_anual = load_data(uploaded_file_mapa)
df_precip_mensual = load_data(uploaded_file_precip)
gdf_municipios = load_shapefile(uploaded_zip_shapefile)
if any(df is None for df in [df_precip_anual, df_precip_mensual, gdf_municipios]):
    st.stop()
    
# Estaciones (mapaCVENSO)
lon_col = next((col for col in df_precip_anual.columns if 'longitud' in col.lower() or 'lon' in col.lower()), None)
lat_col = next((col for col in df_precip_anual.columns if 'latitud' in col.lower() or 'lat' in col.lower()), None)
if not all([lon_col, lat_col]):
    st.error(f"No se encontraron las columnas de longitud y/o latitud en el archivo de estaciones.")
    st.stop()
df_precip_anual[lon_col] = pd.to_numeric(df_precip_anual[lon_col], errors='coerce')
df_precip_anual[lat_col] = pd.to_numeric(df_precip_anual[lat_col], errors='coerce')
df_precip_anual.dropna(subset=[lon_col, lat_col], inplace=True)
gdf_stations = gpd.GeoDataFrame(df_precip_anual, geometry=gpd.points_from_xy(df_precip_anual[lon_col], df_precip_anual[lat_col]), crs="EPSG:9377").to_crs("EPSG:4326")
gdf_stations['Longitud_geo'] = gdf_stations.geometry.x
gdf_stations['Latitud_geo'] = gdf_stations.geometry.y

# Precipitación Mensual (ya enriquecida)
df_precip_mensual.columns = [col.lower() for col in df_precip_mensual.columns]
year_col_precip = next((col for col in df_precip_mensual.columns if 'año' in col or 'ano' in col), None)
enso_mes_col_precip = next((col for col in df_precip_mensual.columns if 'enso_mes' in col or 'enso-mes' in col), None)
if not all([year_col_precip, enso_mes_col_precip]):
    st.error(f"No se encontraron 'año'/'ano' o 'enso_mes' en el archivo de precipitación mensual.")
    st.stop()
df_precip_mensual.rename(columns={year_col_precip: 'Año', enso_mes_col_precip: 'ENSO'}, inplace=True)

station_cols = [col for col in df_precip_mensual.columns if col.isdigit()]
id_vars = [col for col in df_precip_mensual.columns if not col.isdigit()]

df_long = df_precip_mensual.melt(id_vars=id_vars, value_vars=station_cols, var_name='Id_estacion', value_name='Precipitation')
df_long['Precipitation'] = pd.to_numeric(df_long['Precipitation'], errors='coerce')
df_long.dropna(subset=['Precipitation'], inplace=True)
df_long['Fecha'] = pd.to_datetime(df_long['Año'].astype(str) + '-' + df_long['mes'].astype(str), errors='coerce')
df_long.dropna(subset=['Fecha'], inplace=True)

# Mapeo y Fusión
gdf_stations['Id_estacio'] = gdf_stations['Id_estacio'].astype(str).str.strip()
df_long['Id_estacion'] = df_long['Id_estacion'].astype(str).str.strip()
station_mapping = gdf_stations.set_index('Id_estacio')['Nom_Est'].to_dict()
df_long['Nom_Est'] = df_long['Id_estacion'].map(station_mapping)
df_long.dropna(subset=['Nom_Est'], inplace=True)
if df_long.empty: st.stop()

#--- Controles en la Barra Lateral ---
st.sidebar.markdown("### Filtros de Visualización")
municipios_list = sorted(gdf_stations['municipio'].unique())
selected_municipios = st.sidebar.multiselect('1. Filtrar por Municipio', options=municipios_list, default=municipios_list)
stations_available = gdf_stations[gdf_stations['municipio'].isin(selected_municipios)]
stations_options = sorted(stations_available['Nom_Est'].unique())
selected_stations = st.sidebar.multiselect('2. Seleccionar Estaciones', options=stations_options, default=stations_options)

id_cols = [col for col in gdf_stations.columns if not col.isdigit()]
year_cols_numeric = [col for col in gdf_stations.columns if col.isdigit()]
if not year_cols_numeric:
    st.warning("No se encontraron columnas de años en el archivo de estaciones.")
    st.stop()
años_disponibles = sorted([int(y) for y in year_cols_numeric])
year_range = st.sidebar.slider("3. Seleccionar Rango de Años", min(años_disponibles), max(años_disponibles), (min(años_disponibles), max(años_disponibles)))

#--- Preparación de datos filtrados ---
df_anual_melted = gdf_stations[gdf_stations['Nom_Est'].isin(selected_stations)].melt(id_vars=id_cols, value_vars=[str(y) for y in años_disponibles], var_name='Año', value_name='Precipitación')
df_anual_melted['Año'] = pd.to_numeric(df_anual_melted['Año'], errors='coerce')
df_anual_melted.dropna(subset=['Año'], inplace=True)
df_anual_melted['Año'] = df_anual_melted['Año'].astype(int)
df_anual_filtered = df_anual_melted[(df_anual_melted['Año'] >= year_range[0]) & (df_anual_melted['Año'] <= year_range[1])]

df_monthly_filtered = df_long[(df_long['Nom_Est'].isin(selected_stations)) & (df_long['Fecha'].dt.year >= year_range[0]) & (df_long['Fecha'].dt.year <= year_range[1])]

#--- Pestañas Principales ---
tab1, tab2, tab3, tab4 = st.tabs(["Series de Tiempo", "Mapas", "Tabla de Estaciones", "Análisis ENSO"])

with tab1:
    st.header("Visualizaciones de Precipitación")
    st.subheader("Precipitación Anual (mm)")
    if not df_anual_filtered.empty:
        chart_anual = alt.Chart(df_anual_filtered).mark_line().encode(
            x=alt.X('Año:O', title='Año'),
            y=alt.Y('Precipitación:Q', title='Precipitación Total (mm)'),
            color='Nom_Est:N',
            tooltip=['Nom_Est', 'Año', 'Precipitación']
        ).interactive()
        st.altair_chart(chart_anual, use_container_width=True)

    st.subheader("Precipitación Mensual (mm)")
    if not df_monthly_filtered.empty:
        chart_mensual = alt.Chart(df_monthly_filtered).mark_line().encode(
            x=alt.X('Fecha:T', title='Fecha'),
            y=alt.Y('Precipitation:Q', title='Precipitación Total (mm)'),
            color='Nom_Est:N',
            tooltip=[alt.Tooltip('Fecha', format='%Y-%m'), 'Precipitation', 'Nom_Est']
        ).interactive()
        st.altair_chart(chart_mensual, use_container_width=True)

with tab2:
    st.header("Mapa de Estaciones")
    gdf_filtered = gdf_stations[gdf_stations['Nom_Est'].isin(selected_stations)]
    if not gdf_filtered.empty:
        m = folium.Map(location=[gdf_filtered['Latitud_geo'].mean(), gdf_filtered['Longitud_geo'].mean()], zoom_start=6)
        for _, row in gdf_filtered.iterrows():
            folium.Marker([row['Latitud_geo'], row['Longitud_geo']], tooltip=f"{row['Nom_Est']}<br>{row['municipio']}").add_to(m)
        folium_static(m, width=900, height=600)
    else:
        st.warning("No hay estaciones seleccionadas para mostrar en el mapa.")

with tab3:
    st.header("Tabla de Estaciones")
    st.dataframe(gdf_stations[gdf_stations['Nom_Est'].isin(selected_stations)])

with tab4:
    st.header("Análisis de Precipitación y Fenómeno ENSO")
    df_analisis = df_monthly_filtered.copy()
    if not df_analisis.empty:
        st.subheader("Precipitación Media por Fase ENSO")
        enso_col = next((col for col in df_analisis.columns if 'enso' in col.lower()), None)
        if enso_col:
            df_enso_group = df_analisis.groupby(enso_col)['Precipitation'].mean().reset_index()
            fig_enso = px.bar(df_enso_group, x=enso_col, y='Precipitation', color=enso_col,
                              labels={'Precipitation': 'Precipitación Media (mm)', enso_col: 'Fase ENSO'})
            st.plotly_chart(fig_enso, use_container_width=True)
        else:
            st.warning("No se encontró la columna ENSO en los datos de precipitación mensual.")
    else: 
        st.warning("No hay datos para realizar el análisis ENSO.")
