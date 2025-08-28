# Visor de Información Geoespacial de Precipitación y el Fenómeno ENSO
# Versión final y estable
import streamlit as st
import pandas as pd
import altair as alt
import folium
from streamlit_folium import st_folium
import plotly.express as px
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
st.markdown("""<style>div.block-container {padding-top: 2rem;}</style>""", unsafe_allow_html=True)

#--- Funciones de Carga ---
@st.cache_data
def load_data(file_path):
    # ... (código de carga robusta sin cambios)
    if file_path is None: return None
    try: content = file_path.getvalue()
    except Exception as e: st.error(f"No se pudo leer el archivo: {e}"); return None
    if not content.strip(): st.error("El archivo parece estar vacío."); return None
    
    encodings_to_try = ['utf-8', 'latin1', 'cp1252']
    for encoding in encodings_to_try:
        try:
            sample_str = content[:4096].decode(encoding)
            sep = ';' if sample_str.count(';') > sample_str.count(',') else ','
            return pd.read_csv(io.BytesIO(content), sep=sep, encoding=encoding, low_memory=False)
        except Exception:
            continue
    st.error(f"No se pudo decodificar el archivo '{file_path.name}'.")
    return None

@st.cache_data
def load_shapefile(file_path):
    # ... (código de carga de shapefile sin cambios)
    try:
        with tempfile.TemporaryDirectory() as temp_dir:
            with zipfile.ZipFile(file_path, 'r') as zip_ref:
                zip_ref.extractall(temp_dir)
            shp_path = [f for f in os.listdir(temp_dir) if f.endswith('.shp')][0]
            gdf = gpd.read_file(os.path.join(temp_dir, shp_path))
            if gdf.crs is None:
                gdf = gdf.set_crs("EPSG:4326") # Asumir WGS84 si no tiene CRS
            return gdf.to_crs("EPSG:4326")
    except Exception as e:
        st.error(f"Error al procesar el shapefile: {e}")
        return None

#--- Interfaz y Carga de Archivos ---
st.title('Visor de Precipitación y Fenómeno ENSO')
st.sidebar.header("Panel de Control")
with st.sidebar.expander("**Cargar Archivos**", expanded=True):
    uploaded_file_mapa = st.file_uploader("1. Cargar archivo de estaciones (mapaCVENSO.csv)", type="csv")
    uploaded_file_precip = st.file_uploader("2. Cargar archivo de precipitación mensual con datos ENSO", type="csv")
    uploaded_zip_shapefile = st.file_uploader("3. Cargar shapefile de municipios (.zip)", type="zip")

if not all([uploaded_file_mapa, uploaded_file_precip, uploaded_zip_shapefile]):
    st.info("Por favor, suba los 3 archivos requeridos para habilitar la aplicación.")
    st.stop()

#--- Carga y Preprocesamiento de Datos ---
gdf_stations = load_data(uploaded_file_mapa)
df_precip_mensual_enriquecido = load_data(uploaded_file_precip)
gdf_municipios = load_shapefile(uploaded_zip_shapefile)
if any(df is None for df in [gdf_stations, df_precip_mensual_enriquecido, gdf_municipios]):
    st.stop()

# Convertir columnas de coordenadas en gdf_stations
lon_col = next((col for col in gdf_stations.columns if 'lon' in col.lower()), None)
lat_col = next((col for col in gdf_stations.columns if 'lat' in col.lower()), None)
if not all([lon_col, lat_col]): st.error("No se encontraron columnas de longitud/latitud en el archivo de estaciones."); st.stop()
gdf_stations[lon_col] = pd.to_numeric(gdf_stations[lon_col], errors='coerce')
gdf_stations[lat_col] = pd.to_numeric(gdf_stations[lat_col], errors='coerce')
gdf_stations.dropna(subset=[lon_col, lat_col], inplace=True)
gdf_stations['geometry'] = gpd.points_from_xy(gdf_stations[lon_col], gdf_stations[lat_col])

# Procesar el archivo de precipitación mensual enriquecido
df_precip_mensual_enriquecido.columns = [col.lower() for col in df_precip_mensual_enriquecido.columns]
station_cols = [col for col in df_precip_mensual_enriquecido.columns if col.isdigit()]
id_vars = [col for col in df_precip_mensual_enriquecido.columns if not col.isdigit()]
df_long = df_precip_mensual_enriquecido.melt(id_vars=id_vars, value_vars=station_cols, var_name='Id_estacion', value_name='Precipitation')
df_long['Precipitation'] = pd.to_numeric(df_long['Precipitation'], errors='coerce')
df_long['Fecha'] = pd.to_datetime(df_long['año'].astype(str) + '-' + df_long['mes'].astype(str), errors='coerce')
df_long.dropna(subset=['Precipitation', 'Fecha'], inplace=True)
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
year_cols_numeric = [col for col in gdf_stations.columns if col.isdigit()]
if not year_cols_numeric: st.warning("No se encontraron columnas de años."); st.stop()
años_disponibles = sorted([int(y) for y in year_cols_numeric])
year_range = st.sidebar.slider("3. Seleccionar Rango de Años", min(años_disponibles), max(años_disponibles), (min(años_disponibles), max(años_disponibles)))

#--- Preparación de datos filtrados ---
id_cols_safe = [col for col in gdf_stations.columns if not col.isdigit() and col != 'geometry']
df_anual_melted = gdf_stations[gdf_stations['Nom_Est'].isin(selected_stations)].melt(id_vars=id_cols_safe, value_vars=[str(y) for y in años_disponibles], var_name='Año', value_name='Precipitación')
df_anual_melted['Año'] = pd.to_numeric(df_anual_melted['Año'], errors='coerce').astype(int)
df_anual_filtered = df_anual_melted[(df_anual_melted['Año'] >= year_range[0]) & (df_anual_melted['Año'] <= year_range[1])]
df_monthly_filtered = df_long[(df_long['Nom_Est'].isin(selected_stations)) & (df_long['Fecha'].dt.year >= year_range[0]) & (df_long['Fecha'].dt.year <= year_range[1])]

#--- Pestañas Principales ---
tab1, tab2, tab3, tab4 = st.tabs(["Series de Tiempo", "Mapa de Estaciones", "Tabla de Estaciones", "Análisis ENSO"])

with tab1:
    st.header("Visualizaciones de Precipitación")
    st.subheader("Precipitación Anual (mm)")
    if not df_anual_filtered.empty:
        chart_anual = alt.Chart(df_anual_filtered).mark_line(point=True).encode(x='Año:O', y='Precipitación:Q', color='Nom_Est:N', tooltip=['Nom_Est', 'Año', 'Precipitación']).interactive()
        st.altair_chart(chart_anual, use_container_width=True)

    st.subheader("Precipitación Mensual (mm)")
    if not df_monthly_filtered.empty:
        chart_mensual = alt.Chart(df_monthly_filtered).mark_line(point=True).encode(x='Fecha:T', y='Precipitation:Q', color='Nom_Est:N', tooltip=['Nom_Est', 'Fecha', 'Precipitation']).interactive()
        st.altair_chart(chart_mensual, use_container_width=True)

with tab2:
    st.header("Mapa de Estaciones")
    gdf_filtered = gdf_stations[gdf_stations['Nom_Est'].isin(selected_stations)]
    if not gdf_filtered.empty:
        m = folium.Map(location=[gdf_filtered.geometry.y.mean(), gdf_filtered.geometry.x.mean()], zoom_start=6)
        for _, row in gdf_filtered.iterrows():
            folium.Marker([row.geometry.y, row.geometry.x], tooltip=f"{row['Nom_Est']}<br>{row['municipio']}").add_to(m)
        st_folium(m, width=900, height=600)
    else:
        st.warning("No hay estaciones seleccionadas para mostrar en el mapa.")

with tab3:
    st.header("Tabla de Estaciones")
    st.dataframe(gdf_stations[gdf_stations['Nom_Est'].isin(selected_stations)].drop(columns=['geometry']))

with tab4:
    st.header("Análisis de Precipitación y Fenómeno ENSO")
    df_analisis = df_monthly_filtered.copy()
    enso_col = next((col for col in df_analisis.columns if 'enso' in col), None)
    anomalia_col = next((col for col in df_analisis.columns if 'anomalia' in col), None)
    
    if enso_col:
        st.subheader("Precipitación Media por Fase ENSO")
        df_enso_group = df_analisis.groupby(enso_col)['Precipitation'].mean().reset_index()
        fig_enso = px.bar(df_enso_group, x=enso_col, y='Precipitation', color=enso_col)
        st.plotly_chart(fig_enso, use_container_width=True)
    
    if anomalia_col:
        st.subheader("Correlación con Anomalía ONI")
        df_analisis[anomalia_col] = pd.to_numeric(df_analisis[anomalia_col], errors='coerce')
        df_analisis.dropna(subset=[anomalia_col, 'Precipitation'], inplace=True)
        correlation = df_analisis[anomalia_col].corr(df_analisis['Precipitation'])
        st.metric("Coeficiente de Correlación", f"{correlation:.2f}")
