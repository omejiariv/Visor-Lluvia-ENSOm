# -*- coding: utf-8 -*-
"""
Visor de Información Geoespacial de Precipitación y ENSO (Versión Final)
"""

import streamlit as st
import pandas as pd
import geopandas as gpd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import zipfile
import pyproj
from pyproj import Proj, transform

# =============================================================================
# Configuración de la página
# =============================================================================
st.set_page_config(
    page_title="Visor de Datos Climáticos: Precipitación y ENSO",
    page_icon="🌦️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =============================================================================
# Funciones de Carga y Procesamiento de Datos
# =============================================================================

@st.cache_data
def load_data(stations_path, precip_path, enso_path):
    """
    Carga, limpia y fusiona todos los datos tabulares desde archivos CSV.
    """
    try:
        df_stations = pd.read_csv(stations_path, sep=';')
        df_precip = pd.read_csv(precip_path, sep=';')
        df_enso = pd.read_csv(enso_path, sep=';')

        # --- Limpieza y Transformación ---
        df_precip_long = pd.melt(
            df_precip,
            id_vars=['Id_Fecha', 'año', 'mes'],
            var_name='Id_estacio',
            value_name='Precipitacion'
        )
        df_precip_long['Id_estacio'] = pd.to_numeric(df_precip_long['Id_estacio'], errors='coerce')
        df_precip_long.dropna(subset=['Id_estacio'], inplace=True)
        df_precip_long['Id_estacio'] = df_precip_long['Id_estacio'].astype(int)

        df_enso['Id_Fecha'] = pd.to_datetime(df_enso['Id_Fecha'], format='%d/%m/%Y', errors='coerce')
        for col in ['Anomalia_ONI', 'Temp_SST', 'Temp_media']:
            df_enso[col] = df_enso[col].str.replace(',', '.', regex=False).astype(float)
        
        df_stations.rename(columns={'departamento': 'Departamento', 'municipio': 'Municipio'}, inplace=True)

        df_merged = pd.merge(df_precip_long, df_stations, on='Id_estacio', how='left')
        df_merged['Id_Fecha'] = pd.to_datetime(df_merged['Id_Fecha'], format='%d/%m/%Y', errors='coerce')
        df_final = pd.merge(df_merged, df_enso, on='Id_Fecha', how='left')
        df_final.dropna(subset=['Departamento', 'Municipio', 'Precipitacion', 'Id_Fecha'], inplace=True)
        
        return df_final

    except FileNotFoundError as e:
        st.error(f"Error: No se encontró el archivo {e.filename}. Asegúrate de que los archivos CSV estén en la misma carpeta que el script.")
        return None
    except Exception as e:
        st.error(f"Ocurrió un error inesperado al cargar los datos: {e}")
        return None

@st.cache_data
def transform_coords(df, lon_col='Longitud', lat_col='Latitud'):
    """
    Transforma coordenadas planas (MAGNA-SIRGAS / Origen-Nacional) a WGS 84.
    """
    proj_origen = Proj('epsg:3116') 
    proj_destino = Proj('epsg:4326')
    df['lon_wgs84'], df['lat_wgs84'] = transform(
        proj_origen,
        proj_destino,
        df[lon_col].values,
        df[lat_col].values
    )
    return df

@st.cache_data
def load_geodata(zip_path):
    """ Carga el shapefile desde un ZIP y lo reproyecta a WGS 84. """
    try:
        with zipfile.ZipFile(zip_path, 'r') as z:
            shp_filename = [name for name in z.namelist() if name.endswith('.shp')][0]
            gdf = gpd.read_file(f'/vsizip/{zip_path}/{shp_filename}')
        
        if gdf.crs is None:
            gdf.set_crs('epsg:3116', inplace=True)
        
        gdf = gdf.to_crs('epsg:4326')
        return gdf
    except Exception as e:
        st.error(f"Error al cargar los datos geoespaciales: {e}")
        return None

# =============================================================================
# Carga Inicial de Datos
# =============================================================================
STATIONS_CSV_PATH = 'mapaCVENSO.csv'
PRECIP_CSV_PATH = 'DatosPptnmes_ENSO.csv'
ENSO_CSV_PATH = 'ENSO_1950_2023.csv'
SHAPEFILE_ZIP_PATH = 'mapaCV.zip'

data = load_data(STATIONS_CSV_PATH, PRECIP_CSV_PATH, ENSO_CSV_PATH)

if data is None:
    st.warning("La carga de datos tabulares falló. El tablero no puede continuar.")
    st.stop()

data = transform_coords(data)

# =============================================================================
# Panel Lateral (Sidebar)
# =============================================================================
st.sidebar.image("https://i.imgur.com/KEXq15h.png", use_column_width=True)
st.sidebar.title("🛰️ Panel de Control")
st.sidebar.markdown("Usa los filtros para explorar los datos.")

min_year, max_year = int(data['año'].min()), int(data['año'].max())
selected_years = st.sidebar.slider("Rango de años:", min_year, max_year, (min_year, max_year))

sorted_depts = sorted(data['Departamento'].unique())
selected_dept = st.sidebar.selectbox("Departamento:", options=sorted_depts)

municipios_in_dept = sorted(data[data['Departamento'] == selected_dept]['Municipio'].unique())
selected_municipios = st.sidebar.multiselect("Municipios:", options=municipios_in_dept, default=municipios_in_dept[:3])

if selected_municipios:
    stations_in_mun = sorted(data[data['Municipio'].isin(selected_municipios)]['Nom_Est'].unique())
    selected_stations = st.sidebar.multiselect("Estaciones:", options=stations_in_mun, default=stations_in_mun[:5])
else:
    selected_stations = []
    st.sidebar.warning("Selecciona al menos un municipio.")

# =============================================================================
# Lógica de Filtrado
# =============================================================================
filtered_data = data[
    (data['año'].between(selected_years[0], selected_years[1])) &
    (data['Departamento'] == selected_dept) &
    (data['Municipio'].isin(selected_municipios)) &
    (data['Nom_Est'].isin(selected_stations))
]

# =============================================================================
# Cuerpo Principal
# =============================================================================
st.title("🌦️ Visor Geoespacial de Precipitación y ENSO en Colombia")
st.markdown("---")

if filtered_data.empty:
    st.warning("No se encontraron datos para la selección actual. Por favor, ajusta los filtros.")
else:
    # --- Métricas ---
    col1, col2, col3 = st.columns(3)
    avg_precip = filtered_data['Precipitacion'].mean()
    col1.metric("Precipitación Promedio Mensual", f"{avg_precip:.2f} mm")
    col2.metric("Periodo Analizado", f"{selected_years[0]} - {selected_years[1]}")
    col3.metric("Estaciones Seleccionadas", len(selected_stations))
    
    # --- Pestañas ---
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Series de Tiempo", "🗺️ Mapa Interactivo", "📈 Correlación con ENSO", "📥 Datos y Descarga"])

    with tab1:
        st.header("Análisis de Precipitación a lo Largo del Tiempo")
        monthly_avg = filtered_data.groupby('Id_Fecha')['Precipitacion'].mean().reset_index()
        fig_monthly = px.line(monthly_avg, x='Id_Fecha', y='Precipitacion', title='Promedio Mensual de Precipitación', labels={'Id_Fecha': 'Fecha', 'Precipitacion': 'Precipitación (mm)'})
        st.plotly_chart(fig_monthly, use_container_width=True)
        
        annual_sum = filtered_data.groupby(['año', 'Nom_Est'])['Precipitacion'].sum().reset_index()
        fig_annual = px.line(annual_sum, x='año', y='Precipitacion', color='Nom_Est', title='Precipitación Anual por Estación', labels={'año': 'Año', 'Precipitacion': 'Precipitación (mm)'})
        st.plotly_chart(fig_annual, use_container_width=True)

    with tab2:
        st.header("Distribución Geográfica de Estaciones")
        map_data_avg = filtered_data.groupby(['Nom_Est', 'Municipio', 'lon_wgs84', 'lat_wgs84'])['Precipitacion'].mean().reset_index()
        fig_map = px.scatter_mapbox(
            map_data_avg, lat="lat_wgs84", lon="lon_wgs84", size="Precipitacion", color="Precipitacion",
            hover_name="Nom_Est", hover_data={"Municipio": True, "Precipitacion": ":.2f mm"},
            color_continuous_scale=px.colors.sequential.Viridis, size_max=20, zoom=5,
            mapbox_style="open-street-map", title="Ubicación y Precipitación Promedio de Estaciones"
        )
        fig_map.update_layout(mapbox_center={"lat": 4.5709, "lon": -74.2973}, margin={"r":0,"t":40,"l":0,"b":0})
        st.plotly_chart(fig_map, use_container_width=True)
    
    with tab3:
        st.header("Análisis de Correlación: Precipitación vs. ENSO")
        enso_corr_data = filtered_data.groupby('Id_Fecha').agg({'Precipitacion': 'mean', 'Anomalia_ONI': 'first'}).reset_index()
        fig_corr = make_subplots(specs=[[{"secondary_y": True}]])
        fig_corr.add_trace(go.Bar(x=enso_corr_data['Id_Fecha'], y=enso_corr_data['Precipitacion'], name='Precipitación', marker_color='skyblue'), secondary_y=False)
        fig_corr.add_trace(go.Scatter(x=enso_corr_data['Id_Fecha'], y=enso_corr_data['Anomalia_ONI'], name='Anomalía ONI', line=dict(color='red', width=2)), secondary_y=True)
        fig_corr.add_hline(y=0.5, line_dash="dot", line_color="orange", secondary_y=True, annotation_text="El Niño")
        fig_corr.add_hline(y=-0.5, line_dash="dot", line_color="blue", secondary_y=True, annotation_text="La Niña")
        fig_corr.update_layout(title_text='Precipitación Mensual vs. Anomalía ONI', legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
        fig_corr.update_yaxes(title_text="Precipitación (mm)", secondary_y=False)
        fig_corr.update_yaxes(title_text="Anomalía ONI (°C)", secondary_y=True)
        st.plotly_chart(fig_corr, use_container_width=True)

        st.subheader("Animación de Precipitación Anual")
        annual_map_data = filtered_data.groupby(['año', 'Nom_Est', 'lon_wgs84', 'lat_wgs84'])['Precipitacion'].sum().reset_index()
        fig_animated = px.scatter_mapbox(
            annual_map_data, lat="lat_wgs84", lon="lon_wgs84", size="Precipitacion", color="Precipitacion",
            animation_frame="año", animation_group="Nom_Est", hover_name="Nom_Est",
            color_continuous_scale=px.colors.sequential.YlOrRd, size_max=25, zoom=5,
            mapbox_style="carto-positron", title="Evolución Anual de la Precipitación"
        )
        fig_animated.update_layout(mapbox_center={"lat": 4.5709, "lon": -74.2973}, margin={"r":0,"t":40,"l":0,"b":0})
        st.plotly_chart(fig_animated, use_container_width=True)

    with tab4:
        st.header("Datos Filtrados y Descarga")
        st.dataframe(filtered_data.drop(columns=['lon_wgs84', 'lat_wgs84']))
        
        @st.cache_data
        def convert_df_to_csv(df):
            return df.to_csv(index=False, sep=';', encoding='utf-8-sig').encode('utf-8-sig')

        csv_file = convert_df_to_csv(filtered_data)
        st.download_button(
            label="📥 Descargar datos como CSV",
            data=csv_file,
            file_name=f"datos_filtrados_{selected_dept}.csv",
            mime="text/csv",
        )
