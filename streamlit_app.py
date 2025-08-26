# -*- coding: utf-8 -*-
"""
Visor de Información Geoespacial de Precipitación y ENSO
"""

import streamlit as st
import pandas as pd
import geopandas as gpd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import zipfile
import io
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

# =-===========================================================================
# Funciones de Carga y Procesamiento de Datos
# (Se usan decoradores @st.cache_data para optimizar el rendimiento)
# =============================================================================

@st.cache_data
def load_data(stations_path, precip_path, enso_path):
    """
    Carga, limpia y fusiona todos los datos tabulares desde archivos CSV.
    """
    try:
        # Cargar datos de las estaciones
        df_stations = pd.read_csv(stations_path, sep=';')
        
        # Cargar datos de precipitación
        df_precip = pd.read_csv(precip_path, sep=';')

        # Cargar datos de ENSO
        df_enso = pd.read_csv(enso_path, sep=';')

        # --- Limpieza y Transformación de Datos ---

        # 1. Transformar df_precip de formato ancho a largo
        df_precip_long = pd.melt(
            df_precip,
            id_vars=['Id_Fecha', 'año', 'mes'],
            var_name='Id_estacio',
            value_name='Precipitacion'
        )
        
        # Convertir Id_estacio a entero para que coincida con df_stations
        df_precip_long['Id_estacio'] = pd.to_numeric(df_precip_long['Id_estacio'], errors='coerce')
        df_precip_long.dropna(subset=['Id_estacio'], inplace=True)
        df_precip_long['Id_estacio'] = df_precip_long['Id_estacio'].astype(int)

        # 2. Limpiar y formatear df_enso
        df_enso['Id_Fecha'] = pd.to_datetime(df_enso['Id_Fecha'], format='%d/%m/%Y', errors='coerce')
        # Reemplazar comas por puntos y convertir a número
        for col in ['Anomalia_ONI', 'Temp_SST', 'Temp_media']:
            df_enso[col] = df_enso[col].str.replace(',', '.', regex=False).astype(float)
        
        # 3. Limpiar df_stations
        df_stations.rename(columns={'departamento': 'Departamento', 'municipio': 'Municipio'}, inplace=True)

        # 4. Fusionar (Merge) los DataFrames
        # Fusionar precipitación con información de estaciones
        df_merged = pd.merge(df_precip_long, df_stations, on='Id_estacio', how='left')
        
        # Convertir Id_Fecha en df_merged para la fusión con ENSO
        df_merged['Id_Fecha'] = pd.to_datetime(df_merged['Id_Fecha'], format='%d/%m/%Y', errors='coerce')
        
        # Fusionar con datos de ENSO
        df_final = pd.merge(df_merged, df_enso, on='Id_Fecha', how='left')

        # Eliminar filas donde la información clave es nula
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
    Transforma coordenadas planas (asumidas como MAGNA-SIRGAS / Origen-Nacional) a WGS 84.
    """
    # Definir la proyección de origen para Colombia (MAGNA-SIRGAS / Origen-Nacional)
    # EPSG:3116 es una opción común, pero podría ser otra. Ajustar si es necesario.
    # Basado en los valores (ej. Longitud ~ 4,817,943), es un sistema proyectado.
    # Este es el EPSG para MAGNA-SIRGAS / Colombia Bogota zone
    proj_origen = Proj('epsg:3116') 
    # Proyección de destino (WGS 84)
    proj_destino = Proj('epsg:4326')

    # Aplicar la transformación
    df['lon_wgs84'], df['lat_wgs84'] = transform(
        proj_origen,
        proj_destino,
        df[lon_col].values,
        df[lat_col].values
    )
    return df

@st.cache_data
def load_geodata(zip_path, df_stations):
    """
    Carga el shapefile, lo transforma a WGS 84 y lo fusiona con datos de estaciones.
    """
    try:
        with zipfile.ZipFile(zip_path, 'r') as z:
            # Encontrar el archivo .shp dentro del zip
            shp_filename = [name for name in z.namelist() if name.endswith('.shp')][0]
            # Leer el shapefile directamente desde el zip
            gdf = gpd.read_file(f'/vsizip/{zip_path}/{shp_filename}')
        
        # Renombrar columnas para evitar conflictos y estandarizar
        gdf.rename(columns={'Id_estacio': 'Id_estacio_shp'}, inplace=True)

        # Asumir que el shapefile está en el mismo CRS plano y transformar
        # Si el CRS ya está definido en el shapefile, geopandas podría leerlo. 
        # Si no, lo definimos manualmente.
        if gdf.crs is None:
            gdf.set_crs('epsg:3116', inplace=True) # Asignar el CRS de origen
        
        # Reproyectar a WGS 84 para visualización en mapas web
        gdf = gdf.to_crs('epsg:4326')
        
        return gdf
    except IndexError:
        st.error("Error: No se encontró un archivo .shp dentro del archivo ZIP.")
        return None
    except Exception as e:
        st.error(f"Error al cargar los datos geoespaciales: {e}")
        return None


# =============================================================================
# Carga Inicial de Datos
# =============================================================================
# Rutas a los archivos (asegúrate que estén en la misma carpeta)
STATIONS_CSV_PATH = 'mapaCVENSO.csv'
PRECIP_CSV_PATH = 'DatosPptnmes_ENSO.csv'
ENSO_CSV_PATH = 'ENSO_1950_2023.csv'
SHAPEFILE_ZIP_PATH = 'mapaCV.zip'

# Cargar los datos usando las funciones
data = load_data(STATIONS_CSV_PATH, PRECIP_CSV_PATH, ENSO_CSV_PATH)
geodata = load_geodata(SHAPEFILE_ZIP_PATH, data) if data is not None else None

if data is None:
    st.warning("La carga de datos falló. El tablero no puede continuar. Por favor, revisa los errores mostrados arriba.")
    st.stop() # Detiene la ejecución si los datos no se cargan

# Aplicar transformación de coordenadas al dataframe principal
data = transform_coords(data)

# =============================================================================
# Panel Lateral de Controles (Sidebar)
# =============================================================================
st.sidebar.image("https://i.imgur.com/KEXq15h.png", use_column_width=True)
st.sidebar.title("🛰️ Panel de Control")
st.sidebar.markdown("Usa los filtros para explorar los datos de precipitación y su relación con el fenómeno ENSO.")

# --- Filtros ---
# Filtro de Años
min_year, max_year = int(data['año'].min()), int(data['año'].max())
selected_years = st.sidebar.slider(
    "Selecciona un rango de años:",
    min_year, max_year,
    (min_year, max_year)
)

# Filtro de Departamento
sorted_depts = sorted(data['Departamento'].unique())
selected_dept = st.sidebar.selectbox(
    "Selecciona un Departamento:",
    options=sorted_depts,
    index=0 # Por defecto, selecciona el primero
)

# Filtro de Municipio (dependiente del departamento)
municipios_in_dept = sorted(data[data['Departamento'] == selected_dept]['Municipio'].unique())
selected_municipios = st.sidebar.multiselect(
    "Selecciona uno o más Municipios:",
    options=municipios_in_dept,
    default=municipios_in_dept[:3] # Por defecto selecciona los primeros 3
)

# Filtro de Estaciones (dependiente de los municipios)
if selected_municipios:
    stations_in_mun = sorted(data[data['Municipio'].isin(selected_municipios)]['Nom_Est'].unique())
    selected_stations = st.sidebar.multiselect(
        "Selecciona una o más Estaciones:",
        options=stations_in_mun,
        default=stations_in_mun[:5] # Por defecto selecciona las primeras 5
    )
else:
    selected_stations = []
    st.sidebar.warning("Por favor, selecciona al menos un municipio para ver las estaciones.")

# =============================================================================
# Lógica de Filtrado de Datos
# =============================================================================
filtered_data = data[
    (data['año'] >= selected_years[0]) &
    (data['año'] <= selected_years[1]) &
    (data['Departamento'] == selected_dept) &
    (data['Municipio'].isin(selected_municipios)) &
    (data['Nom_Est'].isin(selected_stations))
]

# =============================================================================
# Cuerpo Principal de la Aplicación
# =============================================================================
st.title("🌦️ Visor Geoespacial de Precipitación y ENSO en Colombia")
st.markdown("---")

if filtered_data.empty:
    st.warning("No se encontraron datos para la selección actual. Por favor, ajusta los filtros en el panel lateral.")
else:
    # --- Métricas Resumen ---
    col1, col2, col3 = st.columns(3)
    avg_precip = filtered_data['Precipitacion'].mean()
    max_precip_row = filtered_data.loc[filtered_data['Precipitacion'].idxmax()]

    col1.metric("Precipitación Promedio Mensual", f"{avg_precip:.2f} mm")
    col2.metric("Periodo Analizado", f"{selected_years[0]} - {selected_years[1]}")
    col3.metric("Estaciones Seleccionadas", len(selected_stations))
    
    # --- Contenedor de Pestañas ---
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Series de Tiempo", "🗺️ Mapa Interactivo", "📈 Correlación con ENSO", "📥 Datos Tabulares y Descarga"])

    with tab1:
        st.header("Análisis de Precipitación a lo Largo del Tiempo")
        
        # Gráfico de Serie de Tiempo Mensual
        st.subheader("Precipitación Mensual Promedio")
        monthly_avg = filtered_data.groupby('Id_Fecha')['Precipitacion'].mean().reset_index()
        fig_monthly = px.line(
            monthly_avg, 
            x='Id_Fecha', 
            y='Precipitacion', 
            title='Promedio Mensual de Precipitación de Estaciones Seleccionadas',
            labels={'Id_Fecha': 'Fecha', 'Precipitacion': 'Precipitación Promedio (mm)'},
            template='plotly_white'
        )
        fig_monthly.update_traces(line_color='#1f77b4')
        st.plotly_chart(fig_monthly, use_container_width=True)

        # Gráfico de Serie de Tiempo Anual
        st.subheader("Precipitación Anual Acumulada")
        annual_sum = filtered_data.groupby(['año', 'Nom_Est'])['Precipitacion'].sum().reset_index()
        fig_annual = px.line(
            annual_sum, 
            x='año', 
            y='Precipitacion', 
            color='Nom_Est',
            title='Precipitación Anual por Estación',
            labels={'año': 'Año', 'Precipitacion': 'Precipitación Anual (mm)', 'Nom_Est': 'Estación'},
            template='plotly_white'
        )
        st.plotly_chart(fig_annual, use_container_width=True)

    with tab2:
        st.header("Distribución Geográfica de Estaciones")

        # Filtrar datos para el mapa
        map_data = filtered_data[['Nom_Est', 'Municipio', 'lon_wgs84', 'lat_wgs84', 'Precipitacion']].copy()
        map_data_avg = map_data.groupby(['Nom_Est', 'Municipio', 'lon_wgs84', 'lat_wgs84'])['Precipitacion'].mean().reset_index()

        fig_map = px.scatter_mapbox(
            map_data_avg,
            lat="lat_wgs84",
            lon="lon_wgs84",
            size="Precipitacion",
            color="Precipitacion",
            hover_name="Nom_Est",
            hover_data={"Municipio": True, "Precipitacion": ":.2f mm
