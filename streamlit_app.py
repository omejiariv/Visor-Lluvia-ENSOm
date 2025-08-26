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
from datetime import datetime
from shapely.geometry import Point
import requests
import json
from pyproj import Transformer
import subprocess
import sys

# --- Manejo de dependencias ---
# Este bloque de código asegura que las bibliotecas necesarias estén instaladas.
# Si el script falla, intentará instalar las dependencias faltantes.
try:
    import folium
    import geopandas
    import pyproj
    import requests
    from streamlit_folium import folium_static
except ImportError:
    st.warning("Se están instalando las bibliotecas requeridas. Por favor, espera y la aplicación se recargará automáticamente.")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "folium", "geopandas", "pyproj", "requests"])
        st.experimental_rerun()
    except Exception as e:
        st.error(f"Error al instalar las dependencias: {e}")
        st.stop()

# --- Configuración de la página ---
st.set_page_config(layout="wide", page_title="Visor de Precipitación y ENSO", page_icon="☔")

# --- URLs de GitHub para la carga automática de datos ---
GITHUB_BASE_URL = "https://raw.githubusercontent.com/Juan-Diego-Gaviria/Datos-ENSO-Python/main/"
SHAPEFILE_URL = "https://github.com/Juan-Diego-Gaviria/Datos-ENSO-Python/raw/main/mapaCV.zip"

# --- Funciones de Carga y Preprocesamiento de Datos ---

@st.cache_data
def load_data(url, sep=';'):
    """Carga un archivo CSV desde una URL y lo almacena en caché."""
    try:
        df = pd.read_csv(url, sep=sep)
        st.success(f"Datos cargados correctamente desde: {url}")
        return df
    except Exception as e:
        st.error(f"Error al cargar los datos desde {url}: {e}")
        return None

def download_file_from_github(url, file_path):
    """Descarga un archivo desde GitHub y lo guarda localmente."""
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()
        with open(file_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(f)
        st.success(f"Archivo descargado: {os.path.basename(file_path)}")
        return True
    except Exception as e:
        st.error(f"Error al descargar {url}: {e}")
        return False

def load_shapefile(url):
    """Descarga un archivo zip de un shapefile y lo carga como un GeoDataFrame."""
    try:
        response = requests.get(url)
        response.raise_for_status()
        
        # Guardar el zip en un archivo temporal
        with tempfile.TemporaryDirectory() as temp_dir:
            zip_path = os.path.join(temp_dir, 'shapefile.zip')
            with open(zip_path, 'wb') as f:
                f.write(response.content)
            
            # Descomprimir y leer el shapefile
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(temp_dir)
            
            # Buscar el archivo .shp dentro del directorio extraído
            shp_file = None
            for file in os.listdir(temp_dir):
                if file.endswith(".shp"):
                    shp_file = os.path.join(temp_dir, file)
                    break

            if shp_file:
                # Usar el controlador de GeoPandas adecuado (e.g., fiona)
                gdf = gpd.read_file(shp_file, encoding='utf-8')
                st.success("Shapefile cargado correctamente.")
                return gdf
            else:
                st.error("Archivo .shp no encontrado en el zip.")
                return None
    except Exception as e:
        st.error(f"Error al cargar el shapefile: {e}")
        return None

def transform_coordinates(gdf, x_col, y_col, epsg_from, epsg_to):
    """
    Transforma coordenadas de un GeoDataFrame.
    Utiliza Pyproj para la transformación.
    """
    # Usar Pyproj Transformer para la transformación
    transformer = Transformer.from_crs(f"EPSG:{epsg_from}", f"EPSG:{epsg_to}", always_xy=True)
    
    # Crear nuevas columnas para longitud y latitud en WGS84
    lon_wgs84 = []
    lat_wgs84 = []
    
    for x, y in zip(gdf[x_col], gdf[y_col]):
        lon, lat = transformer.transform(x, y)
        lon_wgs84.append(lon)
        lat_wgs84.append(lat)
        
    gdf['Longitud_WGS84'] = lon_wgs84
    gdf['Latitud_WGS84'] = lat_wgs84
    
    st.success(f"Coordenadas transformadas de EPSG:{epsg_from} a EPSG:{epsg_to} (WGS84).")
    return gdf

# --- Mapeo de columnas flexible ---
def map_columns(df, required_cols):
    """Mapea columnas basándose en palabras clave para hacer el código más robusto."""
    col_map = {}
    df_cols = [col.lower() for col in df.columns]

    for req_col, keywords in required_cols.items():
        found = False
        for keyword in keywords:
            for col in df.columns:
                if keyword in col.lower():
                    col_map[req_col] = col
                    found = True
                    break
            if found:
                break
        if not found:
            st.warning(f"No se encontró una columna para '{req_col}' usando las palabras clave: {keywords}. Algunas funciones pueden no estar disponibles.")
    
    # Manejar el caso especial de las columnas de año
    year_cols = [col for col in df.columns if col.isdigit() and int(col) >= 1970 and int(col) <= 2021]
    if year_cols:
        col_map['year_data'] = year_cols
    else:
        st.warning("No se encontraron columnas de años para los datos de precipitación.")

    return col_map

# --- Main app logic ---
def main():
    st.title("☔ Visor de Información Geoespacial de Precipitación y el Fenómeno ENSO")
    st.markdown("### Un tablero interactivo para el análisis de datos climáticos")
    st.markdown("---")

    # --- Sidebar para la carga de datos y filtros ---
    st.sidebar.header("Carga de Datos")
    load_option = st.sidebar.radio(
        "Selecciona una opción de carga de datos:",
        ("Cargar desde GitHub (Automático)", "Subir mis propios archivos")
    )
    
    # Variables globales para los DataFrames
    df_estaciones = None
    df_precipitacion = None
    df_enso = None
    gdf_municipios = None

    if load_option == "Cargar desde GitHub (Automático)":
        with st.sidebar.spinner("Cargando datos desde GitHub..."):
            try:
                # Cargar datos de estaciones
                df_estaciones = load_data(GITHUB_BASE_URL + "mapaCVENSO.csv")
                
                # Cargar datos de precipitación
                df_precipitacion = load_data(GITHUB_BASE_URL + "DatosPptnmes_ENSO.csv")
                
                # Cargar datos ENSO
                df_enso = load_data(GITHUB_BASE_URL + "ENSO_1950_2023.csv")
                
                # Cargar shapefile
                st.info("Descargando y extrayendo shapefile... Esto puede tomar un momento.")
                gdf_municipios = load_shapefile(SHAPEFILE_URL)
                
            except Exception as e:
                st.error(f"Ocurrió un error inesperado al cargar los datos desde GitHub: {e}")
                
    else: # Subir archivos manual
        st.sidebar.subheader("Subir archivos (separador: ';')")
        uploaded_estaciones = st.sidebar.file_uploader("Subir archivo de Estaciones (.csv)", type=["csv"])
        uploaded_precipitacion = st.sidebar.file_uploader("Subir archivo de Precipitación (.csv)", type=["csv"])
        uploaded_enso = st.sidebar.file_uploader("Subir archivo de ENSO (.csv)", type=["csv"])
        uploaded_shapefile = st.sidebar.file_uploader("Subir archivo de Municipios (.zip)", type=["zip"])
        
        if uploaded_estaciones:
            df_estaciones = pd.read_csv(uploaded_estaciones, sep=';')
        if uploaded_precipitacion:
            df_precipitacion = pd.read_csv(uploaded_precipitacion, sep=';')
        if uploaded_enso:
            df_enso = pd.read_csv(uploaded_enso, sep=';')
        if uploaded_shapefile:
            st.info("Descomprimiendo y cargando shapefile...")
            try:
                with zipfile.ZipFile(uploaded_shapefile, 'r') as zip_ref:
                    temp_dir = tempfile.mkdtemp()
                    zip_ref.extractall(temp_dir)
                    shp_file = [f for f in os.listdir(temp_dir) if f.endswith('.shp')][0]
                    gdf_municipios = gpd.read_file(os.path.join(temp_dir, shp_file))
                    st.success("Shapefile cargado correctamente.")
            except Exception as e:
                st.error(f"Error al cargar el archivo zip del shapefile: {e}")

    # Continuar solo si se han cargado los datos principales
    if df_estaciones is not None and df_precipitacion is not None and df_enso is not None:
        
        # Mapeo de columnas
        map_estaciones = map_columns(df_estaciones, {'id_estacion': ['id_estacio', 'id'], 'nombre_estacion': ['nom_est', 'nombre'], 'longitud': ['longitud', 'x'], 'latitud': ['latitud', 'y'], 'departamento': ['departamento', 'depto'], 'municipio': ['municipio', 'mun']})
        map_precipitacion = map_columns(df_precipitacion, {'id_fecha': ['id_fecha', 'fecha'], 'year': ['año', 'year'], 'mes': ['mes', 'month']})
        map_enso = map_columns(df_enso, {'id_fecha': ['id_fecha', 'fecha'], 'year': ['year', 'año'], 'mes': ['mes', 'month'], 'anomalia_oni': ['anomalia_oni', 'oni'], 'enso_class': ['enso', 'estado']})
        
        # Preprocesamiento de datos de estaciones
        df_estaciones.rename(columns={
            map_estaciones.get('id_estacion'): 'Id_estacion',
            map_estaciones.get('nombre_estacion'): 'Nom_Est'
        }, inplace=True)
        
        # Coordenadas planas a WGS84
        if 'Longitud_WGS84' not in df_estaciones.columns and 'Latitud_WGS84' not in df_estaciones.columns:
            st.sidebar.info("Transformando coordenadas planas a WGS84...")
            try:
                gdf_estaciones_temp = gpd.GeoDataFrame(
                    df_estaciones,
                    geometry=gpd.points_from_xy(df_estaciones[map_estaciones.get('longitud')], df_estaciones[map_estaciones.get('latitud')]),
                    crs="EPSG:3115"  # Asumiendo Magna-Sirgas como coordenadas planas
                )
                gdf_estaciones_temp = gdf_estaciones_temp.to_crs("EPSG:4326") # WGS84
                df_estaciones['Longitud_WGS84'] = gdf_estaciones_temp.geometry.x
                df_estaciones['Latitud_WGS84'] = gdf_estaciones_temp.geometry.y
                st.sidebar.success("Transformación de coordenadas completa.")
            except Exception as e:
                st.sidebar.error(f"Error en la transformación de coordenadas: {e}")
                
        # Preparar datos de precipitación
        df_precipitacion_melted = pd.melt(df_precipitacion, 
                                          id_vars=[map_precipitacion.get('id_fecha'), map_precipitacion.get('year'), map_precipitacion.get('mes')],
                                          var_name='Id_estacion', 
                                          value_name='Precipitacion')
        
        # Eliminar las filas con valores nulos o '0' en la precipitación
        df_precipitacion_melted['Precipitacion'] = pd.to_numeric(df_precipitacion_melted['Precipitacion'], errors='coerce')
        df_precipitacion_melted.dropna(subset=['Precipitacion'], inplace=True)
        df_precipitacion_melted = df_precipitacion_melted[df_precipitacion_melted['Precipitacion'] > 0]
        
        # Fusionar datos de precipitación y estaciones
        df_combined = pd.merge(df_precipitacion_melted, df_estaciones, on='Id_estacion', how='inner')
        df_combined['Año'] = df_combined['año'].astype(int)
        
        # Preparar datos ENSO
        df_enso['Anomalia_ONI'] = pd.to_numeric(df_enso[map_enso.get('anomalia_oni')], errors='coerce')
        df_enso['ENSO'] = df_enso[map_enso.get('enso_class')].apply(lambda x: 'Niño' if 'Niño' in x else 'Niña' if 'Niña' in x else 'Neutral')
        df_enso['Year'] = df_enso[map_enso.get('year')].astype(int)

        # Controles en el sidebar
        st.sidebar.header("Filtros de Análisis")
        
        all_estaciones = sorted(df_combined['Nom_Est'].unique())
        selected_estaciones = st.sidebar.multiselect(
            "Selecciona Estaciones:",
            all_estaciones,
            default=all_estaciones[:5]
        )
        
        min_year, max_year = int(df_combined['Año'].min()), int(df_combined['Año'].max())
        year_range = st.sidebar.slider(
            "Rango de Años:",
            min_value=min_year,
            max_value=max_year,
            value=(min_year, max_year)
        )
        
        all_meses = sorted(df_combined['mes'].unique())
        selected_meses = st.sidebar.multiselect(
            "Selecciona Meses:",
            all_meses,
            default=all_meses
        )

        # Aplicar filtros
        df_filtered = df_combined[
            (df_combined['Nom_Est'].isin(selected_estaciones)) &
            (df_combined['Año'] >= year_range[0]) &
            (df_combined['Año'] <= year_range[1]) &
            (df_combined['mes'].isin(selected_meses))
        ]
        
        # Pestañas para organizar el contenido
        tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(["📊 Datos Brutos", "📈 Gráficos", "🗺️ Mapas", "🎥 Mapa Animado", "☀️ Análisis ENSO", "📥 Descargas"])

        # --- Pestaña 1: Datos Brutos ---
        with tab1:
            st.header("Datos Brutos de Precipitación")
            st.write(df_filtered)

            st.header("Datos de Estaciones")
            st.write(df_estaciones)
            
            st.header("Datos ENSO")
            st.write(df_enso)
            
        # --- Pestaña 2: Gráficos ---
        with tab2:
            st.header("Gráficos de Precipitación")

            if not df_filtered.empty:
                # Gráfico de Serie de Tiempo Anual (usando Altair)
                st.subheader("Precipitación Anual por Estación")
                df_anual = df_filtered.groupby(['Año', 'Nom_Est'])['Precipitacion'].sum().reset_index()
                
                chart_anual = alt.Chart(df_anual).mark_line().encode(
                    x=alt.X('Año:O', title='Año', axis=alt.Axis(format='d')),
                    y=alt.Y('Precipitacion:Q', title='Precipitación Anual Total (mm)'),
                    color='Nom_Est:N',
                    tooltip=['Año:O', 'Nom_Est:N', 'Precipitacion:Q']
                ).properties(
                    title="Precipitación Anual Total"
                ).interactive()
                st.altair_chart(chart_anual, use_container_width=True)

                # Gráfico de Serie de Tiempo Mensual (usando Altair)
                st.subheader("Precipitación Mensual por Estación")
                df_mensual = df_filtered.groupby(['Año', 'mes', 'Nom_Est'])['Precipitacion'].sum().reset_index()
                df_mensual['Fecha'] = pd.to_datetime(df_mensual['Año'].astype(str) + '-' + df_mensual['mes'].apply(lambda x: datetime.strptime(x, '%b').strftime('%m')))
                
                chart_mensual = alt.Chart(df_mensual).mark_line().encode(
                    x=alt.X('Fecha:T', title='Fecha'),
                    y=alt.Y('Precipitacion:Q', title='Precipitación Mensual Total (mm)'),
                    color='Nom_Est:N',
                    tooltip=[alt.Tooltip('Fecha:T', format='%Y-%m'), 'Nom_Est:N', 'Precipitacion:Q']
                ).properties(
                    title="Precipitación Mensual Total"
                ).interactive()
                st.altair_chart(chart_mensual, use_container_width=True)
            else:
                st.warning("No hay datos para mostrar con los filtros seleccionados.")

        # --- Pestaña 3: Mapas ---
        with tab3:
            st.header("Mapa Interactivo de Estaciones")
            if not df_filtered.empty and 'Longitud_WGS84' in df_filtered.columns and 'Latitud_WGS84' in df_filtered.columns:
                
                # Crear el mapa base de Folium
                map_center = [df_filtered['Latitud_WGS84'].mean(), df_filtered['Longitud_WGS84'].mean()]
                m = folium.Map(location=map_center, zoom_start=6, tiles='CartoDB Positron')
                
                # Agregar los municipios (si se cargó el shapefile)
                if gdf_municipios is not None:
                    try:
                        folium.GeoJson(
                            gdf_municipios.to_json(),
                            name='Municipios',
                            style_function=lambda x: {'fillColor': 'grey', 'color': 'black', 'weight': 0.5, 'fillOpacity': 0.1}
                        ).add_to(m)
                    except Exception as e:
                        st.error(f"Error al agregar el GeoJson de municipios: {e}")
                
                # Agregar marcadores de las estaciones
                for index, row in df_filtered.iterrows():
                    folium.Marker(
                        location=[row['Latitud_WGS84'], row['Longitud_WGS84']],
                        popup=f"Estación: {row['Nom_Est']}<br>Precipitación: {row['Precipitacion']} mm",
                        icon=folium.Icon(color="blue", icon="cloud", prefix='fa')
                    ).add_to(m)
                
                # Mostrar el mapa
                folium_static(m)
            else:
                st.warning("No hay datos de estaciones con coordenadas válidas para mostrar en el mapa.")

        # --- Pestaña 4: Mapa Animado (Plotly) ---
        with tab4:
            st.header("Mapa Animado de Precipitación Anual")
            if not df_filtered.empty and 'Longitud_WGS84' in df_filtered.columns and 'Latitud_WGS84' in df_filtered.columns:
                
                # Agregar columna 'text' para el tooltip
                df_anual = df_filtered.groupby(['Año', 'Nom_Est', 'Latitud_WGS84', 'Longitud_WGS84'])['Precipitacion'].sum().reset_index()
                df_anual['text'] = df_anual['Nom_Est'] + '<br>Precipitación: ' + df_anual['Precipitacion'].astype(str) + ' mm'
                
                # Crear el mapa animado con Plotly Express
                fig = px.scatter_geo(df_anual, 
                                     lat="Latitud_WGS84", 
                                     lon="Longitud_WGS84",
                                     color="Precipitacion",
                                     hover_name="Nom_Est",
                                     animation_frame="Año",
                                     size="Precipitacion",
                                     projection="natural earth",
                                     title="Precipitación Anual de las Estaciones a lo largo del Tiempo",
                                     hover_data={'Nom_Est': False, 'Precipitacion': ':.2f', 'Latitud_WGS84': ':.2f', 'Longitud_WGS84': ':.2f'})
                
                fig.update_geos(fitbounds="locations", visible=False)
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("No hay datos suficientes o válidos para crear el mapa animado.")

        # --- Pestaña 5: Análisis ENSO ---
        with tab5:
            st.header("Análisis de Correlación ENSO - Precipitación")
            
            # Unir datos de precipitación (agrupados por año y mes) con datos ENSO
            df_precip_monthly_total = df_filtered.groupby(['Año', 'mes'])['Precipitacion'].sum().reset_index()
            df_enso_precip = pd.merge(df_precip_monthly_total, df_enso, on=['Año', 'mes'], how='inner')
            
            if not df_enso_precip.empty:
                # Correlación de Pearson
                correlation = df_enso_precip['Anomalia_ONI'].corr(df_enso_precip['Precipitacion'])
                
                st.metric(label="Coeficiente de Correlación (Anomalía ONI vs. Precipitación)", value=f"{correlation:.2f}")
                st.info("Un coeficiente de correlación cercano a 1 indica una fuerte relación positiva, mientras que uno cercano a -1 indica una fuerte relación negativa. Un valor cercano a 0 indica poca o ninguna relación lineal.")
                
                # Gráfico de barras que compara precipitación con eventos ENSO
                st.subheader("Precipitación Promedio por Fenómeno ENSO")
                df_enso_avg = df_enso_precip.groupby('ENSO')['Precipitacion'].mean().reset_index()
                fig_enso = px.bar(df_enso_avg, 
                                  x='ENSO', 
                                  y='Precipitacion',
                                  color='ENSO',
                                  title="Precipitación Promedio según el Fenómeno ENSO",
                                  labels={'Precipitacion': 'Precipitación Promedio (mm)', 'ENSO': 'Fenómeno ENSO'})
                st.plotly_chart(fig_enso, use_container_width=True)

            else:
                st.warning("No se encontraron datos coincidentes para el análisis ENSO. Asegúrate de que los años y meses de los archivos de precipitación y ENSO coincidan.")
        
        # --- Pestaña 6: Descargas ---
        with tab6:
            st.header("Descargar Datos Filtrados")
            st.markdown("Haz clic en los botones para descargar los datos filtrados en formato CSV.")
            
            # Descargar datos de precipitación filtrados
            csv_precip = df_filtered.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="Descargar Precipitación Filtrada",
                data=csv_precip,
                file_name='precipitacion_filtrada.csv',
                mime='text/csv'
            )
            
            # Descargar datos de estaciones filtrados
            csv_estaciones = df_estaciones[df_estaciones['Nom_Est'].isin(selected_estaciones)].to_csv(index=False).encode('utf-8')
            st.download_button(
                label="Descargar Estaciones Filtradas",
                data=csv_estaciones,
                file_name='estaciones_filtradas.csv',
                mime='text/csv'
            )
            
    else:
        st.info("Por favor, carga los archivos necesarios para continuar.")

if __name__ == "__main__":
    main()
