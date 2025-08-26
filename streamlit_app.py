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
import base64
from pykrige.ok import OrdinaryKriging
from pykrige.kriging_tools import write_asc_grid
from scipy.interpolate import griddata

# --- VERIFICACIÓN TEMPORAL DE LA VERSIÓN DE PLOTLY ---
try:
    import plotly
    st.write(f"Versión de Plotly instalada: **{plotly.__version__}**")
except Exception as e:
    st.write(f"No se pudo verificar la versión de Plotly: {e}")

# --- Configuración de la página ---
st.set_page_config(layout="wide", page_title="Visor de Precipitación y ENSO", page_icon="☔")

# Aplicar CSS personalizado para reducir el tamaño del texto y optimizar el espacio
st.markdown("""
<style>
    .sidebar .sidebar-content {
        font-size: 13px;
    }
    .stSelectbox label, .stMultiSelect label, .stSlider label {
        font-size: 13px !important;
    }
    .stMultiSelect div[data-baseweb="select"] {
        font-size: 13px !important;
    }
    .stSlider label {
        font-size: 13px !important;
    }
    .css-1d391kg {
        font-size: 13px;
    }
    .css-1cpx93x {
        font-size: 13px;
    }
    h1 {
        margin-top: 0px;
        padding-top: 0px;
    }
</style>
""", unsafe_allow_html=True)

# --- Funciones de carga de datos con caché ---
@st.cache_data
def load_and_preprocess_data(file_mapa, file_enso, file_precip, file_shapefile):
    """
    Carga y preprocesa todos los datos necesarios para la aplicación.
    La función se ejecuta solo una vez gracias a @st.cache_data.
    """
    
    # Función interna para cargar CSV
    def load_csv(file_path, sep=';'):
        if file_path is None: return None
        try:
            content = file_path.getvalue()
            if not content.strip(): return None
            encodings_to_try = ['utf-8', 'latin1', 'cp1252', 'iso-8859-1']
            for encoding in encodings_to_try:
                try:
                    df = pd.read_csv(io.BytesIO(content), sep=sep, encoding=encoding)
                    df.columns = df.columns.str.strip()
                    return df
                except (pd.errors.EmptyDataError, UnicodeDecodeError): continue
            st.error("No se pudo decodificar el archivo.")
            return None
        except Exception as e:
            st.error(f"Error al leer el archivo: {e}")
            return None

    # Función interna para cargar shapefile
    def load_shp(file_path):
        if file_path is None: return None
        try:
            with tempfile.TemporaryDirectory() as temp_dir:
                with zipfile.ZipFile(file_path, 'r') as zip_ref:
                    zip_ref.extractall(temp_dir)
                shp_path_list = [f for f in os.listdir(temp_dir) if f.endswith('.shp')]
                if not shp_path_list:
                    return None
                shp_path = shp_path_list[0]
                gdf = gpd.read_file(os.path.join(temp_dir, shp_path))
                gdf.columns = gdf.columns.str.strip()
                if gdf.crs is None: gdf.set_crs("EPSG:9377", inplace=True)
                gdf = gdf.to_crs("EPSG:4326")
                return gdf
        except Exception as e:
            st.error(f"Error al procesar el shapefile: {e}")
            return None

    # Carga de datos
    df_precip_anual = load_csv(file_mapa)
    df_enso = load_csv(file_enso)
    df_precip_mensual = load_csv(file_precip)
    gdf = load_shp(file_shapefile)
    
    # Validacion para evitar el ValueError
    if df_precip_anual is None or df_precip_anual.empty or \
       df_enso is None or df_enso.empty or \
       df_precip_mensual is None or df_precip_mensual.empty or \
       gdf is None or gdf.empty:
        st.error("Uno o más archivos no se cargaron correctamente o están vacíos. Por favor, intente de nuevo con los archivos originales.")
        return None, None, None, None, None, None
    
    # Preprocesamiento de datos de ENSO
    try:
        for col in ['Anomalia_ONI', 'Temp_SST', 'Temp_media']:
            if col in df_enso.columns and pd.api.types.is_object_dtype(df_enso[col]):
                df_enso[col] = df_enso[col].str.replace(',', '.', regex=True).astype(float)
        meses_es_en = {'ene': 'Jan', 'feb': 'Feb', 'mar': 'Mar', 'abr': 'Apr', 'may': 'May', 'jun': 'Jun', 'jul': 'Jul', 'ago': 'Aug', 'sep': 'Sep', 'oct': 'Oct', 'nov': 'Nov', 'dic': 'Dec'}
        df_enso['Year'] = df_enso['Year'].astype(int)
        df_enso['mes_en'] = df_enso['mes'].str.lower().map(meses_es_en)
        df_enso['fecha_merge'] = pd.to_datetime(df_enso['Year'].astype(str) + '-' + df_enso['mes_en'], format='%Y-%b').dt.strftime('%Y-%m')
    except Exception as e:
        st.error(f"Error en el preprocesamiento del archivo ENSO: {e}")
        return None, None, None, None, None, None

    # Preprocesamiento de datos de precipitación anual (mapa)
    try:
        df_precip_anual.columns = df_precip_anual.columns.str.strip()
        for col in ['Longitud', 'Latitud']:
            if col in df_precip_anual.columns and pd.api.types.is_object_dtype(df_precip_anual[col]):
                df_precip_anual[col] = df_precip_anual[col].str.replace(',', '.', regex=True).astype(float)
        gdf_stations = gpd.GeoDataFrame(
            df_precip_anual,
            geometry=gpd.points_from_xy(df_precip_anual['Longitud'], df_precip_anual['Latitud']),
            crs="EPSG:9377"
        )
        gdf_stations = gdf_stations.to_crs("EPSG:4326")
        gdf_stations['Longitud_geo'] = gdf_stations.geometry.x
        gdf_stations['Latitud_geo'] = gdf_stations.geometry.y
    except Exception as e:
        st.error(f"Error en el preprocesamiento del archivo de estaciones: {e}")
        return None, None, None, None, None, None

    # Preprocesamiento de datos de precipitación mensual
    try:
        df_precip_mensual.columns = df_precip_mensual.columns.str.strip().str.lower().str.replace('á', 'a').str.replace('é', 'e').str.replace('í', 'i').str.replace('ó', 'o').str.replace('ú', 'u').str.replace('ñ', 'n')
        df_precip_mensual.rename(columns={'ano': 'Year', 'mes': 'Mes'}, inplace=True)
        station_cols = [col for col in df_precip_mensual.columns if col.isdigit() and len(col) == 8]
        if not station_cols: return None, None, None, None, None, None
        df_long = df_precip_mensual.melt(
            id_vars=['id_fecha', 'Year', 'Mes'],
            value_vars=station_cols,
            var_name='Id_estacion',
            value_name='Precipitation'
        )
        df_long['Precipitation'] = df_long['Precipitation'].replace('n.d', np.nan).astype(float)
        df_long = df_long.dropna(subset=['Precipitation'])
        df_long['Fecha'] = pd.to_datetime(df_long['Year'].astype(str) + '-' + df_long['Mes'].astype(str), format='%Y-%m')
    except Exception as e:
        st.error(f"Error en el preprocesamiento del archivo de precipitación mensual: {e}")
        return None, None, None, None, None, None

    # Mapeo y Fusión de Estaciones
    gdf_stations['Nom_Est_clean'] = gdf_stations['Nom_Est'].astype(str).str.upper().str.strip()
    gdf_stations['Nom_Est_clean'] = gdf_stations['Nom_Est_clean'].apply(lambda x: re.sub(r'[^A-Z0-9]', '', x))
    gdf['Nom_Est_clean'] = gdf['Nom_Est'].astype(str).str.upper().str.strip()
    gdf['Nom_Est_clean'] = gdf['Nom_Est_clean'].apply(lambda x: re.sub(r'[^A-Z0-9]', '', x))
    gdf_stations['Id_estacio'] = gdf_stations['Id_estacio'].astype(str).str.strip()
    df_long['Id_estacion'] = df_long['Id_estacion'].astype(str).str.strip()
    station_mapping = gdf_stations.set_index('Id_estacio')[['Nom_Est_clean', 'Nom_Est']].to_dict('index')
    df_long['Nom_Est_clean'] = df_long['Id_estacion'].map(lambda x: station_mapping.get(x, {}).get('Nom_Est_clean'))
    df_long['Nom_Est'] = df_long['Id_estacion'].map(lambda x: station_mapping.get(x, {}).get('Nom_Est'))
    df_long = df_long.dropna(subset=['Nom_Est_clean'])

    return df_precip_anual, df_enso, df_precip_mensual, gdf_stations, df_long, gdf

# --- Interfaz de usuario ---
st.title('☔ Visor de Precipitación y Fenómeno ENSO')
st.markdown("""
Esta aplicación interactiva permite visualizar y analizar datos de precipitación
y su correlación con los eventos climáticos de El Niño-Oscilación del Sur (ENSO).
""")

# --- Panel de control (sidebar) ---
st.sidebar.header("Panel de Control")
st.sidebar.markdown("Por favor, suba los archivos requeridos para comenzar.")

with st.sidebar.expander("📂 **Cargar Archivos**"):
    uploaded_file_mapa = st.file_uploader("1. Cargar archivo de estaciones (mapaCVENSO.csv)", type="csv")
    uploaded_file_enso = st.file_uploader("2. Cargar archivo de ENSO (ENSO_1950_2023.csv)", type="csv")
    uploaded_file_precip = st.file_uploader("3. Cargar archivo de precipitación mensual (DatosPptnmes_ENSO.csv)", type="csv")
    uploaded_zip_shapefile = st.file_uploader("4. Cargar shapefile (.zip)", type="zip")

# Proceso de carga y caché de datos
if uploaded_file_mapa and uploaded_file_enso and uploaded_file_precip and uploaded_zip_shapefile:
    df_precip_anual, df_enso, df_precip_mensual, gdf_stations, df_long, gdf = load_and_preprocess_data(uploaded_file_mapa, uploaded_file_enso, uploaded_file_precip, uploaded_zip_shapefile)
else:
    st.info("Por favor, suba los 4 archivos para habilitar la aplicación.")
    st.stop()

if df_precip_anual is None:
    st.error("Uno o más archivos no se cargaron correctamente. Por favor, intente de nuevo.")
    st.stop()

# --- Controles en la barra lateral ---
all_stations_list = sorted(gdf_stations['Nom_Est'].unique())
celdas_list = sorted(gdf_stations['Celda_XY'].unique())
municipios_list = sorted(gdf_stations['municipio'].unique())

selected_municipios = st.sidebar.multiselect(
    '1. Filtrar por municipio',
    options=municipios_list,
    default=[]
)
selected_celdas = st.sidebar.multiselect(
    '2. Filtrar por celda (Celda_XY)',
    options=celdas_list,
    default=[]
)

stations_filtered_by_criteria = all_stations_list
if selected_municipios and selected_celdas:
    stations_by_municipio = set(gdf_stations[gdf_stations['municipio'].isin(selected_municipios)]['Nom_Est'].tolist())
    stations_by_celda = set(gdf_stations[gdf_stations['Celda_XY'].isin(selected_celdas)]['Nom_Est'].tolist())
    stations_filtered_by_criteria = sorted(list(stations_by_municipio.intersection(stations_by_celda)))
elif selected_municipios:
    stations_filtered_by_criteria = sorted(gdf_stations[gdf_stations['municipio'].isin(selected_municipios)]['Nom_Est'].tolist())
elif selected_celdas:
    stations_filtered_by_criteria = sorted(gdf_stations[gdf_stations['Celda_XY'].isin(selected_celdas)]['Nom_Est'].tolist())

select_all_stations = st.sidebar.checkbox('Seleccionar todas las estaciones', value=True)

if select_all_stations:
    filtered_stations = stations_filtered_by_criteria
else:
    filtered_stations = st.sidebar.multiselect(
        '3. Seleccione una o varias estaciones',
        options=stations_filtered_by_criteria,
        default=stations_filtered_by_criteria
    )

if not filtered_stations:
    st.warning("No hay estaciones seleccionadas. Por favor, ajuste sus filtros o seleccione estaciones.")
    st.stop()

selected_stations_clean = [gdf_stations[gdf_stations['Nom_Est'] == s]['Nom_Est_clean'].iloc[0] for s in filtered_stations if s in gdf_stations['Nom_Est'].values]

años_disponibles = sorted([int(col) for col in gdf_stations.columns if str(col).isdigit() and len(str(col)) == 4])
year_range = st.sidebar.slider(
    "Seleccione el rango de años",
    min_value=min(años_disponibles),
    max_value=max(años_disponibles),
    value=(min(años_disponibles), max(años_disponibles))
)

meses = ['Ene', 'Feb', 'Mar', 'Abr', 'May', 'Jun', 'Jul', 'Ago', 'Sep', 'Oct', 'Nov', 'Dic']
meses_seleccionados = st.sidebar.multiselect(
    'Seleccione los meses',
    options=meses,
    default=meses
)
meses_dict = {'Ene': 1, 'Feb': 2, 'Mar': 3, 'Abr': 4, 'May': 5, 'Jun': 6, 'Jul': 7, 'Ago': 8, 'Sep': 9, 'Oct': 10, 'Nov': 11, 'Dic': 12}
meses_numeros = [meses_dict[m] for m in meses_seleccionados]

# --- Pestañas de la aplicación ---
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 Gráficos de Series de Tiempo",
    "🗺️ Mapas",
    "📋 Tabla de Estaciones",
    "🔍 Análisis ENSO",
    "⬇️ Opciones de Descarga"
])

# --- Contenido de la Pestaña de Gráficos ---
with tab1:
    st.header("Visualizaciones de Precipitación 💧")

    # Gráfico de Serie de Tiempo Anual
    st.subheader("Precipitación Anual Total (mm)")
    df_precip_anual_filtered = gdf_stations[gdf_stations['Nom_Est'].isin(filtered_stations)].copy()
    year_cols = [col for col in df_precip_anual_filtered.columns if str(col).isdigit() and len(str(col)) == 4]
    df_precip_anual_filtered_melted = df_precip_anual_filtered.melt(
        id_vars=['Nom_Est', 'Nom_Est_clean', 'Latitud_geo', 'Longitud_geo', 'municipio', 'Celda_XY'],
        value_vars=year_cols,
        var_name='Año',
        value_name='Precipitación'
    )
    df_precip_anual_filtered_melted['Año'] = df_precip_anual_filtered_melted['Año'].astype(int)
    df_precip_anual_filtered_melted = df_precip_anual_filtered_melted[
        (df_precip_anual_filtered_melted['Año'] >= year_range[0]) &
        (df_precip_anual_filtered_melted['Año'] <= year_range[1])
    ].copy()

    mean_precip_by_station = df_precip_anual_filtered_melted.groupby('Nom_Est')['Precipitación'].transform('median')
    df_precip_anual_filtered_melted['Precipitación'].fillna(mean_precip_by_station, inplace=True)
    df_precip_anual_filtered_melted['Precipitación'] = df_precip_anual_filtered_melted.apply(
        lambda row: mean_precip_by_station[row.name] if row['Precipitación'] == 0 else row['Precipitación'],
        axis=1
    )

    if not df_precip_anual_filtered_melted.empty:
        selection_anual = alt.selection_point(fields=['Nom_Est'], bind='legend')
        chart_anual = alt.Chart(df_precip_anual_filtered_melted).mark_line().encode(
            x=alt.X('Año:O', title='Año'),
            y=alt.Y('Precipitación:Q', title='Precipitación Total (mm)'),
            color='Nom_Est:N',
            opacity=alt.condition(selection_anual, alt.value(1.0), alt.value(0.2)),
            tooltip=['Nom_Est', 'Año', 'Precipitación']
        ).properties(
            title='Precipitación Anual Total por Estación'
        ).add_params(selection_anual).interactive()
        st.altair_chart(chart_anual, use_container_width=True)
    else:
        st.warning("No hay datos para las estaciones y el rango de años seleccionados.")

    # Gráfico de Serie de Tiempo Mensual
    st.subheader("Precipitación Mensual Total (mm)")
    df_monthly_total = df_long.groupby(['Nom_Est', 'Year', 'Mes'])['Precipitation'].sum().reset_index()
    df_monthly_total['Fecha'] = pd.to_datetime(df_monthly_total['Year'].astype(str) + '-' + df_monthly_total['Mes'].astype(str), format='%Y-%m')
    df_monthly_filtered = df_monthly_total[
        (df_monthly_total['Nom_Est'].isin(filtered_stations)) &
        (df_monthly_total['Year'] >= year_range[0]) &
        (df_monthly_total['Year'] <= year_range[1]) &
        (df_monthly_total['Mes'].isin(meses_numeros))
    ].copy()

    if not df_monthly_filtered.empty:
        selection_mensual = alt.selection_point(fields=['Nom_Est'], bind='legend')
        chart_mensual = alt.Chart(df_monthly_filtered).mark_line().encode(
            x=alt.X('Fecha:T', title='Fecha'),
            y=alt.Y('Precipitation:Q', title='Precipitación Total (mm)'),
            color='Nom_Est:N',
            opacity=alt.condition(selection_mensual, alt.value(1.0), alt.value(0.2)),
            tooltip=[alt.Tooltip('Fecha', format='%Y-%m'), 'Precipitation', 'Nom_Est']
        ).properties(
            title='Precipitación Mensual Total por Estación'
        ).add_params(selection_mensual).interactive()
        st.altair_chart(chart_mensual, use_container_width=True)
    else:
        st.warning("No hay datos mensuales para las estaciones, el rango de años y los meses seleccionados.")

# --- Contenido de la Pestaña de Mapas ---
with tab2:
    st.header("Mapas de Lluvia y Precipitación")
    
    tab_folium, tab_animado, tab_interp = st.tabs([
        "Mapa de Estaciones",
        "Mapa Animado",
        "Superficie de Lluvia (Kriging)"
    ])

    # Sub-pestaña: Mapa de Estaciones (Folium)
    with tab_folium:
        st.subheader("Mapa de Estaciones de Lluvia en Colombia")
        st.markdown("Ubicación de las estaciones seleccionadas.")

        gdf_filtered = gdf_stations[gdf_stations['Nom_Est'].isin(filtered_stations)].copy()

        if not gdf_filtered.empty:
            tab_auto, tab_predef = st.tabs(["Centrado Automático", "Centrado Predefinido"])

            with tab_auto:
                st.info("El mapa se centra y ajusta automáticamente a las estaciones seleccionadas.")
                m_auto = folium.Map(location=[gdf_filtered['Latitud_geo'].mean(), gdf_filtered['Longitud_geo'].mean()], zoom_start=6)
                bounds = gdf_filtered.total_bounds
                m_auto.fit_bounds([[bounds[1], bounds[0]], [bounds[3], bounds[2]]])

                for _, row in gdf_filtered.iterrows():
                    folium.Marker(
                        location=[row['Latitud_geo'], row['Longitud_geo']],
                        tooltip=f"Estación: {row['Nom_Est']}<br>Municipio: {row['municipio']}<br>Porc. Datos: {row['Porc_datos']}<br>Celda: {row['Celda_XY']}",
                        icon=folium.Icon(color="blue", icon="cloud-rain", prefix='fa')
                    ).add_to(m_auto)
                folium_static(m_auto, width=900, height=600)
                
            with tab_predef:
                st.info("Use los botones para centrar el mapa en ubicaciones predefinidas.")
                col1, col2, col3 = st.columns(3)
                with col1:
                    if st.button("Ver Colombia"):
                        st.session_state.map_view = {"location": [4.5709, -74.2973], "zoom": 5}
                with col2:
                    if st.button("Ver Antioquia"):
                        st.session_state.map_view = {"location": [6.2442, -75.5812], "zoom": 8}
                with col3:
                    if st.button("Ver Estaciones Seleccionadas"):
                        bounds = gdf_filtered.total_bounds
                        st.session_state.map_view = {"location": [(bounds[1]+bounds[3])/2, (bounds[0]+bounds[2])/2], "zoom": 8}

                if 'map_view' not in st.session_state:
                    st.session_state.map_view = {"location": [4.5709, -74.2973], "zoom": 5}

                m_predef = folium.Map(location=st.session_state.map_view["location"], zoom_start=st.session_state.map_view["zoom"])

                for _, row in gdf_filtered.iterrows():
                    folium.Marker(
                        location=[row['Latitud_geo'], row['Longitud_geo']],
                        tooltip=f"Estación: {row['Nom_Est']}<br>Municipio: {row['municipio']}<br>Porc. Datos: {row['Porc_datos']}<br>Celda: {row['Celda_XY']}",
                        icon=folium.Icon(color="blue", icon="cloud-rain", prefix='fa')
                    ).add_to(m_predef)
                folium_static(m_predef, width=900, height=600)
        else:
            st.warning("No hay estaciones seleccionadas o datos de coordenadas para mostrar en el mapa.")

    # Sub-pestaña: Mapa Animado (Plotly)
    with tab_animado:
        st.subheader("Mapa Animado de Precipitación Anual")
        st.markdown("Visualice la precipitación anual a lo largo del tiempo.")
        if not df_precip_anual_filtered_melted.empty:
            tab_anim_auto, tab_anim_predef = st.tabs(["Centrado Automático", "Centrado Predefinido"])
            
            with tab_anim_auto:
                fig_mapa_animado = px.scatter_geo(
                    df_precip_anual_filtered_melted,
                    lat='Latitud_geo',
                    lon='Longitud_geo',
                    color='Precipitación',
                    size='Precipitación',
                    hover_name='Nom_Est',
                    animation_frame='Año',
                    projection='natural earth',
                    title='Precipitación Anual de las Estaciones',
                    color_continuous_scale=px.colors.sequential.RdBu,
                    width=1000,
                    height=700
                )
                fig_mapa_animado.update_geos(fitbounds="locations", showcountries=True, countrycolor="black")
                st.plotly_chart(fig_mapa_animado, use_container_width=True)

            with tab_anim_predef:
                col1, col2, col3 = st.columns(3)
                with col1:
                    if st.button("Ver Colombia", key="anim_col"):
                        st.session_state.anim_map_view = {"location": [4.5709, -74.2973], "zoom": 5}
                with col2:
                    if st.button("Ver Antioquia", key="anim_ant"):
                        st.session_state.anim_map_view = {"location": [6.2442, -75.5812], "zoom": 8}
                with col3:
                    if st.button("Ver Estaciones Seleccionadas", key="anim_est"):
                        bounds = gdf_filtered.total_bounds
                        st.session_state.anim_map_view = {"location": [(bounds[1]+bounds[3])/2, (bounds[0]+bounds[2])/2], "zoom": 8}
                
                if 'anim_map_view' not in st.session_state:
                    st.session_state.anim_map_view = {"location": [4.5709, -74.2973], "zoom": 5}

                fig_mapa_animado_predef = px.scatter_geo(
                    df_precip_anual_filtered_melted,
                    lat='Latitud_geo',
                    lon='Longitud_geo',
                    color='Precipitación',
                    size='Precipitación',
                    hover_name='Nom_Est',
                    animation_frame='Año',
                    projection='natural earth',
                    title='Precipitación Anual de las Estaciones',
                    color_continuous_scale=px.colors.sequential.RdBu,
                    width=1000,
                    height=700
                )
                fig_mapa_animado_predef.update_layout(
                    geo = dict(
                        center = dict(
                            lat=st.session_state.anim_map_view['location'][0],
                            lon=st.session_state.anim_map_view['location'][1]
                        ),
                        projection_scale=st.session_state.anim_map_view['zoom'],
                        showcountries=True, countrycolor="black"
                    )
                )
                st.plotly_chart(fig_mapa_animado_predef, use_container_width=True)
        else:
            st.warning("No hay datos suficientes para generar el mapa animado.")

    # Sub-pestaña: Mapa de Interpolación
    with tab_interp:
        st.subheader("Superficie de Precipitación Anual por Kriging")
        st.markdown("Mapa base con la superficie de lluvia interpolada a partir de la precipitación media anual.")

        if not df_precip_anual_filtered_melted.empty:
            df_mean_precip = df_precip_anual_filtered_melted.groupby('Nom_Est')[['Longitud_geo', 'Latitud_geo', 'Precipitación']].mean().reset_index()

            longs = np.unique(df_mean_precip['Longitud_geo'])
            lats = np.unique(df_mean_precip['Latitud_geo'])

            if len(longs) < 2 or len(lats) < 2:
                st.warning("No hay suficientes estaciones seleccionadas para generar una superficie de interpolación. Seleccione al menos dos estaciones con diferentes longitudes y latitudes.")
            else:
                lon_grid = np.linspace(min(longs), max(longs), 100)
                lat_grid = np.linspace(min(lats), max(lats), 100)

                z_grid = None
                interpolation_method = "Kriging"
                
                try:
                    OK = OrdinaryKriging(
                        df_mean_precip['Longitud_geo'].values,
                        df_mean_precip['Latitud_geo'].values,
                        df_mean_precip['Precipitación'].values,
                        variogram_model='linear',
                        verbose=False,
                        enable_plotting=False
                    )
                    z_grid, ss_grid = OK.execute("grid", lon_grid, lat_grid)
                    z_grid = z_grid.data
                    
                    if np.isnan(z_grid).all() or z_grid.size == 0:
                        raise ValueError("Kriging no pudo generar una grilla válida. Intentando con un método alternativo.")

                except Exception as e:
                    st.warning(f"Kriging no pudo generar una superficie de interpolación válida: {e}. Usando un método de interpolación más simple como respaldo.")
                    interpolation_method = "Griddata (Fallback)"
                    
                    points = df_mean_precip[['Longitud_geo', 'Latitud_geo']].values
                    values = df_mean_precip['Precipitación'].values
                    grid_x, grid_y = np.meshgrid(lon_grid, lat_grid, indexing='ij')
                    
                    z_grid = griddata(points, values, (grid_x, grid_y), method='linear')
                
                if z_grid is not None and not np.isnan(z_grid).all() and z_grid.size > 0:
                    
                    fig_interp = go.Figure()

                    # Interpolated surface
                    fig_interp.add_trace(go.Choroplethmapbox(
                        geojson={"type": "FeatureCollection", "features": [{"type": "Feature", "geometry": {"type": "Polygon", "coordinates": [[(x, y) for x, y in zip(lon_grid, lat_grid)]]}, "properties": {"z": z_grid}}]},
                        locations=["z"],
                        z=[z_grid.flatten().tolist()],
                        colorscale='YlGnBu',
                        showscale=True,
                        colorbar_title="Precipitación (mm)",
                        zmin=z_grid.min(),
                        zmax=z_grid.max()
                    ))

                    # Station markers
                    fig_interp.add_trace(go.Scattermapbox(
                        lat=df_mean_precip['Latitud_geo'],
                        lon=df_mean_precip['Longitud_geo'],
                        mode='markers',
                        marker=dict(
                            size=10,
                            color='black',
                            line=dict(width=1, color='white')
                        ),
                        hoverinfo='text',
                        hovertext=df_mean_precip['Nom_Est'] + '<br>Pptn. Anual: ' + df_mean_precip['Precipitación'].round(2).astype(str),
                        name='Estaciones de Lluvia'
                    ))
                    
                    center_lat = (max(lats) + min(lats)) / 2
                    center_lon = (max(longs) + min(longs)) / 2
                    zoom_factor = 10

                    fig_interp.update_layout(
                        title_text=f'Superficie de Precipitación Media Anual ({interpolation_method})',
                        mapbox=dict(
                            style="stamen-terrain",
                            center=go.layout.mapbox.Center(lat=center_lat, lon=center_lon),
                            zoom=zoom_factor
                        )
                    )
                    st.plotly_chart(fig_interp, use_container_width=True)
                else:
                    st.warning("La grilla de interpolación generada no es válida. Por favor, intente con una selección de estaciones diferente.")
        else:
            st.warning("No hay datos suficientes para generar el mapa de interpolación.")

# --- Contenido de la Pestaña de Tablas ---
with tab3:
    st.header("📋 Información Detallada de las Estaciones")
    st.markdown("Esta tabla muestra información relevante para las estaciones seleccionadas, incluyendo la precipitación media anual calculada para el rango de años elegido.")

    df_mean_precip = df_precip_anual_filtered_melted.groupby('Nom_Est')['Precipitación'].mean().reset_index()
    df_mean_precip.rename(columns={'Precipitación': 'Precipitación media anual (mm)'}, inplace=True)
    df_mean_precip['Precipitación media anual (mm)'] = df_mean_precip['Precipitación media anual (mm)'].round(2)

    gdf_info_table = gdf_stations[gdf_stations['Nom_Est'].isin(filtered_stations)].copy()
    gdf_info_table = gdf_info_table.merge(df_mean_precip, on='Nom_Est', how='left')

    columns_to_show = [
        'Nom_Est', 'Porc_datos', 'Celda_XY', 'Cant_Est', 'departamento', 'municipio', 'AHZ', 'SZH',
        'Longitud', 'Latitud', 'vereda', 'SUBREGION', 'Precipitación media anual (mm)'
    ]

    existing_columns = [col for col in columns_to_show if col in gdf_info_table.columns]
    df_info_table = gdf_info_table[existing_columns].copy()

    if not df_info_table.empty:
        st.dataframe(df_info_table)
    else:
        st.warning("No hay datos para las estaciones y el rango de años seleccionados. La tabla no se puede mostrar.")

# --- Contenido de la Pestaña de Análisis ENSO ---
with tab4:
    st.header("Análisis de Precipitación y el Fenómeno ENSO")
    st.markdown("Esta sección explora la relación entre la precipitación y los eventos de El Niño-Oscilación del Sur.")

    df_analisis = df_long.copy()
    try:
        df_analisis['fecha_merge'] = df_analisis['Fecha'].dt.strftime('%Y-%m')
        df_analisis = pd.merge(df_analisis, df_enso[['fecha_merge', 'Anomalia_ONI', 'ENSO']], on='fecha_merge', how='left')
        df_analisis = df_analisis.dropna(subset=['ENSO']).copy()

        df_enso_group = df_analisis.groupby('ENSO')['Precipitation'].mean().reset_index()
        df_enso_group = df_enso_group.rename(columns={'Precipitation': 'Precipitación'})

        fig_enso = px.bar(
            df_enso_group,
            x='ENSO',
            y='Precipitación',
            title='Precipitación Media por Evento ENSO',
            labels={'ENSO': 'Evento ENSO', 'Precipitación': 'Precipitación Media (mm)'},
            color='ENSO'
        )
        st.plotly_chart(fig_enso, use_container_width=True)

        df_corr = df_analisis[['Anomalia_ONI', 'Precipitation']].dropna()
        if not df_corr.empty:
            correlation = df_corr['Anomalia_ONI'].corr(df_corr['Precipitation'])
            st.write(f"### Coeficiente de Correlación entre Anomalía ONI y Precipitación: **{correlation:.2f}**")
            st.info("""
            **Interpretación:**
            - Un valor cercano a 1 indica una correlación positiva fuerte.
            - Un valor cercano a -1 indica una correlación negativa fuerte.
            - Un valor cercano a 0 indica una correlación débil o nula.
            """)
        else:
            st.warning("No hay suficientes datos para calcular la correlación.")
    except Exception as e:
        st.error(f"Error en el análisis ENSO: {e}")

# --- Contenido de la Pestaña de Descarga ---
with tab5:
    st.header("Opciones de Descarga 📥")
    st.markdown("""
    **Exportar a CSV:**
    Para obtener los datos filtrados en formato CSV, haga clic en los botones de descarga a continuación.
    """)

    st.markdown("**Datos de Precipitación Anual**")
    csv_anual = df_precip_anual_filtered_melted.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="Descargar datos anuales (CSV)",
        data=csv_anual,
        file_name='precipitacion_anual.csv',
        mime='text/csv',
    )

    st.markdown("**Datos de Precipitación Mensual**")
    csv_mensual = df_monthly_filtered.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="Descargar datos mensuales (CSV)",
        data=csv_mensual,
        file_name='precipitacion_mensual.csv',
        mime='text/csv',
    )

    st.markdown("---")
    st.markdown("""
    **Exportar a Imagen (PNG/SVG):**
    Para descargar los **gráficos de Plotly** como imagen, simplemente pase el cursor sobre el gráfico y haga clic en el ícono de la cámara 📷 que aparece en la parte superior derecha. Para los **mapas de Folium**, use una captura de pantalla.

    **Exportar a PDF:**
    Para guardar una copia de toda la página (incluyendo todos los gráficos y tablas visibles) como un archivo PDF, utilice la función de su navegador:
    1. Vaya al menú del navegador (usualmente en la esquina superior derecha).
    2. Seleccione **"Imprimir..."**.
    3. En el destino, elija **"Guardar como PDF"**.
    """)
