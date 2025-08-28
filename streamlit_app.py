# script_para_preparar_datos.py
import pandas as pd

try:
    # Carga tus archivos originales (asegúrate de que los nombres y rutas sean correctos)
    df_precip_mensual = pd.read_csv('DatosPptnmes_ENSO.csv', sep=';')
    df_precip_anual = pd.read_csv('mapaCVENSO.csv', sep=';')
    df_enso = pd.read_csv('ENSO_1950_2023.csv', sep=';')

    # --- 1. Preparar el archivo de precipitación mensual ---
    
    # Estandarizar columnas para la unión
    df_precip_mensual.rename(columns={'Id_Fecha': 'Id_Fecha'}, inplace=True)
    df_enso.rename(columns={'Id_Fecha': 'Id_Fecha'}, inplace=True)
    df_precip_mensual['Id_Fecha'] = pd.to_numeric(df_precip_mensual['Id_Fecha'], errors='coerce')
    df_enso['Id_Fecha'] = pd.to_numeric(df_enso['Id_Fecha'], errors='coerce')

    # Unir TODA la información de ENSO a la tabla mensual
    df_mensual_final = pd.merge(df_precip_mensual, df_enso, on='Id_Fecha', how='left')

    # --- 2. Preparar el archivo de estaciones/anual ---
    
    # Crear un resumen de ENSO por año
    df_enso['Año'] = (df_enso['Id_Fecha'] // 100)
    df_enso_anual = df_enso.groupby('Año')['ENSO'].agg(lambda x: x.mode().iloc[0]).reset_index()
    df_enso_anual.rename(columns={'ENSO': 'ENSO_anual'}, inplace=True)

    # Unir el resumen anual con la tabla de estaciones
    # Esto es complejo, lo manejaremos directamente en el app por ahora.
    # Por ahora, solo usaremos el archivo de estaciones original.
    
    # --- 3. Guardar el nuevo archivo mensual ---
    df_mensual_final.to_csv('DatosPptnmes_CON_ENSO.csv', sep=';', index=False)

    print("¡Archivo 'DatosPptnmes_CON_ENSO.csv' creado con éxito!")
    print("Por favor, usa este nuevo archivo y el 'mapaCVENSO.csv' original en la aplicación.")

except FileNotFoundError as e:
    print(f"Error: No se encontró el archivo {e.filename}. Asegúrate de que los archivos CSV originales estén en la misma carpeta que este script.")
except Exception as e:
    print(f"Ocurrió un error: {e}")
