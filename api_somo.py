from operator import ipow
import pandas as pd
import numpy as np
import plotly.express as px
import requests
import matplotlib.pyplot as plt
from utils.utils import reshample_time_serie


def fetch_qf(api_url, params=None):
    try:
        # Si no se pasan parámetros, solo se hace un POST sin datos (filtrando todo)
        if params is None:
            response = requests.post(api_url)  # No se envían parámetros
        else:
            response = requests.post(api_url, json=params)  # Enviar parámetros de filtrado
        
        # Comprobar si la solicitud fue exitosa
        if response.status_code == 200:
            return response.json()  # Devuelve los datos en formato JSON
        else:
            return {"error": f"Error {response.status_code}: {response.text}"}
    
    except requests.exceptions.RequestException as e:
        # Manejar excepciones de la solicitud
        return {"error": f"Request failed: {str(e)}"}

import requests
import pandas as pd
import logging
from typing import Optional, Dict, Any

# Setting up the logger for monitoring
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DataFetcher:

    @staticmethod
    def _send_post_request(api_url: str, params: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Send a POST request to the API and handle any potential errors.

        :param api_url: The URL of the API to send the request to.
        :param params: The parameters to send with the request.
        :return: JSON response or None if an error occurs.
        """
        try:
            logger.info(f"Sending request to {api_url} with parameters: {params}")
            response = requests.post(api_url, json=params)
            response.raise_for_status()  # Raises an error for bad HTTP responses (4xx or 5xx)

            return response.json()

        except requests.exceptions.RequestException as e:
            logger.error(f"Request error: {e}")
            return None

    @staticmethod
    def _process_data(data: Dict[str, Any]) -> Optional[pd.DataFrame]:
        """
        Process raw data from the API into a structured DataFrame.

        :param data: JSON data from the API.
        :return: Processed DataFrame or None if the data is invalid.
        """
        if not data:
            logger.warning("Received data is empty or None.")
            return None

        try:
            df = pd.DataFrame(data)
            logger.info(f"Original columns: {df.columns.tolist()}")

            # Column name mapping
            column_map = {
                'date_time': 'timestamp',
                'sensor_data': 'value',
                'quality_flag': 'qf'
            }

            # Rename columns if they exist
            for original, renamed in column_map.items():
                if original in df.columns:
                    df.rename(columns={original: renamed}, inplace=True)
                else:
                    logger.warning(f"Column '{original}' not found in the DataFrame.")

            logger.info(f"Columns after renaming: {df.columns.tolist()}")

            # Convert 'timestamp' to datetime and set it as the index
            if 'timestamp' in df.columns:
                df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True)
                df.set_index('timestamp', inplace=True)
                logger.info("Timestamp conversion and indexing complete.")
            
            return df

        except Exception as e:
            logger.error(f"Error processing data: {e}")
            return None

    @staticmethod
    def fetch_and_process_data(api_url: str, params: Dict[str, Any]) -> Optional[pd.DataFrame]:
        """
        Fetch historical data from the API, process it, and return a DataFrame.

        :param api_url: The URL of the API to send the request to.
        :param params: The parameters to send with the request.
        :return: Processed DataFrame or None if there was an error.
        """
        # 1. Fetch data from the API
        data = DataFetcher._send_post_request(api_url, params)

        if data is None:
            logger.error("Failed to fetch data from the API.")
            return pd.DataFrame()  # Return an empty DataFrame in case of failure

        # 2. Process the fetched data
        df = DataFetcher._process_data(data)
        
        df.sort_index(inplace=True)

        if df is None or df.empty:
            logger.error("The historical series is empty.")
            return pd.DataFrame()  # Return an empty DataFrame if processing fails

        return df


def fetch_data_year(apli_url, params):
    # Supongo que fetch_and_process_data devuelve un DataFrame con un índice temporal
    serie = fetch_and_process_data(apli_url, params)
    
    # Asegúrate de que el índice sea de tipo datetime
    serie.index = pd.to_datetime(serie.index)
    
    dfs = []  # Lista para almacenar los DataFrames por año
    
    # Iterar sobre los años únicos en la serie
    for year in serie.index.year.unique():
        # Filtra los datos por el año actual
        serie_year = serie[serie.index.year == year]
        
        # Renombra la columna 'value' al año correspondiente
        serie_year = serie_year.rename(columns={"value": str(year)})
        
        # Añade el DataFrame a la lista (solo la columna renombrada)
        dfs.append(serie_year[[str(year)]])
    
    # Combina todas las series temporales usando el índice
    combined_df = pd.concat(dfs, axis=1)
    
    # Ordena por el índice (timestamp)
    combined_df.sort_index(inplace=True)
    
    return combined_df

def unify_series(api_url, stations):
    dfs = []
    
    for station, fetch_data in stations.items():
        # Obtén los datos de la estación
        station_data = fetch_and_process_data(api_url, fetch_data)
        
        # Renombra la columna de valores a algo único por estación
        station_data.rename(columns={"value": station}, inplace=True)
        
        # Agrega el DataFrame a la lista
        dfs.append(station_data[[station]])  # Solo la columna renombrada
    
    # Combina todas las series temporales usando el índice (timestamp)
    combined_df = pd.concat(dfs, axis=1)
    
    # Ordena por el índice (timestamp)
    combined_df.sort_index(inplace=True)
    
    return combined_df

def plot_imputation_results(original_df, imputations):
    """
    Grafica los resultados de las diferentes imputaciones.
    """
    plt.figure(figsize=(14, 8))
    
    # Gráfico original
    plt.plot(original_df.index, original_df['value'], label='Original', color='black', linestyle='--')
    
    # Gráficos imputados
    for method, series in imputations.items():
        plt.plot(series.index, series, label=method)
    
    plt.title('Comparación de Métodos de Imputación')
    plt.xlabel('Fecha')
    plt.ylabel('Precipitación')
    plt.legend()
    plt.grid(True)
    plt.show()

def plot_time_series(df):
    """
    Esta función recibe un DataFrame de series temporales y genera una gráfica de serie temporal
    mostrando los valores faltantes como interrupciones en la línea.
    """
    # Verificar si las columnas necesarias existen en el DataFrame
    '''if 'timestamp' not in df.columns or 'value' not in df.columns:
        print("El DataFrame debe contener las columnas 'timestamp' y 'value'.")
        return'''
    
    # Asegurarse de que la columna 'timestamp' sea un Datetime
    df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')  # Convertir a datetime

    # Establecer 'timestamp' como el índice si no lo es
    df.set_index('timestamp', inplace=True)

    # Crear la gráfica de serie temporal
    fig = px.line(
        df,
        x=df.index,
        y='value',
        color='qf',  # Asegúrate de que 'qf' sea una columna en tu DataFrame
        title='Serie Temporal con Valores Faltantes'
    )
    
    # Configurar para no conectar los espacios con valores NaN
    #fig.update_traces(connectgaps=True)

    # Mostrar la gráfica
    fig.show()

def booxplot(df):
    # Filtrar los datos según la calidad seleccionada
    df_filtered = df[df['qf'] == 1]

    

    fig = px.box(df_filtered, x='month', y='value', color= 'month', title='Distribución estacional de corriente',
                        labels={'Marea_hor': 'marea'})
    
    fig.show()

'''
# Parámetros y URL
api_url = "http://127.0.0.1:8000/time-series/qfcontroler/filtrate_time_serie/"
params = {
    "station_name": "juanchaco",
    "variable_name": "Precipitación acumulada",
    "start_date": "2019-01-22",
    "end_date": "2020-12-23"
}

# Obtener los datos
df = fetch_and_process_data(api_url, params)

# Si los datos fueron correctamente obtenidos, graficar
if df is not None:
    #plot_time_series(df)
    booxplot(df)
'''