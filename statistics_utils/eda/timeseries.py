from urllib import response
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import acf
from statsmodels.tsa.stattools import adfuller


def decompose_time_series(serie, periods_s=30):
    """
    Performs seasonal decomposition of a time series into trend, seasonal, and residual components.
    
    Parameters:
    - series (pd.DataFrame): A DataFrame with a DateTimeIndex and a column named 'value' containing the time series data.
    - frequency (int): The frequency of the time series (e.g., 12 for monthly data, 365 for daily data with yearly seasonality).
    - model (str): The decomposition model ('additive' or 'multiplicative').
    
    Returns:
    - decomposition (DecomposeResult): An object containing the trend, seasonal, and residual components.
    """
    model='additive'
    # Ensure the index is of type DatetimeIndex
    if not isinstance(serie.index, pd.DatetimeIndex):
        raise ValueError("The index must be of type DatetimeIndex.")

    # Perform seasonal decomposition
    # Obtener la frecuencia inferida
    frequency = pd.infer_freq(serie.index)
    print(f"Frecuencia inferida: {frequency}")

    # Convertir la frecuencia inferida a un entero
    frequency_int = None

    if frequency == 'D':  # Diaria
        frequency_int = 1  # 1 día
    elif frequency == 'W':  # Semanal
        frequency_int = 7  # 7 días
    elif frequency == 'ME':  # Mensual
        frequency_int = 30  # Aproximadamente 30 días
    elif frequency == 'Q':  # Trimestral
        frequency_int = 90  # Aproximadamente 90 días
    elif frequency == 'Y':  # Anual
        frequency_int = 365  # Aproximadamente 365 días
    elif frequency == 'ME':  # Fin de mes
        frequency_int = 30  # Aproximadamente 30 días
    elif frequency == 'B':  # Días hábiles
        frequency_int = 1  # Consideramos 1 día como unidad
    else:
        print(f"Frecuencia '{frequency}' no soportada para conversión.")
    decomposition = seasonal_decompose(serie['value'], model=model, period=periods_s*frequency_int)

    trend_df = decomposition.trend.to_frame(name='value')
    seasonal_df = decomposition.seasonal.to_frame(name='value')
    resid_df = decomposition.resid.to_frame(name='value')
    ajusted_seasonal_df = seasonal_df*trend_df
    response = {
        'trend':trend_df,
        'seasonal':seasonal_df,
        'ajusted_seasonal': ajusted_seasonal_df,
        'resid':resid_df
    }
    return response


def correlation(serie, lags):
    """
    Calcula la autocorrelación para una serie temporal y retorna un DataFrame.
    
    Parámetros:
    - serie (pd.Series): Serie temporal para calcular la autocorrelación.
    - lags (int): Número máximo de lags a calcular.
    
    Retorna:
    - pd.DataFrame: DataFrame con las columnas 'lag' y 'autocorrelation'.
    """
    # Calcular la autocorrelación para los lags especificados
    autocorr_values = acf(serie, nlags=lags, fft=True)
    lag_indices = np.arange(len(autocorr_values))
    
    # Crear el DataFrame
    autocorr_df = pd.DataFrame({
        'lag': lag_indices,
        'values': autocorr_values
    })
    autocorr_df.set_index('lag', inplace=True)
    return autocorr_df


def test_stationarity(time_series):
    """
    Realiza la prueba de Dickey-Fuller aumentada para verificar la estacionariedad de una serie de tiempo.
    """
    # Aplicar la prueba de Dickey-Fuller
    result = adfuller(time_series, autolag='AIC')

    print("Resultados de la Prueba de Dickey-Fuller Aumentada:")
    print(f"Estadístico de Prueba: {result[0]}")
    print(f"Valor P: {result[1]}")
    print(f"Número de Retardos Usados: {result[2]}")
    print(f"Número de Observaciones: {result[3]}")
    print("Valores Críticos:")
    for key, value in result[4].items():
        print(f"  {key}: {value}")

    # Conclusión
    if result[1] <= 0.05:
        print("\nLa serie es estacionaria (se rechaza H₀).")
    else:
        print("\nLa serie no es estacionaria (no se rechaza H₀).")
