from urllib import response
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import acf

def decompose_time_series(serie):
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
    frequency = 90  
    decomposition = seasonal_decompose(serie['value'], model=model, period=frequency)

    trend_df = decomposition.trend.to_frame(name='value')
    seasonal_df = decomposition.seasonal.to_frame(name='value')
    resid_df = decomposition.resid.to_frame(name='value')
    response = {
        'trend':trend_df,
        'seasonal':seasonal_df,
        'resid':resid_df
    }
    return response

from statsmodels.tsa.stattools import adfuller

import pandas as pd
import numpy as np
from statsmodels.tsa.stattools import acf

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
