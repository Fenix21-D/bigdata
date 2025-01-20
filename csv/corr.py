import pandas as pd
import numpy as np
import plotly.graph_objects as go
from statsmodels.tsa.stattools import acf
from csv_ import get_df_serie_temporal

# Función para cargar datos de la serie temporal
def cargar_datos(ruta):
    """Carga la serie temporal desde un archivo CSV."""
    try:
        # Cargar el DataFrame
        serie = get_df_serie_temporal(ruta)
        # Asegurarse de que la columna 'RLS' esté presente
        if 'RLS' not in serie.columns:
            raise ValueError("La columna 'RLS' no se encuentra en el DataFrame.")
        return serie['RLS']
    except Exception as e:
        print(f"Error al cargar los datos: {e}")
        return None

# Función para calcular y graficar la autocorrelación
def graficar_autocorrelacion(serie, lags=40):
    """Calcula y grafica la función de autocorrelación (ACF) de una serie temporal."""
    # Verificar que la serie no sea nula y tenga suficientes datos
    if serie is None or len(serie) < lags:
        print("La serie es nula o no contiene suficientes datos.")
        return

    # Calcular la autocorrelación
    acf_values = acf(serie, nlags=lags)
    
    # Crear la gráfica con Plotly
    fig = go.Figure()

    # Añadir las barras de autocorrelación
    fig.add_trace(go.Bar(
        x=list(range(len(acf_values))),
        y=acf_values,
        name='ACF',
        marker_color='blue'
    ))

    # Añadir líneas de confianza
    conf_interval = 1.96 / np.sqrt(len(serie))  # Límite de confianza del 95%
    fig.add_trace(go.Scatter(
        x=list(range(lags + 1)),
        y=[conf_interval] * (lags + 1),
        mode='lines',
        name='Límite Superior',
        line=dict(color='orange', dash='dash')
    ))

    fig.add_trace(go.Scatter(
        x=list(range(lags + 1)),
        y=[-conf_interval] * (lags + 1),
        mode='lines',
        name='Límite Inferior',
        line=dict(color='orange', dash='dash')
    ))

    # Configurar la gráfica
    fig.update_layout(
        title='Función de Autocorrelación (ACF)',
        xaxis_title='Lags',
        yaxis_title='Autocorrelación',
        showlegend=True,
        template='plotly_white'  # Cambiar el tema de la gráfica
    )

    # Mostrar la gráfica
    fig.show()

# Ruta al archivo CSV
ruta = r"C:\Users\dreng\OneDrive - dimar.mil.co\Documentos\2024\HISTORICO_ACTUALIZADO_2023\BBV\RLS\HOM\Resultados nivelación\BBV_RLS_HOM_20090323-20231231_H_HL.csv"

# Cargar los datos de la serie temporal
serie = cargar_datos(ruta)

# Llamar a la función para graficar la autocorrelación
graficar_autocorrelacion(serie, lags=40)
