import pandas as pd
import numpy as np
import plotly.graph_objects as go
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.statespace.sarimax import SARIMAX
from csv_ import get_df_serie_temporal
from statsmodels.tsa.stattools import adfuller
import matplotlib.pyplot as plt

# Función para descomponer y graficar componentes
def descomponer_y_graficar(serie, periodo):
    """Descompone la serie temporal y grafica sus componentes."""
    if serie is None or len(serie) < periodo:
        print("La serie es nula o no contiene suficientes datos.")
        return

    # Descomponer la serie
    descomposicion = seasonal_decompose(serie, model='additive', period=periodo)

    # Crear la gráfica
    fig = go.Figure()

    # Graficar la tendencia
    fig.add_trace(go.Scatter(
        x=serie.index,
        y=descomposicion.trend,
        mode='lines',
        name='Tendencia',
        line=dict(color='blue')
    ))

    # Graficar la estacionalidad
    fig.add_trace(go.Scatter(
        x=serie.index,
        y=descomposicion.seasonal,
        mode='lines',
        name='Estacionalidad',
        line=dict(color='green')
    ))

    # Graficar los residuales
    fig.add_trace(go.Scatter(
        x=serie.index,
        y=descomposicion.resid,
        mode='lines',
        name='Residuales',
        line=dict(color='orange')
    ))

    # Configurar la gráfica
    fig.update_layout(
        title='Descomposición de la Serie Temporal',
        xaxis_title='Fecha',
        yaxis_title='Valor',
        showlegend=True,
        template='plotly_white'
    )

    # Mostrar la gráfica
    fig.show()

# Función para aplicar el modelo SARIMA
def aplicar_sarima(serie, order, seasonal_order):
    """Aplica el modelo SARIMA y grafica los resultados."""
    model = SARIMAX(serie, order=order, seasonal_order=seasonal_order)
    model_fit = model.fit(disp=False)

    # Resumen del modelo
    print(model_fit.summary())

    # Hacer predicciones
    pred = model_fit.forecast(steps=10)  # Cambia 10 por el número de pasos que deseas predecir

    # Graficar resultados
    plt.figure(figsize=(10, 5))
    plt.plot(serie.index, serie, label='Serie Original')
    plt.plot(pd.date_range(start=serie.index[-1], periods=10, freq='H'), pred, label='Predicción SARIMA', color='red')
    plt.legend()
    plt.title('Predicciones del modelo SARIMA')
    plt.show()

# Ruta al archivo CSV
ruta = r"C:\Users\dreng\OneDrive - dimar.mil.co\Documentos\2024\HISTORICO_ACTUALIZADO_2023\BBV\RLS\HOM\Resultados nivelación\BBV_RLS_HOM_20090323-20231231_H_HL.csv"

# Cargar los datos de la serie temporal
serie = get_df_serie_temporal(ruta)
serie = serie.iloc[:, 0]

# Asegúrate de que la serie tiene un índice de tiempo y establece la frecuencia
fecha_inicial = serie.index[0]  # Obtener la fecha de la primera entrada
serie.index = pd.date_range(start=fecha_inicial, periods=len(serie), freq='h')  # Establecer el índice


# Suponiendo que el periodo de estacionalidad es de 8760 para datos anuales (24 horas por 365 días)
periodo = 8760

# Llamar a la función para descomponer y graficar
descomponer_y_graficar(serie, periodo=periodo)

# Verificar estacionariedad
result = adfuller(serie)
print('p-value:', result[1])  # Si es < 0.05, la serie es estacionaria

# Definir parámetros de SARIMA
order = (1, 1, 1)  # Ajusta p, d, q según tu análisis
seasonal_order = (1, 1, 1, 24)  # Ajusta P, D, Q y s según tu análisis (s=24 para datos horarios)

# Llamar a la función para aplicar el modelo SARIMA
aplicar_sarima(serie, order, seasonal_order)
