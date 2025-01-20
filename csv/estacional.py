import pandas as pd
import numpy as np
import plotly.graph_objects as go
from statsmodels.tsa.seasonal import seasonal_decompose
from csv_ import get_df_serie_temporal

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

# Ruta al archivo CSV
ruta = r"C:\Users\dreng\OneDrive - dimar.mil.co\Documentos\2024\HISTORICO_ACTUALIZADO_2023\BBV\RLS\HOM\Resultados nivelación\BBV_RLS_HOM_20090323-20231231_H_HL.csv"

# Cargar los datos de la serie temporal
serie = get_df_serie_temporal(ruta)
serie = serie.iloc[:,0]
print(serie.head())
periodo = len(serie) // 2
print(periodo)
# Llamar a la función para descomponer y graficar
# Asume que el periodo de estacionalidad es de 24 para datos horarios
descomponer_y_graficar(serie, periodo=8760*2) #1 año
