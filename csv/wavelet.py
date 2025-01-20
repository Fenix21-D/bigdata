import pandas as pd
import numpy as np
import plotly.graph_objects as go
import pywt
from csv_ import get_df_serie_temporal

# Ruta a tu archivo CSV
ruta = r"C:\Users\dreng\OneDrive - dimar.mil.co\Documentos\2024\HISTORICO_ACTUALIZADO_2023\BBV\RLS\HOM\Resultados nivelación\BBV_RLS_HOM_20090323-20231231_H_HL.csv"
serie = get_df_serie_temporal(ruta)

# Selecciona la columna de nivel del mar
nivel_del_mar = serie['RLS']
nivel_del_mar = nivel_del_mar - np.mean(nivel_del_mar)  # Centra la serie

def wavelet1(serie):
    # Realiza la Transformada Wavelet
    wavelet = 'haar'  # Puedes elegir otros como 'db1', 'db2', etc.
    coeffs = pywt.wavedec(serie, wavelet)

    # Descomponer los coeficientes
    cA = coeffs[0]  # Coeficientes de aproximación
    cD = coeffs[1:]  # Coeficientes de detalle

    # Graficar los coeficientes de la wavelet
    fig = go.Figure()

    # Graficar coeficiente de aproximación
    fig.add_trace(go.Scatter(x=np.arange(len(cA)), y=cA, mode='lines', name='Coeficientes de Aproximación (cA)'))

    # Graficar coeficientes de detalle
    for i, detail in enumerate(cD):
        fig.add_trace(go.Scatter(x=np.arange(len(detail)), y=detail, mode='lines', name=f'Coeficientes de Detalle (cD{i+1})'))

    fig.update_layout(title='Transformada Wavelet de la Serie Temporal de Nivel del Mar',
                    xaxis_title='Índice',
                    yaxis_title='Valor',
                    showlegend=True)

    # Mostrar la gráfica
    fig.show()

def wavelet2(data):

    # Realiza la Transformada Wavelet (por ejemplo, usando 'haar' como tipo de wavelet)
    coeffs = pywt.wavedec(data, wavelet='haar', level=None)  # Puedes ajustar el nivel si lo deseas

    # Calcula el tamaño de la escala
    scales = np.arange(1, len(coeffs[0]) + 1)

    # Graficar la transformada wavelet
    fig = go.Figure()

    # Añade la transformada wavelet a la figura
    for i, coeff in enumerate(coeffs):
        fig.add_trace(go.Scatter(x=scales, y=coeff, mode='lines', name=f'Coef. {i}'))

    fig.update_layout(title='Transformada Wavelet de la Serie Temporal de Nivel del Mar',
                    xaxis_title='Escalas',
                    yaxis_title='Coeficientes',
                    showlegend=True)

    # Mostrar la gráfica
    fig.show()

wavelet2(serie)

