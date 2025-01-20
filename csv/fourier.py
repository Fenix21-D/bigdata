import pandas as pd
import numpy as np
import plotly.graph_objects as go
from scipy.fft import fft
from csv_ import get_df_serie_temporal

# Ruta a tu archivo CSV
ruta = r"C:\Users\dreng\OneDrive - dimar.mil.co\Documentos\2024\HISTORICO_ACTUALIZADO_2023\BBV\RLS\HOM\Resultados nivelación\BBV_RLS_HOM_20090323-20231231_H_HL.csv"
serie = get_df_serie_temporal(ruta)

# Selecciona la columna de nivel del mar
serie = serie['RLS']
serie = serie - np.mean(serie)

# Realiza la Transformada de Fourier
n = len(serie)  # Número de puntos
frecuencia = np.fft.fftfreq(n, d=1)  # d=1 para suponer que el intervalo de muestreo es 1 hora
# Normaliza los datos (opcional)
serie_normalizada = (serie - serie.mean()) / serie.std()

# Realiza la Transformada de Fourier sobre los datos normalizados
transformada = fft(serie_normalizada)
magnitud = np.abs(transformada)
print("Máxima Magnitud:", magnitud.max())  # Verifica la magnitud máxima


# Magnitudes
magnitud = np.abs(transformada)

# Graficar con Plotly
fig = go.Figure()
# Graficar solo la mitad positiva de la transformada (frecuencia positiva)
fig.add_trace(go.Scatter(x=frecuencia[:n // 2], y=magnitud[:n // 2], mode='lines', name='Transformada de Fourier'))
fig.update_layout(title='Transformada de Fourier de la Serie Temporal de Nivel del Mar',
                  xaxis_title='Frecuencia (Hz)',
                  yaxis_title='Magnitud (m)',
                  showlegend=True)

# Mostrar la gráfica
fig.show()
