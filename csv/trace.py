import plotly.graph_objects as go  # Cambiado a graph_objects
import pandas as pd
from csv_ import get_df_serie_temporal

# Ruta a tu archivo CSV
ruta = r"C:\Users\dreng\OneDrive - dimar.mil.co\Documentos\2024\HISTORICO_ACTUALIZADO_2023\BBV\RLS\HOM\Resultados nivelación\BBV_RLS_HOM_20090323-20231231_H_HL.csv"

# Obtener la serie temporal
serie = get_df_serie_temporal(ruta)

# Graficar la serie temporal
fig = go.Figure()
fig.add_trace(go.Scatter(x=serie.index, y=serie['RLS'], mode='lines', name='Serie Temporal'))  # Asegúrate de que 'RLS' es el nombre de la columna
fig.update_layout(title='Serie Temporal de Nivel del Mar',
                  xaxis_title='Fecha',
                  yaxis_title='Nivel del Mar (m)',
                  showlegend=True)

# Mostrar la gráfica
fig.show()
