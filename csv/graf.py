import pandas as pd
import plotly.express as px
from csv_ import get_df_serie_temporal
from filter_ import filter_qf

# Ruta al archivo CSV
ruta = r"E:\045-Diego Rengifo 2024\04-PRODUCTOS SERIES TEMPORALES\HISTORICO_ACTUALIZADO_2023\BSO\RLS\CC\BSO_RLS_CC.csv"
ruta_prec = r"E:\045-Diego Rengifo 2024\04-PRODUCTOS SERIES TEMPORALES\Precipitaciones\control de calidad\BMA\BMA_PREC_CC.csv"
serie = get_df_serie_temporal(ruta_prec)
# Cargar los datos del CSV
serie = filter_qf(serie, 1)
print(serie)
# Seleccionar columnas para graficar (ejemplo: 'timestamp' y alguna columna de datos numéricos, como 'Valor')
# Cambia 'Valor' por el nombre de la columna numérica que deseas graficar
fig = px.line(serie, x=serie.index, y=serie.iloc[:,0], title='Serie temporal de datos')


# Mostrar la gráfica
fig.show()
