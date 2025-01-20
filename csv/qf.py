from csv_ import get_df_serie_temporal
from filter_ import filter_qf

ruta = r"E:\045-Diego Rengifo 2024\04-PRODUCTOS SERIES TEMPORALES\Precipitaciones\control de calidad\BMA\BMA_PREC_CC.csv"

serie = get_df_serie_temporal(ruta)
serie_qf = filter_qf(serie, 1)
print(serie_qf.describe())
print(serie)