import pandas as pd

def filter_qf(serie, qf):
    serie_qf = serie.copy()  # Hacer una copia para no modificar la original
    serie_qf.loc[serie_qf["QF"] != qf, :] = pd.NA  # Reemplazar los valores donde "QF" no coincide
    return serie_qf

