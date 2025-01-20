import pandas as pd
from datetime import datetime

# Cargar datos de sea level desde CSV
ruta_archivo_sealevel = r'E:\045-Diego Rengifo 2024\04-PRODUCTOS SERIES TEMPORALES\HISTORICO_ACTUALIZADO_2023\TUM\RLS\CC\TUM_RLS_CC.csv'

def csv_ts_format(dir):
    
    ts = pd.read_csv(dir)
    ts['timestamp'] = pd.to_datetime(ts['Fecha'] + ' ' + ts['Hora'], format='%Y-%m-%d %H:%M:%S')
    ts.drop(columns=['Fecha', 'Hora'], inplace=True)
    return ts