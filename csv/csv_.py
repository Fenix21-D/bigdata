import pandas as pd

# Ruta al archivo CSV
#ruta = r"E:\045-Diego Rengifo 2024\04-PRODUCTOS SERIES TEMPORALES\HISTORICO_ACTUALIZADO_2023\TUM\RLS\CC\TUM_RLS_CC.csv"
#ruta_tum = r"C:\Users\dreng\OneDrive - dimar.mil.co\Documentos\2024\HISTORICO_ACTUALIZADO_2023\BSO\RLS\CC\BSO_RLS_CC.csv"
#ruta_bbv = r"C:\Users\dreng\OneDrive - dimar.mil.co\Documentos\2024\HISTORICO_ACTUALIZADO_2023\BBV\RLS\HOM\Resultados nivelación\BBV_RLS_HOM_20090323-20231231_H_HL.csv"
# Cargar los datos del CSV
#serie_original = pd.read_csv(ruta_tum)

import pandas as pd

def get_df_serie_temporal(ruta):
    # Leer el archivo CSV
    serie_original = pd.read_csv(ruta)
    serie_temporal = serie_original.copy()
    # Crear una columna 'timestamp' combinando las columnas 'Fecha' y 'Hora'
    serie_temporal['timestamp'] = serie_temporal['Fecha'] + ' ' + serie_temporal['Hora']
    
    # Convertir la columna 'timestamp' a formato datetime con el formato adecuado
    serie_temporal['timestamp'] = pd.to_datetime(serie_temporal['timestamp'], format='%Y-%m-%d %H:%M:%S')
    
    # Establecer 'timestamp' como el índice del DataFrame
    serie_temporal.set_index('timestamp', inplace=True)

    serie_temporal=serie_temporal.drop(['Fecha', 'Hora'], axis=1)
    
    #print(round(serie_temporal.describe(),3))  # Muestra estadísticas descriptivas de la serie

    return serie_temporal

def _count(serie):
    No_registros= len(serie)
    #print(No_registros)
    #print(serie.count())
    return No_registros

def qf_report(serie):
    # Contar las ocurrencias de cada valor de la columna 'QF'
    conteo_qf = serie['QF'].value_counts(dropna=False)  # Incluir valores NaN en el conteo
    porcentaje = (conteo_qf / len(serie.index)) * 100  # Multiplicamos por 100 para obtener porcentaje
    #print(serie['QF'])

    # Imprimir el conteo y el porcentaje de cada bandera de calidad
    resultado = pd.DataFrame({
        'Conteo': conteo_qf,
        'Porcentaje': porcentaje
    })
    resultado = resultado.sort_index(ascending=True)
    resultado.index.name = 'QF'  # Renombrar el índice para referencia
    resultado.reset_index(inplace=True)  # Asegurar que 'QF' sea una columna

    return resultado



def qf_report_monthly(data_year):
    """
    Genera un reporte mensual de las banderas de calidad (QF) para el año dado.

    Args:
        data_year (pd.DataFrame): DataFrame que contiene datos agrupados por fecha.

    Returns:
        pd.DataFrame: DataFrame consolidado con reportes mensuales de QF, incluyendo la columna 'QF'.
    """
    # Lista para almacenar los reportes de cada mes
    report_list = []

    # Agrupar los datos por mes
    for month, data_month in data_year.groupby(pd.Grouper(freq='M')):
        # Llamada a la función qf_report para cada mes
        report = qf_report(data_month)
        
        # Agregar el nombre del mes al DataFrame de reporte
        report['month'] = month.strftime('%B')
        
        # Añadir el reporte a la lista
        report_list.append(report)

    # Concatenar todos los reportes en un solo DataFrame
    report_df = pd.concat(report_list, ignore_index=True)
    
    # Reordenar el DataFrame para que 'month' sea una columna y 'QF' sea el índice
    #report_df.set_index(['month', 'QF'], inplace=True)
    #print(report_df)


    return report_df  # Devolver el DataFrame final       
        
def remuestrear_serie_temporal(serie_original, nueva_frecuencia, metodo='mean'):
    """
    Remuestrea una serie temporal a una nueva frecuencia.
    
    :param serie_original: DataFrame con la serie temporal
    :param nueva_frecuencia: Frecuencia deseada ('D' para días, 'H' para horas, etc.)
    :param metodo: Método de agregación, por defecto 'mean' (puede ser 'sum', 'min', 'max', etc.)
    :return: Serie temporal remuestreada
    """
    # Accede a la primera columna de la serie
    primera_columna = serie_original.iloc[:, 0]
    
    # Aplicar el remuestreo según la nueva frecuencia
    if metodo == 'mean':
        primera_columna_remuestreada = primera_columna.resample(nueva_frecuencia).mean()
    elif metodo == 'sum':
        primera_columna_remuestreada = primera_columna.resample(nueva_frecuencia).sum()
    elif metodo == 'min':
        primera_columna_remuestreada = primera_columna.resample(nueva_frecuencia).min()
    elif metodo == 'max':
        primera_columna_remuestreada = primera_columna.resample(nueva_frecuencia).max()
    else:
        raise ValueError(f"Método {metodo} no soportado. Prueba con 'mean', 'sum', 'min' o 'max'.")
    
    # Remuestrear las demás columnas usando el primer valor
    otras_columnas_remuestreadas = serie_original.resample(nueva_frecuencia).first()

    # Reemplazar la primera columna en el DataFrame original
    otras_columnas_remuestreadas.iloc[:, 0] = primera_columna_remuestreada
    return otras_columnas_remuestreadas

def agregar_columna_mes(serie):
    """
    Agrega una columna 'Mes' a la serie de datos de precipitación, basada en el índice de tiempo.

    Parámetros:
    - serie (pd.DataFrame): Serie de precipitación con un índice de fecha y hora cada 10 minutos.

    Retorna:
    - pd.DataFrame: Serie con la nueva columna 'Mes'.
    """
    # Verifica que el índice esté en formato datetime
    if not pd.api.types.is_datetime64_any_dtype(serie.index):
        serie.index = pd.to_datetime(serie.index, errors='coerce')
    
    # Añade la columna 'Mes' utilizando el índice
    #serie['month'] = serie.index.month
    #serie['year'] = serie.index.year
    
    return serie

def promedio_acumulado(serie):
    """
    Calcula el promedio acumulado diario de una serie de precipitación.

    Parámetros:
    - serie (pd.DataFrame): Serie de precipitación con un índice de tiempo en formato datetime.

    Retorna:
    - pd.DataFrame: Serie con el promedio acumulado diario de precipitación.
    """
    # Verifica que el índice esté en formato datetime
    if not pd.api.types.is_datetime64_any_dtype(serie.index):
        serie.index = pd.to_datetime(serie.index, errors='coerce')
    
    # Asegura que el nombre de la columna de precipitación sea 'PREC' para el cálculo
    if 'PREC' not in serie.columns:
        raise ValueError("La serie de datos debe contener una columna llamada 'PREC'.")

    # Resample por día y calcula la suma diaria
    suma_diaria = serie['PREC'].resample('D').sum()


    # Convierte a DataFrame y devuelve
    return pd.DataFrame({
        'month': suma_diaria.index,
        'cumrain': suma_diaria.values
    }).set_index('month')
