import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt

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
    
    serie_temporal.rename(columns={serie_temporal.columns[0]: 'value'}, inplace=True)

    
    return serie_temporal

def _count(serie):
    if not isinstance(serie.index, pd.DatetimeIndex):
        raise ValueError("El índice de la serie debe ser de tipo DatetimeIndex.")

    # Agrupar por mes y contar registros
    reporte_mensual = serie.resample('6M').count()

    return reporte_mensual

def analyze_time_series(data, value_col='value'):
    """
    Analiza una serie temporal para generar un reporte mensual de conteo
    y visualiza datos ausentes o inconsistencias en la serie.

    Args:
        data (pd.DataFrame): DataFrame con un índice de tipo DatetimeIndex y una columna de valores.
        value_col (str): Nombre de la columna que contiene los valores de interés.

    Returns:
        pd.Series: Conteo mensual de registros de la columna seleccionada.
    """
    if not isinstance(data.index, pd.DatetimeIndex):
        raise ValueError("El índice del DataFrame debe ser de tipo DatetimeIndex.")
    if value_col not in data.columns:
        raise ValueError(f"La columna '{value_col}' no está en el DataFrame.")

    # Generar reporte mensual
    reporte_mensual = data[value_col].resample('ME').count()

    # Gráficos
    fig, ax = plt.subplots(2, 1, sharex=True)

    # Gráfico 1: Serie original
    ax[0].plot(data.index, data[value_col], label="Datos Originales", color="blue", marker='o', alpha=0.7)
    ax[0].set_title("Serie Temporal Original")
    ax[0].set_ylabel(value_col)
    ax[0].grid(True)
    ax[0].legend()

    # Gráfico 2: Conteo mensual
    ax[1].bar(reporte_mensual.index, reporte_mensual.values, color="orange", alpha=0.8, width=20)
    ax[1].set_title("Conteo Mensual de Registros")
    ax[1].set_ylabel("Número de Registros")
    ax[1].grid(True)

    # Ajustes finales
    plt.xlabel("Fecha")
    plt.tight_layout()
    plt.show()

    return reporte_mensual

def analyze_time_series_index(data):
    """
    Analiza una serie temporal usando el índice de tiempo para generar un reporte mensual de conteo
    y visualiza datos ausentes o inconsistencias en la serie.

    Args:
        data (pd.DataFrame): DataFrame con un índice de tipo DatetimeIndex.

    Returns:
        pd.Series: Conteo mensual basado en el índice temporal.
    """
    if not isinstance(data.index, pd.DatetimeIndex):
        raise ValueError("El índice del DataFrame debe ser de tipo DatetimeIndex.")

    # Generar reporte mensual basado en el índice
    reporte_mensual = data.index.to_series().resample('ME').count()

    # Gráficos
    fig, ax = plt.subplots(2, 1, sharex=True)

    # Gráfico 1: Frecuencia de ocurrencia por fecha
    ax[0].plot(data.index, range(len(data.index)), label="Frecuencia Temporal", color="blue", marker='o', alpha=0.7)
    ax[0].set_title("Frecuencia Temporal de Registros")
    ax[0].set_ylabel("Cantidad de Registros")
    ax[0].grid(True)
    ax[0].legend()

    # Gráfico 2: Conteo mensual basado en el índice
    ax[1].bar(reporte_mensual.index, reporte_mensual.values, color="orange", alpha=0.8, width=20)
    ax[1].set_title("Conteo Mensual de Registros (Índice)")
    ax[1].set_ylabel("Número de Registros")
    ax[1].grid(True)

    # Ajustes finales
    plt.xlabel("Fecha")
    plt.tight_layout()
    plt.show()

    return reporte_mensual

def qf_monthly_report(serie):
    """
    Genera un reporte mensual multi-anual sobre los factores de calidad (QF).
    
    :param serie: DataFrame con un índice datetime y una columna 'qf'.
    :return: DataFrame con columnas 'Año-Mes', 'QF', 'Conteo' y 'Porcentaje'.
    """
    # Asegúrate de que el índice sea de tipo datetime
    serie.index = pd.to_datetime(serie.index)
    # Agrega una columna con el periodo Año-Mes
    serie['year_month'] = serie.index.to_period('M')
    
    reportes = []
    for period, group in serie.groupby('year_month'):
        # Calcula el conteo de cada valor de QF en el periodo
        conteo_qf = group['qf'].value_counts(dropna=False)
        # Calcula el porcentaje para cada valor de QF
        porcentaje = (conteo_qf / len(group)) * 100
        # Construye el reporte para el periodo actual
        reporte_periodo = pd.DataFrame({
            'Año-Mes': str(period),  # Año-Mes
            'QF': conteo_qf.index,  # Factor de Calidad
            'Conteo': conteo_qf.values,  # Cantidad de ocurrencias
            'Porcentaje': porcentaje.values  # Porcentaje
        })
        reportes.append(reporte_periodo)
    
    # Concatena todos los reportes
    reporte_final = pd.concat(reportes, ignore_index=True)
    return reporte_final


import numpy as np

def normalize_series(series, method='min-max'):
    """
    Normaliza una serie usando Min-Max Scaling o Z-Score.

    Parameters:
        series (pd.Series): Serie a normalizar.
        method (str): Método de normalización ('min-max' o 'z-score').

    Returns:
        pd.Series: Serie normalizada.
    """
    if method == 'min-max':
        return (series - series.min()) / (series.max() - series.min())
    elif method == 'z-score':
        return (series - series.mean()) / series.std()
    else:
        raise ValueError("El método debe ser 'min-max' o 'z-score'.")



def qf_report_monthly2(data_year):
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


    return report_df  # Devolver el DataFrame final       
        
def reshample_time_serie(serie_original, nueva_frecuencia, metodo='mean'):
    """
    Remuestrea una serie temporal a una nueva frecuencia.
    
    :param serie_original: DataFrame con la serie temporal
    :param nueva_frecuencia: Frecuencia deseada ('D' para días, 'H' para horas, etc.)
    :param metodo: Método de agregación, por defecto 'mean' (puede ser 'sum', 'min', 'max', etc.)
    :return: Serie temporal remuestreada
    """
    # Accede a la primera columna de la serie
    value = serie_original['value']
    
    # Aplicar el remuestreo según la nueva frecuencia
    if metodo == 'mean':
        primera_columna_remuestreada = value.resample(nueva_frecuencia).mean()
    elif metodo == 'sum':
        primera_columna_remuestreada = value.resample(nueva_frecuencia).sum()
    elif metodo == 'min':
        primera_columna_remuestreada = value.resample(nueva_frecuencia).min()
    elif metodo == 'max':
        primera_columna_remuestreada = value.resample(nueva_frecuencia).max()
    elif metodo == 'first':
        primera_columna_remuestreada = value.resample(nueva_frecuencia).first()
    else:
        raise ValueError(f"Método {metodo} no soportado. Prueba con 'mean', 'sum', 'min' o 'max'.")
    
    # Remuestrear las demás columnas usando el primer valor
    otras_columnas_remuestreadas = serie_original.resample(nueva_frecuencia).max()

    # Reemplazar la primera columna en el DataFrame original
    otras_columnas_remuestreadas.iloc[:, 0] = primera_columna_remuestreada
    otras_columnas_remuestreadas = otras_columnas_remuestreadas.sort_index()

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
    serie1 = serie.copy()  # Create a copy of the slice
    serie1.loc[:, 'month'] = serie1.index.month  # This should not raise a warning now

    # Example modification
    serie1.loc[:, 'month'] = serie1.index.month
    #serie['year'] = serie.index.year
    
    return serie1

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
    if 'value' not in serie.columns:
        raise ValueError("La serie de datos debe contener una columna llamada 'PREC'.")

    # Resample por día y calcula la suma diaria
    suma_diaria = serie['value'].resample('D').sum()


    # Convierte a DataFrame y devuelve
    return pd.DataFrame({
        'month': suma_diaria.index,
        'cumrain': suma_diaria.values
    }).set_index('month')

"""def categorize_precipitation(value):
    if value == 0:
        return 'Sin lluvia'
    elif 0 < value <= 10:
        return 'Lluvia ligera'
    elif 10 < value <= 50:
        return 'Lluvia moderada'
    elif 50 < value <= 100:
        return 'Lluvia alta'
    else:
        return 'Lluvia extrema'"""

def categorize_precipitation(serie, params):
    """
    Añade una columna categórica al DataFrame basada en rangos de precipitación.

    Args:
        serie (DataFrame): DataFrame con la columna 'value' que contiene los datos de precipitación.
        params (dict): Parámetros adicionales (si es necesario, para el nombre de columna, etc.).

    Returns:
        DataFrame: DataFrame con una nueva columna 'precipitation_category'.
    """
    serie_category = serie.copy()
    max = serie_category['value'].max()
    min = serie_category['value'].min()


    # Función para categorizar los valores de precipitación
    def categorize(value):
        if value == 0:
            #return 'Sin lluvia'
            return 'Sin lluvia'
        elif 0 < value <= 1*max/4:
            #return 'Lluvia ligera'
            return 'Lluvia ligera'
        elif 1*max/4 < value <= 2*max/4:
            #return 'Lluvia moderada'
            return 'Lluvia moderada'
        elif 2*max/4 < value <= 3*max/4:
            #return 'Lluvia alta'
            return 'Lluvia alta'
        elif 3*max/4 < value <= max:
            #return 'Lluvia alta'
            return 'Lluvia alta2'
        else:
            #return 'Lluvia extrema'
            return 'Lluvia extrema'

    # Crear la nueva columna categórica
    serie_category['category'] = serie_category['value'].apply(categorize)

    return serie_category

def describe_historical_serie(serie, params=None):
    """
    Genera un resumen estadístico de una serie histórica.

    Args:
        serie (pd.Series or pd.DataFrame): Serie o DataFrame a analizar.
        params (dict, opcional): Parámetros adicionales para el análisis (actualmente no utilizados).

    Returns:
        pd.DataFrame: Resumen estadístico generado por pandas.
    """
    # Validar que sea una Serie o un DataFrame
    if not isinstance(serie, (pd.Series, pd.DataFrame)):
        raise TypeError("El argumento 'serie' debe ser un DataFrame o una Serie de pandas.")
    
    # Generar el resumen estadístico
    return serie.describe()

def autocorrelacion(serie, params=None):
    """
    Calcula la autocorrelación de una serie temporal con un número dinámico de lags.
    
    Args:
        serie (pd.DataFrame): DataFrame con una columna 'value' que contiene la serie temporal.
        params (dict, opcional): Un diccionario que puede incluir configuraciones adicionales como el porcentaje de lags.
            Ejemplo: {"lag_percentage": 0.1} (10% del tamaño de la serie).

    Returns:
        pd.DataFrame: DataFrame con las columnas 'lags' y 'autocorrelation'.
    """
    # Determinar el número dinámico de lags basado en el tamaño de la serie
    lag_percentage = params.get("lag_percentage", 0.2) if params else 0.1  # Por defecto, 10% del tamaño
    max_lags = min(int(len(serie["value"]) * lag_percentage), len(serie["value"]) - 1)
    
    # Calcular la autocorrelación usando statsmodels
    acf_values = sm.tsa.acf(serie["value"], nlags=max_lags)
    
    # Crear un DataFrame con los valores de autocorrelación
    acf_df = pd.DataFrame({
        "lags": range(0, len(acf_values)),
        "autocorrelation": acf_values,
    })
    return acf_df


#serie['precipitation_category'] = serie['value'].apply(categorize_precipitation)

