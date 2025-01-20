from plotly.subplots import make_subplots
import plotly.express as px
import pandas as pd
import calendar

def fig_boxplot(serie, params, color=None):
    # Crear una nueva columna de mes como tipo de cadena
    serie['month'] = serie.index.month.map(lambda x: calendar.month_name[x])
    frequency = pd.infer_freq(serie.index)
    # Filtrar los valores mayores a 0
    serie = serie[serie['value'] > 0]
    # Crear el gráfico de cajas
    fig = px.box(
        serie, 
        x='month', 
        y='value', 
        color=color, 
        title = (f"Distribución mensual de {params['variable_name']} de {params['station_name']} ({serie.index.min().strftime('%Y-%m-%d')} a {serie.index.max().strftime('%Y-%m-%d')})")

    )
    # Retornar la figura
    return fig

def fig_pie(serie, params):
    """
    Genera un gráfico de pie que muestra la distribución de valores agregados por mes,
    ordenados cronológicamente.

    Args:
        serie (DataFrame): DataFrame que contiene las columnas 'value' y un índice de tipo fecha.
        params (dict): Diccionario con información para el título del gráfico.

    Returns:
        fig (Figure): Figura interactiva de Plotly con el gráfico de pie.
    """
    # Asegúrate de que el índice sea de tipo datetime
    if not isinstance(serie.index, pd.DatetimeIndex):
        raise ValueError("El índice de la serie debe ser de tipo DatetimeIndex.")
    
    # Agrupar por mes y sumar los valores
    monthly_values = serie.groupby(serie.index.month)['value'].sum().reset_index()
    monthly_values.columns = ['month', 'value_sum']
    
    # Mapear los nombres de los meses
    monthly_values['month_name'] = monthly_values['month'].map(lambda x: calendar.month_name[x])
    
    # Ordenar por el número de mes
    monthly_values = monthly_values.sort_values(by='month', ascending=True)  # Orden ascendente por mes
    
    # Crear el gráfico de pie
    fig = px.pie(
        monthly_values, 
        names='month_name', 
        values='value_sum',
        title=f"Distribución de {params['variable_name']} de {params['station_name']} ({params['start_date']} - {params['end_date']})"
    )
    
    return fig

def fig_histogram(serie, params):

    # Crear el histograma
    fig = px.histogram(
        serie,
        x='value',  # Columna del DataFrame a usar para los valores
        nbins=30,  # Número de bins
        title=params.get("title", "Histograma de Precipitación"),
        labels={"value": "Precipitación (mm)"},  # Etiquetas de los ejes
        color_discrete_sequence=["blue"],  # Color de las barras
        opacity=0.75  # Transparencia
    )
    
    # Personalizar el diseño del gráfico
    fig.update_layout(
        xaxis_title=params.get("xaxis_title", "Precipitación (mm)"),
        yaxis_title=params.get("yaxis_title", "Frecuencia"),
        bargap=0.1,
        template="plotly_white"
    )
    
    return fig

def fig_lineplot(serie, params, color=None):
    
    # Crear un gráfico de línea agrupado por el valor de 'qf' para obtener distintas trazas
    fig = px.line(
        serie,
        x=serie.index,
        y=serie['value'],
        color=color,
        title=f"Gráfico de Líneas {params['station_name']},{params['variable_name']}",
        labels={'value': 'Valor del Sensor',  'timestamp': 'Fecha'},
        template='plotly'
    )
    fig.update_traces(connectgaps=False)  # Evitar conectar puntos con NaN

    # Añadir interacción para poder mostrar/ocultar trazas de forma dinámica
    fig.update_traces(mode='lines', marker=dict(size=4), line=dict(width=2))

    return fig


def fig_scatter(serie, params, color=None):
    """
    Genera un gráfico de dispersión (scatter plot) interactivo con colores basados en una columna opcional.

    Args:
        serie (DataFrame): DataFrame que contiene las columnas de datos, 'qf', y un índice con fechas.
        params (dict): Parámetros para personalizar el gráfico.
        color (str): Columna que define los colores de los puntos (opcional).

    Returns:
        fig (Figure): Objeto de figura de Plotly para visualizar el scatter plot.
    """
    # Verificar que las columnas necesarias existan en la serie
    if color and color not in serie.columns:
        raise ValueError(f"La columna '{color}' no se encuentra en la serie de datos.")

    if 'value' not in serie.columns:
        raise ValueError("La columna 'value' no se encuentra en la serie de datos.")
    
    # Crear el gráfico de dispersión
    fig = px.scatter(
        serie,
        x=serie.index,
        y='value',
        color=color,  # Esta columna define los colores
        title=f"{params['variable_name']} de la estación {params['station_name']}",
        labels={'value': 'Valor del Sensor', 'qf': 'Calidad (QF)', 'timestamp': 'Fecha'},
        template='plotly_white'
    )

    # Configurar los puntos para hacerlos más visibles
    fig.update_traces(marker=dict(size=8, opacity=0.7), mode='markers')

    # Ajustar las opciones de interacción y diseño
    fig.update_layout(
        legend_title=dict(text="Calidad (QF)"),
        xaxis_title="Fecha",
        yaxis_title="Valor del Sensor",
        hovermode="closest"
    )

    return fig


def fig_pie_month(serie, params):
    # Asegúrate de que 'month' y 'year' estén presentes
    serie['month'] = serie.index.month
    serie['year'] = serie.index.year
    
    # Obtener los meses únicos
    unique_months = serie['month'].unique()
    num_months = len(unique_months)
    #frequency = pd.infer_freq(serie.index)
    # Crear la figura con subgráficas de tipo domain
    # Crear la figura con subgráficas de tipo domain
    fig = make_subplots(rows=1, cols=num_months, specs=[[{'type': 'domain'}]*len(unique_months)])

    for i, month in enumerate(unique_months):
        month_data = serie[serie['month'] == month]
        report_df = month_data['qf'].value_counts().reset_index()
        report_df.columns = ['qf', 'count']
        pie_chart = px.pie(report_df, values='count', names='qf', title=f'Distribución de QF en {calendar.month_name[month]}')
        
        for trace in pie_chart.data:
            fig.add_trace(trace, row=1, col=i + 1)

    return fig

def fig_boxplot_category(serie, params):
    """
    Genera un gráfico de caja para analizar la distribución mensual de una variable categorizada.

    Args:
        serie (DataFrame): DataFrame con índice de tipo datetime o cadena.
                           Debe contener las columnas 'value' y 'category'.
        params (dict): Diccionario con parámetros para personalizar el gráfico:
                       - 'variable_name': Nombre de la variable analizada.
                       - 'station_name': Nombre de la estación.
                       - 'start_date': Fecha inicial del rango de análisis.
                       - 'end_date': Fecha final del rango de análisis.

    Returns:
        plotly.graph_objs._figure.Figure: Gráfico de caja generado.
    """

    # Convertir el índice a datetime si no lo es ya
    if not pd.api.types.is_datetime64_any_dtype(serie.index):
        try:
            serie.index = pd.to_datetime(serie.index)
        except Exception as e:
            raise ValueError(f"El índice no puede convertirse a datetime: {e}")
    
    # Ordenar el DataFrame por índice (fecha)
    serie = serie.sort_index()

    # Crear una nueva columna con los nombres de los meses
    serie['month'] = serie.index.month.map(lambda x: calendar.month_name[x])
    
    # Determinar la frecuencia del índice
    frequency = pd.infer_freq(serie.index)
    if not frequency:
        try:
            frequency = (serie.index[1] - serie.index[0]).days
            frequency = f"{frequency} días"
        except IndexError:
            frequency = "Indeterminado"
    
    # Filtrar datos omitiendo categorías no deseadas (si existen)
    if 'category' in serie.columns:
        serie = serie[serie['category'] != 'Sin lluvia']
        # Unificar etiquetas similares
        serie['category'] = serie['category'].replace({'Lluvia alta2': 'Lluvia alta'})
    else:
        raise KeyError("La columna 'category' no existe en el DataFrame.")
    
    # Definir el orden correcto de los meses
    month_order = list(calendar.month_name)[1:]  # Excluye el primer elemento vacío

    # Crear el gráfico de caja
    try:
        fig = px.box(
            serie, 
            x='month', 
            y='value', 
            color='category',
            title=(
                f"Distribución mensual de {params['variable_name']} (Frecuencia: {frequency}) "
                f"de {params['station_name']} ({params['start_date']} a {params['end_date']})"
            )
        )
        # Asegurar el orden correcto de los meses en el eje X
        fig.update_xaxes(categoryorder='array', categoryarray=month_order)
    except KeyError as e:
        raise KeyError(f"Falta una columna requerida en el DataFrame: {e}")
    
    return fig

def line_correlacion(serie, params):
    # Graficar con Plotly Express
    fig = px.line(
        serie,
        x="lags",
        y="autocorrelation",
        title="Autocorrelación de la serie temporal de precipitación",
        labels={"lags": "Lags", "autocorrelation": "Autocorrelación"},
        template="plotly_white"
    )

    # Agregar línea horizontal en y=0
    fig.add_hline(y=0, line_dash="dash", line_color="red", annotation_text="y=0", annotation_position="top left")
    return fig

def bar_autocorrelacion(serie, params):
    # Graficar la autocorrelación
    fig = px.bar(serie, x="lags", 
                 y="autocorrelation", 
                 title="Autocorrelación de la Serie Temporal")

    return fig

