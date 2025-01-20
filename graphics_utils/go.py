import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import calendar

def multi_line(**kwargs):
    """
    Crea un gráfico de líneas para múltiples series temporales usando **kwargs.
    
    Parámetros:
    - **kwargs: Series temporales nombradas, donde la clave es el nombre de la serie (str)
      y el valor es un DataFrame con un índice de fechas y una columna "value".
    
    Retorna:
    - fig (go.Figure): Figura de Plotly con las series temporales en formato de líneas.
    """
    # Crear la figura
    fig = go.Figure()

    # Colores predefinidos (se puede expandir o hacer dinámico)
    colors = [
        "royalblue", "mediumturquoise", "tomato", "gold", "purple",
        "darkorange", "green", "gray", "pink", "brown"
    ]

    # Agregar cada serie usando kwargs
    for i, (name, serie) in enumerate(kwargs.items()):
        color = colors[i % len(colors)]  # Usar colores cíclicamente si hay más series que colores
        fig.add_trace(go.Scatter(
            x=serie.index,
            y=serie,
            mode="markers",  # Cambiar a solo líneas lines, marcadores markers
            name=name,
            line=dict(color=color)
        ))

    # Personalizar diseño
    fig.update_layout(
        title="Comparación de múltiples series temporales (Líneas)",
        xaxis_title="Fecha",
        yaxis_title="Valor",
        template="ggplot2",
        legend=dict(title="Series"),
    )
    
    return fig

def multi_scatter(**kwargs):
    """
    Crea un gráfico de dispersión para múltiples series temporales usando **kwargs.
    
    Parámetros:
    - **kwargs: Series temporales nombradas, donde la clave es el nombre de la serie (str)
      y el valor es un DataFrame con un índice de fechas y una columna "value".
    
    Retorna:
    - fig (go.Figure): Figura de Plotly con las series temporales.
    """
    # Crear la figura
    fig = go.Figure()

    # Colores predefinidos (se puede expandir o hacer dinámico)
    colors = [
        "royalblue", "mediumturquoise", "tomato", "gold", "purple",
        "darkorange", "green", "gray", "pink", "brown"
    ]

    # Agregar cada serie usando kwargs
    for i, (name, serie) in enumerate(kwargs.items()):
        color = colors[i % len(colors)]  # Usar colores cíclicamente si hay más series que colores
        fig.add_trace(go.Scatter(
            x=serie.index,
            y=serie,
            mode="markers", #"lines+markers"
            name=name,
            line=dict(color=color)
        ))

    # Personalizar diseño
    fig.update_layout(
        title="Comparación de múltiples series temporales",
        xaxis_title="Fecha",
        yaxis_title="Valor",
        template="ggplot2",
        legend=dict(title="Series"),
    )
    
    return fig

import plotly.graph_objects as go

def multi_boxplot(**kwargs):
    """
    Función para comparar múltiples series temporales usando boxplot.
    
    Parameters:
        **kwargs: Argumentos nombrados donde la clave es el nombre de la serie y el valor es un DataFrame.
                  Cada DataFrame debe contener las columnas 'value' y 'month'.
    
    Returns:
        go.Figure: Objeto de Plotly con el gráfico boxplot.
    """
    # Crear el gráfico
    fig = go.Figure()

    # Colores predeterminados para las series
    colors = [
        "royalblue", "mediumturquoise", "salmon", "gold", 
        "lightgreen", "mediumpurple", "coral", "darkorange"
    ]
    color_cycle = iter(colors)

    # Añadir cada serie como un boxplot
    for series_name, data in kwargs.items():
        if 'value' not in data or 'month' not in data:
            raise ValueError(f"La serie '{series_name}' debe contener las columnas 'value' y 'month'.")
        
        fig.add_trace(go.Box(
            y=data["normalized_value"],
            x=data['month'],  # Agrupar por mes
            name=series_name,  # Nombre de la serie
            marker=dict(color=next(color_cycle, "gray")),  # Asignar color o gris si se acaban
            boxmean='sd'  # Mostrar la media y desviación estándar
        ))

    # Personalizar diseño
    fig.update_layout(
        title="Comparación de series temporales con Boxplot",
        xaxis_title="Mes",
        yaxis_title="Valor",
        template="ggplot2",
        legend=dict(title="Series"),
        boxmode='group'  # Agrupar boxplots por mes
    )
    
    return fig

def fig_pie_chart_by_month(serie):
    # Asegúrate de que 'month' y 'year' estén presentes
    serie['month'] = serie.index.month
    serie['year'] = serie.index.year
    
    # Obtener los meses únicos
    unique_months = sorted(serie['month'].unique())
    num_months = len(unique_months)

    # Calcular las filas y columnas necesarias automáticamente
    cols = 3  # Fijo para que sea responsivo (3 columnas)
    rows = (num_months // cols) + (num_months % cols > 0)  # Calcular filas dinámicamente

    # Crear la figura con subgráficas
    fig = make_subplots(
        rows=rows,
        cols=cols,
        specs=[[{'type': 'domain'}] * cols] * rows,  # Todas las subgráficas son de tipo "domain" (para pie charts)
    )

    # Añadir gráficas de pastel por mes
    for i, month in enumerate(unique_months):
        row = (i // cols) + 1  # Fila actual
        col = (i % cols) + 1   # Columna actual

        month_data = serie[serie['month'] == month]
        report_df = month_data['qf'].value_counts().reset_index()
        report_df.columns = ['qf', 'count']

        fig.add_trace(
            go.Pie(
                labels=report_df['qf'],
                values=report_df['count'],
                hole=0.2,  # Gráfico de dona
                textinfo='percent',  # Solo mostrar el porcentaje
                hoverinfo='label+percent+value',  # Mostrar más información en hover
                textfont=dict(size=10),  # Reducir el tamaño del texto dentro de la torta
                title=dict(text=f"{calendar.month_name[month]}", font=dict(size=14), position="top center")
            ),
            row=row,
            col=col
        )

    # Ajustar diseño general
    fig.update_layout(
        title_text="Distribución por Mes",
        title_x=0.5,
        showlegend=False,  # Ocultar leyenda global
        height=rows * 400,  # Altura dinámica
        width=cols * 400,   # Ancho dinámico
        margin=dict(t=50, b=50, l=50, r=50),  # Márgenes generales
        template="plotly"  # Plantilla visual
    )

    return fig


def arrange_figures_in_subplots(figures, rows, cols):
    """
    Toma una lista de figuras generadas previamente (go.Figure) y las acomoda en subplots.
    
    Parameters:
    - figures (list): Una lista de objetos de go.Figure a ser acomodados en subplots.
    - rows (int): El número de filas en la figura combinada de subplots.
    - cols (int): El número de columnas en la figura combinada de subplots.
    
    Returns:
    - fig (go.Figure): Una figura combinada con todos los subplots.
    """
    
    # Crear la figura combinada con subplots
    fig = make_subplots(
        rows=rows, cols=cols, 
        vertical_spacing=0.15, 
        horizontal_spacing=0.1
    )
    
    # Acomodar cada figura en el subplot correspondiente
    fig_idx = 0  # Contador para iterar sobre las figuras
    for i in range(1, rows + 1):
        for j in range(1, cols + 1):
            if fig_idx < len(figures):
                # Añadir las trazas de cada figura a la figura combinada
                for trace in figures[fig_idx].data:
                    fig.add_trace(trace, row=i, col=j)
                fig_idx += 1
    
    # Ajustar el layout de la figura combinada
    fig.update_layout(
        title_text="Subplots Combinados",
        height=300 * rows,  # Ajustar altura en función del número de filas
        width=600 * cols,   # Ajustar ancho en función del número de columnas
        template="ggplot2"
    )
    
    return fig

def fig_histogram(serie, params):
    # Crear el histograma con Plotly
    fig = go.Figure()

    fig.add_trace(
        go.Histogram(
            x=serie['value'],
            nbinsx=30,
            name="Precipitación",
            marker_color="blue",
            opacity=0.75
        )
    )

    fig.update_layout(
        title="Histograma de Precipitación (Simulado)",
        xaxis_title="Precipitación (mm)",
        yaxis_title="Frecuencia",
        bargap=0.1,
        template="ggplot2"
    )

    return fig

def autocorrelogram(serie, lags=40):
    """
    Genera un autocorrelograma para una serie temporal utilizando Plotly.
    
    Parámetros:
    - series (pd.Series): Serie temporal (1D) para calcular las autocorrelaciones.
    - lags (int): Número de retardos a calcular (default: 40).
    
    Retorna:
    - fig (go.Figure): Figura de Plotly con el autocorrelograma.
    """
    # Calcular la autocorrelación para los lags especificados
    autocorr_values = serie['values']
    lag_indices = serie.index
    
    # Crear el gráfico
    fig = go.Figure()

    # Agregar las barras de autocorrelación
    fig.add_trace(go.Bar(
        x=lag_indices,
        y=autocorr_values,
        marker_color="royalblue",
        name="Autocorrelación"
    ))

    # Agregar líneas de significancia (basado en el nivel de confianza)
    confidence_level = 1.96 / np.sqrt(len(serie))
    fig.add_trace(go.Scatter(
        x=[0, lags],
        y=[confidence_level, confidence_level],
        mode="lines",
        line=dict(dash="dash", color="tomato"),
        name="Nivel de confianza (+)"
    ))
    fig.add_trace(go.Scatter(
        x=[0, lags],
        y=[-confidence_level, -confidence_level],
        mode="lines",
        line=dict(dash="dash", color="tomato"),
        name="Nivel de confianza (-)"
    ))

    # Personalizar diseño
    fig.update_layout(
        title="Autocorrelograma",
        xaxis_title="Retardos (Lags)",
        yaxis_title="Autocorrelación",
        template="ggplot2",
        showlegend=True
    )

    return fig