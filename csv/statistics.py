import dash
from dash import dcc, html, Input, Output
import plotly.graph_objs as go
import plotly.express as px
import pandas as pd
import calendar
from csv_ import get_df_serie_temporal, agregar_columna_mes, remuestrear_serie_temporal, promedio_acumulado, qf_report, qf_report_monthly
from filter_ import filter_qf

# Ruta al archivo CSV
ruta_prec = r"C:\Users\drengifo\OneDrive - dimar.mil.co\Documentos\2024\HISTORICO_ACTUALIZADO_2023\Precipitaciones\control de calidad\BSO\BSO_PREC_CC.csv"

# Cargar y filtrar los datos del CSV
serie = get_df_serie_temporal(ruta_prec)
report_serie = qf_report(serie)
serie = filter_qf(serie, 1)
serie = agregar_columna_mes(serie)

# Agregar columnas de mes y año
serie['month'] = serie.index.month
serie['year'] = serie.index.year

# Calcular el promedio acumulado diario
serie_acumulada = remuestrear_serie_temporal(serie, 'D', 'sum')

# Crear la aplicación Dash
app = dash.Dash(__name__)

app.layout = html.Div([
    html.H1("Boxplot mensual de precipitaciones"),
    dcc.Dropdown(
        id='year-dropdown',
        options=[{'label': str(year), 'value': year} for year in serie_acumulada['year'].unique()],
        value=serie_acumulada['year'].min(),  # Asegurarse de usar un año válido
        clearable=False
    ),
    dcc.Graph(id='boxplot'),
    dcc.Graph(id='quality-pie-chart')  # Agregar el gráfico de torta aquí
])

@app.callback(
    Output('boxplot', 'figure'),
    Input('year-dropdown', 'value')
)
def update_boxplot(selected_year):
    # Filtrar datos para el año seleccionado
    data_year = serie_acumulada[serie_acumulada['year'] == selected_year]
    data_year.loc[:, 'month'] = data_year['month'].map(lambda x: calendar.month_name[x])
    
    if data_year.empty:
        return go.Figure()  # Devuelve un gráfico vacío si no hay datos
    
    fig = px.box(data_year, x='month', y='PREC', color='month', title='Distribución estacional de precipitaciones',
                 labels={
                     'month': 'Mes',
                     'PREC': 'Precipitación (mm)'
                 })
    
    return fig

import plotly.subplots as sp

@app.callback(
    Output('quality-pie-chart', 'figure'),
    Input('year-dropdown', 'value')
)
def update_quality_pie_chart(selected_year):
    # Filtrar datos para el año seleccionado
    data_year = serie_acumulada[serie_acumulada['year'] == selected_year]
    
    # Generar el reporte de calidad mensual
    report_df = qf_report_monthly(data_year)
    if report_df.empty:
        return go.Figure()  # Devuelve un gráfico vacío si no hay datos

    # Reiniciar el índice para facilitar el uso de los datos
    report_df_reset = report_df.reset_index()

    # Crear subplots para cada mes
    unique_months = report_df_reset['month'].unique()
    num_months = len(unique_months)
    
    fig = sp.make_subplots(rows=1, cols=num_months, specs=[[{'type': 'pie'}] * num_months],subplot_titles=unique_months)

    # Crear un gráfico de torta para cada mes
    for i, month in enumerate(unique_months):
        month_data = report_df_reset[report_df_reset['month'] == month]
        
        fig.add_trace(go.Pie(
            labels=month_data['QF'].astype(str),
            values=month_data['Conteo'],
            name=month,
            title=f'Distribución de QF en {month}',
            hole=0.4
        ), row=1, col=i + 1)

    # Configurar el diseño
    fig.update_layout(
        title=f'Distribución de Calidad por Mes para el Año {selected_year}',
        showlegend=True,
        height=400  # Ajusta la altura según lo necesites
    )

    return fig




if __name__ == '__main__':
    app.run_server(debug=True)
