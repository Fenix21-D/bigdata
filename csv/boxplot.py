import dash
from dash import dcc, html, Input, Output
import plotly.graph_objs as go
import pandas as pd
from csv_ import get_df_serie_temporal
from filter_ import filter_qf

# Ruta al archivo CSV
ruta_prec = r"E:\045-Diego Rengifo 2024\04-PRODUCTOS SERIES TEMPORALES\Precipitaciones\control de calidad\BMA\BMA_PREC_CC.csv"

# Cargar los datos del CSV
serie = get_df_serie_temporal(ruta_prec)
serie = filter_qf(serie, 1)

# Asegurarse de que el índice sea un DatetimeIndex
if not isinstance(serie.index, pd.DatetimeIndex):
    serie['timestamp'] = pd.to_datetime(serie['timestamp'])  # Convertir si es necesario
    serie.set_index('timestamp', inplace=True)


# Agrupar por mes desde el índice y calcular la precipitación acumulada mensual
monthly_sum = serie.groupby(serie.index.to_period('M'))['PREC'].sum().reset_index(name='total_rain')
# Crear la aplicación Dash
app = dash.Dash(__name__)

# Definir el layout de la aplicación
app.layout = html.Div([
    html.H1("Análisis de Precipitación en TUMACO"),
    
    html.Label("Filtrar por Calidad de Datos (QF):"),
    dcc.Checklist(
        id='qf-filter-rn',
        options=[{'label': f'Precipitación: {qf}', 'value': qf} for qf in serie['QF'].unique()],
        value=[]  # Valor inicial vacío
    ),
    
    dcc.Graph(id='rain-timeseries-graph'),  # Serie temporal
    dcc.Graph(id='rain-monthly-histogram'),  # Histograma de promedios mensuales
    dcc.Graph(id='rain-monthly-boxplot')  # Boxplot de acumulación mensual
])

# Callback para actualizar el gráfico de la serie temporal de precipitación
@app.callback(
    Output('rain-timeseries-graph', 'figure'),
    Input('qf-filter-rn', 'value')
)
def update_rain_timeseries_graph(selected_qf):
    if selected_qf:
        filtered_df = serie[serie['QF'].isin(selected_qf)]
    else:
        filtered_df = serie
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=filtered_df.index, y=filtered_df.iloc[:, 0], mode='lines', name='Precipitación (mm)', line=dict(color='green')))
    fig.update_layout(title='Serie Temporal de Precipitación', xaxis_title='Fecha', yaxis_title='Precipitación (mm)')
    return fig

# Callback para actualizar el histograma del promedio mensual acumulado de precipitación
@app.callback(
    Output('rain-monthly-histogram', 'figure'),
    Input('qf-filter-rn', 'value')
)
def update_rain_monthly_histogram(selected_qf):
    if selected_qf:
        filtered_df = serie[serie['QF'].isin(selected_qf)]
    else:
        filtered_df = serie
    
    # Suma mensual filtrada
    monthly_sum_filtered = filtered_df.groupby(filtered_df.index.to_period('ME')).apply(lambda x: x.iloc[:, 0].sum()).reset_index(name='total_rain')
    monthly_sum_filtered['month'] = monthly_sum_filtered['month'].dt.to_timestamp()  # Convertir a timestamp para el eje x
    
    fig = go.Figure()
    
     # Línea de acumulación mensual
    fig.add_trace(go.Scatter(x=monthly_sum_filtered['month'], y=monthly_sum_filtered['total_rain'], mode='lines+markers', name='Precipitación Acumulada (mm)', line=dict(color='red', dash='dash')))
    
    fig.update_layout(title='Promedio Mensual y Precipitación Acumulada', xaxis_title='Mes', yaxis_title='Precipitación (mm)')
    
    return fig

# Callback para actualizar el boxplot de la precipitación acumulada mensual
@app.callback(
    Output('rain-monthly-boxplot', 'figure'),
    Input('qf-filter-rn', 'value')
)
def update_rain_monthly_boxplot(selected_qf):
    if selected_qf:
        filtered_df = serie[serie['QF'].isin(selected_qf)]
    else:
        filtered_df = serie
    
    # Suma mensual filtrada
    monthly_sum_filtered = filtered_df.groupby(filtered_df.index.to_period('M')).apply(lambda x: x.iloc[:, 0].sum()).reset_index(name='total_rain')
    monthly_sum_filtered['month'] = monthly_sum_filtered['month'].dt.to_timestamp()  # Convertir a timestamp para el eje x
    
    fig = go.Figure()
    
    # Crear boxplot de la precipitación acumulada mensual
    fig.add_trace(go.Box(y=monthly_sum_filtered['total_rain'], name='Precipitación Acumulada (mm)', boxmean=True))
    
    fig.update_layout(title='Boxplot de Precipitación Acumulada Mensual', yaxis_title='Precipitación (mm)')
    
    return fig

# Ejecutar la aplicación
if __name__ == '__main__':
    app.run_server(debug=True)
