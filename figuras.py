from plotly.subplots import make_subplots

# Función que combina las tres figuras
def combined_figures(boxplot_fig, lineplot_fig, pie_fig):
    # Crear la figura con subgráficas
    num_months = pie_fig['layout']['annotations']
    num_pie = len(num_months)  # Obtiene el número de meses del gráfico de pastel
    fig = make_subplots(rows=3, cols=1, 
                        specs=[[{'type': 'xy'}], [{'type': 'xy'}], [{'type': 'domain'}]],  # Asegúrate de que las filas estén configuradas correctamente
                        subplot_titles=("Boxplot de Valores", "Gráfico de Líneas por QF", "Distribución de Quality Factor por Mes"))

    # Agregar boxplot
    for trace in boxplot_fig.data:
        fig.add_trace(trace, row=1, col=1)

    # Agregar gráfico de líneas
    for trace in lineplot_fig.data:
        fig.add_trace(trace, row=2, col=1)

    # Agregar gráfico de pastel (pie)
    for trace in pie_fig.data:
        fig.add_trace(trace, row=3, col=1)

    # Actualizar el diseño de la figura
    fig.update_layout(title_text="Análisis de Datos Combinados", showlegend=True)

    return fig