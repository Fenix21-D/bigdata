import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np

def fig_scatter_by_category(serie, params, category=None):
    """
    Genera un gráfico de dispersión donde los puntos están clasificados por categoría con diferentes estilos.

    Args:
        serie (DataFrame): DataFrame con columnas 'value', una columna opcional para categorías, y un índice con fechas.
        params (dict): Parámetros para personalizar el gráfico.
        category (str): Columna que define las categorías de los puntos (opcional).

    Returns:
        fig, ax: Objeto de figura y ejes de Matplotlib para visualizar el scatter plot.
    """
    # Crear la figura y los ejes
    fig, ax = plt.subplots(figsize=(10, 6))

    # Si hay categorías, dividir los datos por ellas
    if category and category in serie.columns:
        unique_categories = serie[category].unique()
        markers = ['o', 's', '^', 'D', 'P', '*', 'x']  # Marcadores disponibles
        colors = plt.cm.tab10.colors  # Colores predefinidos
        for i, cat in enumerate(unique_categories):
            subset = serie[serie[category] == cat]
            ax.scatter(
                subset.index,
                subset['value'],
                label=f"{category}: {cat}",
                marker=markers[i % len(markers)],  # Reusar marcadores si hay muchas categorías
                color=colors[i % len(colors)],    # Reusar colores si hay muchas categorías
                edgecolor='k',
                s=70,
                alpha=0.8
            )
    else:
        # Si no hay categoría, graficar todos los puntos de manera uniforme
        ax.scatter(
            serie.index,
            serie['value'],
            color='blue',
            edgecolor='k',
            s=50,
            alpha=0.7
        )

    # Configurar el formato de fechas en el eje X
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    ax.xaxis.set_major_locator(mdates.AutoDateLocator())
    plt.xticks(rotation=45)

    # Añadir títulos y etiquetas
    ax.set_title(f"{params['variable_name']} de la estación {params['station_name']}")
    ax.set_xlabel("Fecha")
    ax.set_ylabel("Valor del Sensor")

    # Mostrar leyenda si hay categorías
    if category:
        ax.legend(title=category)

    # Ajustar el diseño
    plt.tight_layout()

    return fig, ax
