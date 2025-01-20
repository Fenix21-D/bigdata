import ee
import pandas as pd

# Inicializar Google Earth Engine
ee.Authenticate()
ee.Initialize()

# Definir el área de interés (Tumaco, Colombia)
tumaco = ee.Geometry.Point([-78.8077, 1.7986])  # Coordenadas de Tumaco

def extract_precipitation(image, tumaco):
    """
    Extrae valores de precipitación para una imagen en un punto específico (Tumaco).
    """
    # Reducir la región para obtener estadísticas
    stats = image.reduceRegion(
        reducer=ee.Reducer.mean(),
        geometry=tumaco,
        scale=5000,
        maxPixels=1e13
    )
    
    # Agregar propiedades a la imagen
    return image.set("date", image.date().format()).set("precipitation", stats.get("precipitation"))

def ee_precipitation_data(params, collection):
    """
    Procesa una colección de datos de precipitación y genera un DataFrame.
    """
    # Filtrar por rango de fechas y región
    dataset = ee.ImageCollection(collection).filterDate(
        params['start_date'], params['end_date']
    ).filterBounds(tumaco)
    
    # Mapear para extraer datos
    precipitation_data = dataset.map(lambda image: extract_precipitation(image, tumaco))
    
    # Extraer fechas y valores
    dates = precipitation_data.aggregate_array("date").getInfo()
    values = precipitation_data.aggregate_array("precipitation").getInfo()
    
    # Crear DataFrame
    df = pd.DataFrame({"timestamp": dates, "value": values})
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce", utc=True)
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    
    # Agregar columna de mes
    df["month"] = df["timestamp"].dt.month
    
    # Establecer índice
    df.set_index("timestamp", inplace=True)

    return df