import pandas as pd
from pymongo import MongoClient
from datetime import datetime

# Conexión a MongoDB
client = MongoClient('mongodb://localhost:27017/')
db = client['dimar']
coleccion = db['prueba1']

# Cargar datos desde CSV
ruta_archivo = r"E:\045-Diego Rengifo 2024\04-PRODUCTOS SERIES TEMPORALES\Precipitaciones\control de calidad\TUM\TUM_PREC_CC.csv"
df = pd.read_csv(ruta_archivo)

# Mostrar las primeras filas del DataFrame para verificar los datos cargados
print("Primeras filas del archivo CSV:")
print(df.head())

# Unir 'Fecha' y 'Hora' en una columna 'date_time' con formato datetime
df['date_time'] = pd.to_datetime(df['Fecha'] + ' ' + df['Hora'], format='%Y-%m-%d %H:%M:%S')

# Eliminar las columnas 'Fecha' y 'Hora' ya que ahora tenemos 'date_time'
columnas_eliminadas = ['Fecha', 'Hora']
df.drop(columns=columnas_eliminadas, inplace=True)

# Añadir los campos adicionales
df['station'] = 'BUENAVENTURA'  # Nombre de la estación
df['variable'] = 'RLS'              # Variable que estás midiendo
df['units'] = 'm'                 # Unidad de la variable medida
df['latitude'] = 1.2345             # Latitud de la estación
df['longitude'] = -73.5678          # Longitud de la estación

# Mostrar el DataFrame modificado con los nuevos campos
print("Vista previa de los datos modificados con los nuevos campos (primeras 5 filas):")
#print(df.head())

# Convertir DataFrame a diccionario y cargar a MongoDB
data_dict = df.to_dict("records")
#print(data_dict[:3])

# Insertar datos en MongoDB y contar los registros insertados
result = coleccion.insert_many(data_dict)
print(f"Datos migrados exitosamente a MongoDB. Número total de registros insertados: {len(result.inserted_ids)}")