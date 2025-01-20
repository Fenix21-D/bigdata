from conexion import stations_collection, timeseries_collection
from formats import csv_ts_format

# Cargar datos de sea level desde CSV
ruta_archivo_sealevel = r'E:\045-Diego Rengifo 2024\04-PRODUCTOS SERIES TEMPORALES\HISTORICO_ACTUALIZADO_2023\TUM\RLS\CC\TUM_RLS_CC.csv'
sl = csv_ts_format(ruta_archivo_sealevel)

# Cargar datos de rain desde CSV
ruta_archivo_rain = r"D:\Diego_Rengifo\Repositorio_Datos_REDMPOMM (AROPE)\Pacifico\TUM_EMMA\PREC\ARQ\TUM_PREC_ARQ_2009-2022.csv"
rn =csv_ts_format(ruta_archivo_rain)

# Crear índice en la colección de series temporales para optimizar las consultas
timeseries_collection.create_index([('station_id', 1), ('timestamp', 1)])

# Datos de la estación que se insertarán en la colección de estaciones si no existen
station_data = {
    "station_name": "TUMACO",
    "latitude": 1.8200,
    "longitude": -78.73
}

# Verificar si la estación ya existe en la base de datos, si no, agregarla
station = stations_collection.find_one({"station_name": station_data['station_name']})
if not station:
    station_id = stations_collection.insert_one(station_data).inserted_id
else:
    station_id = station['_id']

# Añadir el campo station_id a ambos DataFrames
sl['station_id'] = station_id
rn['station_id'] = station_id

# Combinar los dos DataFrames por la columna 'timestamp' para alinear ambos conjuntos de datos
combined_df = pd.merge(sl, rn, on=['station_id', 'timestamp'], how='inner')
print(combined_df.columns)

# Formatear el DataFrame para MongoDB
data_dict = []
for _, row in combined_df.iterrows():
    data_dict.append({
        "station_id": row['station_id'],
        "timestamp": row['timestamp'],
        "climate_data": {
            "sealevel": {                
                "value": row['RLS'] if not pd.isna(row['RLS']) else None,
                "QF": row['QF_x'] if not pd.isna(row['QF_x']) else None,
                "PD": row['PD_x'] if not pd.isna(row['PD_x']) else None,
                "units": "m"
            },
            "rain": {
                "value": row['PREC'] if not pd.isna(row['PREC']) else None,
                "QF": row['QF_y'] if not pd.isna(row['QF_y']) else None,
                "PD": row['PD_y'] if not pd.isna(row['PD_y']) else None,
                "units": "mm"
            }
        }
    })

# Insertar los datos en MongoDB
result = timeseries_collection.insert_many(data_dict)
print(f"Datos migrados exitosamente a MongoDB. Número total de registros insertados: {len(result.inserted_ids)}")
