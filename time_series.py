from conexion import series_temporales
from formats import csv_ts_format
import pandas as pd
import os

# Definir las rutas de los archivos CSV y la información de las estaciones
csv_files_info = [
    {
        "ruta": r"E:\045-Diego Rengifo 2024\04-PRODUCTOS SERIES TEMPORALES\HISTORICO_ACTUALIZADO_2023\TUM\RLS\HOM\Resultados nivelación\TUM_RLS_HOM_20090207-20231231_H_HL.csv",
        "info_station": {
            "instal_place": "CCCP Tumaco - Muelle Guardacosta",
            "hydras_code": "0000TUMACO",
            "name": "EMMA Tumaco",
            "abbreviation": 'TUM',
            "town": "Nariño",
            "departament": "Tumaco",
            "latitude": 1.82011,
            "longitude": -78.72871
        }
    },
    {
        "ruta": r"E:\045-Diego Rengifo 2024\04-PRODUCTOS SERIES TEMPORALES\HISTORICO_ACTUALIZADO_2023\BBV\RLS\HOM\Resultados nivelación\BBV_RLS_HOM_20090323-20231231_H_HL.csv",
        "info_station": {
            "instal_place": "Muelle Capitanía",
            "hydras_code": "CP01JCH016",
            "name": "EMAR Buenaventura",
            "abbreviation": 'BBV',
            "town": "Valle del Cauca",
            "departament": "Buenaventura",
            "latitude": 3.891,
            "longitude": -77.08
        }
    },
    {
        "ruta": r"E:\045-Diego Rengifo 2024\04-PRODUCTOS SERIES TEMPORALES\HISTORICO_ACTUALIZADO_2023\JCH\RLS\HOM\Resultados nivelación\JCH_RLS_HOM_20140822-20231231_H_HL.csv",
        "info_station": {
            "instal_place": "Juanchaco",
            "hydras_code": "CP01BMGA19",
            "name": "EMMA Juanchaco",
            "abbreviation": 'JCH',
            "town": "Valle del Cauca",
            "departament": "Buenaventura",
            "latitude": 3.9151,
            "longitude": -77.359
        }
    },
    {
        "ruta": r"E:\045-Diego Rengifo 2024\04-PRODUCTOS SERIES TEMPORALES\HISTORICO_ACTUALIZADO_2023\BMA\RLS\HOM\Resultados nivelación\BMA_RLS_HOM_20141230-20231231_H_HL.csv",
        "info_station": {
            "instal_place": "Bahía Málaga",
            "hydras_code": "CP01BMGA19",
            "name": "EMMA Bahía Málaga",
            "abbreviation": 'BMA',
            "town": "Valle del Cauca",
            "departament": "Bahía Málaga",
            "latitude": 3.973,
            "longitude": -77.3277
        }
    },
    {
        "ruta": r"E:\045-Diego Rengifo 2024\04-PRODUCTOS SERIES TEMPORALES\HISTORICO_ACTUALIZADO_2023\CAM\RLS\HOM\Resultados nivelación\CAM_RLS_HOM_20220101-20231231_H_HL.csv",
        "info_station": {
            "instal_place": "Candelilla de la mar",
            "hydras_code": "CP02CMAR41",
            "name": "EMAR Candelilla de la mar",
            "abbreviation": 'CAM',
            "town": "Tumaco",
            "departament": "Nariño",
            "latitude": 1.4753,
            "longitude": -78.8459
        }
    },
    {
        "ruta": r"E:\045-Diego Rengifo 2024\04-PRODUCTOS SERIES TEMPORALES\HISTORICO_ACTUALIZADO_2023\BSO\RLS\HOM\Resultados nivelación\BSO_RLS_HOM_20131024-20231231_H_HL.csv",
        "info_station": {
            "instal_place": "Muelle Guardacostas",
            "hydras_code": "CP10BSO017",
            "name": "EMAR Bahía Solano",
            "abbreviation": 'BSO',
            "town": "Bahia Solano",
            "departament": "Choco",
            "latitude": 6.2326,
            "longitude": -77.4128
        }
    }
]

# Función para formatear e insertar datos en MongoDB
def insert_data_from_csv(csv_info):
    ruta_csv = csv_info['ruta']
    info_station = csv_info['info_station']
    
    # Leer el archivo CSV
    df = csv_ts_format(ruta_csv)

    # Formatear el DataFrame para MongoDB
    data_dict = []
    for _, row in df.iterrows():
        data_dict.append({
            "info_station": info_station,
            "data": {
                "sea_level": {
                    'timestamp': row['timestamp'],
                    'value': row['RLS']
                }
            }
        })

    # Insertar los datos en MongoDB
    result = series_temporales.insert_many(data_dict)
    print(f"Datos migrados exitosamente para {info_station['name']} a MongoDB. Número total de registros insertados: {len(result.inserted_ids)}")

# Procesar cada archivo CSV
for csv_info in csv_files_info:
    insert_data_from_csv(csv_info)
