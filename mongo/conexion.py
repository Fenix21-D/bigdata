from pymongo import MongoClient

# Conexión a MongoDB
client = MongoClient('mongodb://localhost:27017/')
db = client['dimar']

# Crear o usar la colección de estaciones y series temporales
variables_collection = db['variables']
stations_collection = db['stations']
timeseries_collection = db['timeseries']
<<<<<<< HEAD

series_temporales = db['series_temporales']
=======
>>>>>>> 2d60dae8d7a2b18eeee1087a81ee3dd8f5443759
