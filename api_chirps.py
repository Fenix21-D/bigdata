import os
import requests
import xarray as xr

# Función para descargar un archivo
def download_chirps_file(url, output_path):
    response = requests.get(url, stream=True)
    if response.status_code == 200:
        with open(output_path, 'wb') as file:
            for chunk in response.iter_content(chunk_size=8192):
                file.write(chunk)
        print(f"Descarga completada: {output_path}")
        return True
    else:
        print(f"Error al descargar {url}. Código de estado: {response.status_code}")
        return False

# Función para procesar un archivo NetCDF
def process_chirps_data(nc_file):
    if not os.path.exists(nc_file):
        print(f"Archivo no encontrado: {nc_file}")
        return None
    try:
        data = xr.open_dataset(nc_file)
        print(data)
        return data
    except Exception as e:
        print(f"Error al procesar el archivo {nc_file}: {e}")
        return None

# URL base de CHIRPS
base_url = "https://data.chc.ucsb.edu/products/CHIRPS-2.0/global_monthly/netcdf"
years = [2020, 2021, 2022]
output_dir = "./chirps_data

# Crear directorio de salida si no existe
os.makedirs(output_dir, exist_ok=True)

# Descargar y procesar los archivos
for year in years:
    filename = f"chirps-v2.0.{year}.monthly.nc"
    url = f"{base_url}/{filename}"
    output_path = os.path.join(output_dir, filename)

    # Descargar archivo
    if download_chirps_file(url, output_path):
        # Procesar archivo si la descarga fue exitosa
        process_chirps_data(output_path)"""
