import xarray as xr
import os

# Ruta local del archivo NetCDF
filename = r"D:\master BigData\01 proyecto final\codigos\bigdata\chirps-v2.0.days_p05\chirps-v2.0.2024.days_p05.nc"  # Ajusta esta ruta si el archivo está en otro directorio

# Verificar si el archivo existe localmente
if os.path.exists(filename):
    print(f"Archivo '{filename}' encontrado localmente.")
    
    # Abrir el archivo con xarray
    ds = xr.open_dataset(filename)

    # Verificar los nombres de las dimensiones del archivo
    print("Dimensiones disponibles:", ds.dims)
    print("Coordenadas disponibles:", ds.coords)

    # Definir los límites geográficos para la Ensenada de Panamá (según los nuevos valores proporcionados)
    lat_min = -1  # Latitud mínima (1°S)
    lat_max = 10  # Latitud máxima (10°N)
    lon_min = -85  # Longitud mínima (85°W)
    lon_max = -76  # Longitud máxima (76°W)

    # Recortar el dataset a la región especificada
    ds_cropped = ds.sel(latitude=slice(lat_min, lat_max), longitude=slice(lon_min, lon_max))

    # Obtener el nombre del archivo original sin la extensión
    base_filename = os.path.splitext(filename)[0]

    # Crear el nombre del archivo de salida con "acotado" al final
    output_filename = f"{base_filename}_acotado.nc"

    # Guardar el dataset recortado
    ds_cropped.to_netcdf(output_filename)

    # Mostrar el nombre del archivo guardado
    print(f"Archivo recortado guardado como: {output_filename}")
else:
    print(f"El archivo '{filename}' no existe en el directorio especificado.")
