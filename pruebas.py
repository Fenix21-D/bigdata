import xarray as xr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs

# Ruta del archivo NetCDF
filename = "chirps-v2.0.2022.daily_acotado.nc"  # Ajusta esta ruta si el archivo está en otro directorio

# Abrir el archivo NetCDF con xarray
ds = xr.open_dataset(filename)

# Verificar las coordenadas y las variables disponibles
print("Dimensiones:", ds.dims)
print("Coordenadas:", ds.coords)
print("Variables disponibles:", ds.variables)

# Suponiendo que la variable de interés es 'precipitation' o similar
# Si tu archivo tiene una variable diferente, reemplaza 'precipitation' por el nombre correcto
var_name = 'precip'  # Ajusta según el nombre de la variable en tu archivo

# Seleccionar la variable de interés
precip = ds[var_name]

# Iterar sobre los primeros 15 días del año (ajustar el rango de días si es necesario)
for day in range(15):
    # Tomar una muestra del tiempo para el día específico
    precip_sample = precip.isel(time=day)
    
    # Crear una figura para la gráfica
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Usar proyección de Cartopy (proyección de Mercator o similar)
    ax = plt.axes(projection=ccrs.PlateCarree())
    
    # Graficar los datos con un mapa de colores
    precip_sample.plot(ax=ax, cmap='viridis', add_colorbar=True)
    
    # Añadir características del mapa (líneas de latitudes y longitudes)
    ax.coastlines()
    ax.gridlines(draw_labels=True)
    
    # Título y etiquetas
    ax.set_title(f"Precipitación - {var_name} - Día {day+1}")
    
    # Mostrar la gráfica
    plt.show()
