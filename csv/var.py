import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.api import VAR
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.stattools import adfuller
from csv_ import get_df_serie_temporal

# Función para comprobar la estacionariedad
def check_stationarity(series):
    result = adfuller(series)
    print(f'ADF Statistic: {result[0]}')
    print(f'p-value: {result[1]}')
    if result[1] <= 0.05:
        print("La serie es estacionaria.")
    else:
        print("La serie no es estacionaria.")

# Cargar tu serie temporal
ruta = r"C:\Users\dreng\OneDrive - dimar.mil.co\Documentos\2024\HISTORICO_ACTUALIZADO_2023\BBV\RLS\HOM\Resultados nivelación\BBV_RLS_HOM_20090323-20231231_H_HL.csv"
serie = get_df_serie_temporal(ruta)
serie = serie['RLS']

# Comprobar estacionariedad
check_stationarity(serie)

# Graficar ACF y PACF
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plot_acf(serie, lags=40, ax=plt.gca())
plt.title('Autocorrelación (ACF)')

plt.subplot(1, 2, 2)
plot_pacf(serie, lags=40, ax=plt.gca())
plt.title('Autocorrelación Parcial (PACF)')

plt.tight_layout()
plt.show()

# Ajustar el modelo VAR con datos de ejemplo
# Para ilustrar el modelo VAR, supongamos que tienes otra serie que podría estar relacionada
# Generar una segunda serie temporal (puedes usar tus propios datos)
np.random.seed(42)
n = len(serie)  # Usar la misma longitud que tu serie
data = {
    'Serie1': serie,
    'Serie2': np.random.randn(n).cumsum() + 20  # Ejemplo de segunda serie
}
df = pd.DataFrame(data)

# Ajustar el modelo VAR
model = VAR(df)
results = model.fit(maxlags=15, ic='aic')  # Ajustar el modelo VAR
print(results.summary())

# Realizar pronósticos
forecast = results.forecast(df.values[-results.k_ar:], steps=5)
forecast_index = pd.date_range(start=df.index[-1] + pd.DateOffset(1), periods=5, freq='H')  # Cambiar a 'M' si es mensual
forecast_df = pd.DataFrame(forecast, index=forecast_index, columns=df.columns)

# Graficar pronósticos
plt.figure(figsize=(12, 6))
plt.plot(df, label='Datos Históricos')
plt.plot(forecast_df, label='Pronósticos', linestyle='--')
plt.title('Pronósticos del Modelo VAR')
plt.xlabel('Fecha')
plt.ylabel('Valores')
plt.legend()
plt.show()
