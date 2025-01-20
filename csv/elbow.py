import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from csv_ import get_df_serie_temporal

ruta =r"C:\Users\dreng\OneDrive - dimar.mil.co\Documentos\2024\HISTORICO_ACTUALIZADO_2023\BBV\RLS\HOM\Resultados nivelación\BBV_RLS_HOM_20090323-20231231_H_HL.csv"
st = get_df_serie_temporal(ruta)
# Crear características para el clustering (ejemplo: medias semanales)
features = st.resample('W').mean()  # Cambia 'mean' por cualquier otro método que necesites

# Método del codo
inertia = []
k_values = range(1, 11)  # Probar con diferentes valores de k

for k in k_values:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(features)
    inertia.append(kmeans.inertia_)

# Graficar el resultado
plt.plot(k_values, inertia, marker='o')
plt.xlabel('Número de Clústeres')
plt.ylabel('Inercia')
plt.title('Método del Codo')
plt.grid()
plt.show()
