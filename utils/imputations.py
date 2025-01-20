import pandas as pd
from sklearn.impute import KNNImputer

# Crear un DataFrame de ejemplo

def imputar_knn(serie, n_neighbors=3):
    """
    Imputa los valores faltantes en una serie temporal utilizando el algoritmo KNN.
    
    Parameters:
    - serie (pd.DataFrame): DataFrame con la columna 'Precipitación' que contiene valores faltantes.
    - n_neighbors (int): Número de vecinos a considerar en el KNN para la imputación (por defecto es 3).
    
    Returns:
    - pd.DataFrame: DataFrame con la serie temporal imputada.
    """
    
    # Asegurarse de que la serie esté ordenada por fecha
    serie = serie.sort_index()
    
    # Crear el imputador KNN
    imputer = KNNImputer(n_neighbors=n_neighbors, weights='distance')
    
    # Aplicar la imputación (usando solo la columna de precipitación)
    serie_imputada = serie.copy()
    serie_imputada['value'] = imputer.fit_transform(serie_imputada[['value']])
        
    return serie_imputada