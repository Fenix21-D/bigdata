from utils.utils import get_df_serie_temporal, agregar_columna_mes, categorize_precipitation, reshample_time_serie, describe_historical_serie, autocorrelacion
from statistics_utils.express import fig_boxplot, fig_lineplot_by_qf, fig_pie_chart_by_month, fig_pie, fig_scatterplot_by_month, fig_boxplot_categories, line_correlacion, bar_autocorrelacion
from statistics_utils.eda.timeseries import decompose_time_series
from figuras import combined_figures
from api_somo import fetch_and_process_data, fetch_qf
from earth_engine import ee_precipitation_data
from utils.imputations import imputar_knn
from statistics_utils.go import multi_scatter, multi_boxplot, arrange_figures_in_subplots
import kaleido
import numpy as np

# PARAMETROS PARA OBTENER SERIE DE TIEMPO DESDE LAS APIS
api_filtrate_seire = "http://127.0.0.1:8000/time-series/qfcontroler/filtrate_time_serie/"
params = {
    "station_name": "tumaco",
    "variable_name": "Precipitación acumulada",
    "start_date": "2010-01-01",
    "end_date": "2010-12-31"
}
# serie insitu en df
insitu = fetch_and_process_data(api_filtrate_seire, params)
# serie de earth engine
collection = {
    "collection": "UCSB-CHG/CHIRPS/DAILY"  # Cambia esto por tu colección específica
}
ee_serie = ee_precipitation_data(params, collection)

# PRE-PROCESAMIENTO DE LAS SERIES
insitu.loc[insitu['qf'] != 1, 'value'] = np.nan
insitu1 = categorize_precipitation(insitu, params)
imputado = imputar_knn(insitu1, n_neighbors=60)

imputado = reshample_time_serie(imputado, 'D', 'sum')
insitu = reshample_time_serie(insitu, 'D', 'sum')
insitu = categorize_precipitation(insitu, params)
#ee_serie = categorize_precipitation(ee_serie, params)
print('la longitud de la serie insitu es:' , len(insitu))
print('la longitud de la serie imputada es :', len(imputado))


IS = decompose_time_series(insitu)
EE = decompose_time_series(ee_serie)

corr_trend = IS['trend'].iloc[:, 0].corr(EE['trend'].iloc[:, 0])  # Asegúrate de que solo seleccionas una columna
corr_seasonal = IS['seasonal'].iloc[:, 0].corr(EE['seasonal'].iloc[:, 0])  # Asegúrate de que solo seleccionas una columna
corr_resid = IS['resid'].iloc[:, 0].corr(EE['resid'].iloc[:, 0])  # Asegúrate de que solo seleccionas una columna
print('tendencia:', corr_trend)
print('estacionalidad: ', corr_seasonal)
print('residuo: ', corr_trend)

com0 = multi_scatter(InSitu=insitu, Earlth_Engine=ee_serie)
com1 = multi_scatter(InSitu=IS['trend'], Earlth_Engine=EE['trend'])
com2 = multi_scatter(InSitu=IS['seasonal'], Earlth_Engine=EE['seasonal'])
com3 = multi_scatter(InSitu=IS['resid'], Earlth_Engine=EE['resid'])
boxplot_fig_ee = fig_boxplot(insitu,params)
com = arrange_figures_in_subplots([boxplot_fig_ee, com1,com2,com3],2,2)

#autocorrelaciones de las series
#insitu_acorr = autocorrelacion(insitu, params)
#ee_acorr = autocorrelacion(ee_serie, params)

# GRAFICOS
#unitarios
#scatt = fig_scatterplot_by_month(insitu1, params)
#scatt.show() 
'''boxplot_fig_ee = fig_boxplot(ee_serie,params)
boxplot_fig_insitu = fig_boxplot(insitu,params)
lineplot_fig = fig_lineplot_by_qf(insitu, params)
pie_fig_monthly = fig_pie(insitu,params)
pie_fig = fig_pie_chart_by_month(insitu, params)
scatt_fig = fig_scatterplot_by_month(insitu, params)
corr_fig = line_correlacion(ee_serie,params)
corr_bar = bar_autocorrelacion(insitu_acorr, params)'''

#com = multi_scatter(InSitu=IS['trend'], Earlth_Engine=EE['trend'])
#com2 = multi_scatter(Tendencia=IS['trend'], Estacionalidad=IS['seasonal'], Residuo=IS['resid'])
#boxplot_comparations = multi_boxplot(insitu, ee_serie)
com.show()
#com2.show()
#boxplot_comparations.show()

'''
descr_original = describe_historical_serie(serie, params)
descripcion = describe_historical_serie(a, params)
print(descr_original['value'])
print(descripcion['value'])
'''
#boxplot_categorie = fig_boxplot_categories(a, params)
#scatt_fig.show()
#boxplot_fig_ee.show()
#boxplot_fig_insitu.show()
#corr_bar.show()
#fig = fig_pie_chart_by_month(serie.loc["2009"])
#fig = combined_figures(boxplot_fig, lineplot_fig, pie_fig_monthly)
#boxplot_categorie.show()
#boxplot_fig.show()

r_out = f"gorgona/"
# Exportar las figuras como HTML
#boxplot_fig_insitu.write_html(r_out + f"boxplot_{params['variable_name']}, {params['station_name']}.html")
#lineplot_fig.write_html(f"lineplot_{params['variable_name']}, {params['station_name']}.html")
#pie_fig.write_html(r_out + f"pie_{params['variable_name']}, {params['station_name']}.html")
#pie_fig_monthly.write_html(r_out + f"pie_{params['variable_name']}, {params['station_name']}.html")
#scatt_fig.write_html(r_out + f"scatter_{params['variable_name']}, {params['station_name']}.html")


"""# Combinar las figuras
combined_fig = combined_figures(boxplot_fig, lineplot_fig, pie_fig)

# Mostrar la figura combinada
combined_fig.show()
# Exportar la figura como PNG
combined_fig.write_image("grafico_combinado.png")"""

"""from statsmodels.tsa.arima.model import ARIMA

# Definir el modelo ARIMA
model = ARIMA(serie['value'], order=(1, 1, 1))  # (p, d, q)
model_fit = model.fit()

# Resumen del modelo
print(model_fit.summary())
"""