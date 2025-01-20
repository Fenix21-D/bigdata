#!/usr/bin/python
from http.cookiejar import CookieJar
import urllib.request

# Las credenciales de usuario que se utilizarán para autenticar el acceso a los datos
username = "<Your Earthdata login username>"
password = "<Your Earthdata login password>"

# La URL del archivo que deseamos recuperar
url = "https://daacdata.apps.nsidc.org/pub/DATASETS/nsidc0192_seaice_trends_climo_v3/total-ice-area-extent/nasateam/gsfc.nasateam.month.anomaly.area.1978-2021.s"

# Crea un administrador de contraseñas para manejar la respuesta 401
password_manager = urllib.request.HTTPPasswordMgrWithDefaultRealm()
password_manager.add_password(None, "https://urs.earthdata.nasa.gov", username, password)

# Crea una jarra de cookies para almacenar las cookies de sesión
cookie_jar = CookieJar()

# Instala todos los controladores
opener = urllib.request.build_opener(
    urllib.request.HTTPBasicAuthHandler(password_manager),
    urllib.request.HTTPCookieProcessor(cookie_jar)
)
urllib.request.install_opener(opener)

# Crea y envía la solicitud
request = urllib.request.Request(url)

try:
    response = urllib.request.urlopen(request)
    body = response.read()
    
    # Imprimir el resultado (puede ser binario, dependiendo del archivo)
    print(body)

except urllib.error.HTTPError as e:
    print(f'HTTP Error: {e.code} - {e.reason}')
except urllib.error.URLError as e:
    print(f'URL Error: {e.reason}')
except Exception as e:
    print(f'An unexpected error occurred: {e}')
