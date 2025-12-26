"""
SCRAPER DE FLIGHTAWARE PARA VERIFICAR CÓDIGOS DE VUELO
Requisitos: pip install requests beautifulsoup4
"""
import requests
from bs4 import BeautifulSoup
import re
from urllib.parse import urljoin
import time
from datetime import datetime

class FlightAwareScraper:
    def __init__(self):
        self.base_url = "https://es.flightaware.com/live/flight/"
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Accept-Language': 'es-ES,es;q=0.9,en;q=0.8'
        })
        self.timeout = 10
        self.retry_delay = 2
        self.max_retries = 3

    def _make_request(self, url):
        """Maneja las peticiones HTTP con reintentos"""
        for attempt in range(self.max_retries):
            try:
                response = self.session.get(url, timeout=self.timeout)
                response.raise_for_status()
                return response
            except requests.exceptions.RequestException as e:
                if attempt == self.max_retries - 1:
                    print(f"Error al hacer la petición a {url}: {str(e)}")
                    return None
                time.sleep(self.retry_delay * (attempt + 1))

    def verificar_vuelo(self, codigo_vuelo):
        """
        Verifica si existe un código de vuelo y obtiene información básica
        
        Args:
            codigo_vuelo (str): Código de vuelo a verificar (ej: 'AA123', 'IB6451')
        
        Returns:
            dict: Información del vuelo o None si no existe
        """
        codigo_vuelo = codigo_vuelo.upper().strip()
        url = f"{self.base_url}{codigo_vuelo}"
        
        print(f"🔍 Verificando vuelo {codigo_vuelo}...")
        
        response = self._make_request(url)
        if not response:
            return {"error": "No se pudo conectar con FlightAware"}
        
        # Verificar si la página indica que el vuelo no existe
        if "Unknown Flight" in response.text:
            print(f"✖ No se encontró información para el vuelo {codigo_vuelo}")
            return {"existe": False, "codigo_vuelo": codigo_vuelo}
        
        # Buscar datos JSON en la página (FlightAware incrusta datos en JavaScript)
        try:
            # Buscar el objeto JSON con los datos del vuelo
            import json
            import re
            
            # Patrón para encontrar el objeto JSON con datos del vuelo
            json_pattern = r'trackpollBootstrap\s*=\s*({.*?});'
            match = re.search(json_pattern, response.text, re.DOTALL)
            
            if match:
                json_data = json.loads(match.group(1))
                
                # Extraer información del vuelo desde el JSON
                flights_data = json_data.get('flights', {})
                flight_id = list(flights_data.keys())[0] if flights_data else None
                
                if flight_id and flights_data[flight_id]:
                    flight_info = flights_data[flight_id]
                    
                    # Extraer información relevante
                    result = {
                        "existe": True,
                        "codigo_vuelo": codigo_vuelo,
                        "url": url,
                        "timestamp": datetime.now().isoformat()
                    }
                    
                    # Aerolínea
                    if 'airline' in flight_info:
                        airline = flight_info['airline']
                        result["aerolinea"] = airline.get('fullName', 'No disponible')
                        result["aerolinea_corto"] = airline.get('shortName', 'No disponible')
                        result["aerolinea_icao"] = airline.get('icao', 'No disponible')
                        result["aerolinea_iata"] = airline.get('iata', 'No disponible')
                    
                    # Origen y destino
                    if 'origin' in flight_info:
                        origin = flight_info['origin']
                        result["origen"] = {
                            "codigo": origin.get('iata', 'No disponible'),
                            "nombre": origin.get('friendlyName', 'No disponible'),
                            "ciudad": origin.get('friendlyLocation', 'No disponible')
                        }
                    
                    if 'destination' in flight_info:
                        dest = flight_info['destination']
                        result["destino"] = {
                            "codigo": dest.get('iata', 'No disponible'),
                            "nombre": dest.get('friendlyName', 'No disponible'),
                            "ciudad": dest.get('friendlyLocation', 'No disponible')
                        }
                    
                    # Estado del vuelo
                    result["estado"] = flight_info.get('flightStatus', 'No disponible')
                    
                    # Tipo de aeronave
                    if 'aircraftType' in flight_info:
                        result["tipo_aeronave"] = flight_info.get('aircraftType', 'No disponible')
                    
                    # Horarios
                    if 'takeoffTimes' in flight_info:
                        times = flight_info['takeoffTimes']
                        result["horarios"] = {
                            "programado": times.get('scheduled'),
                            "estimado": times.get('estimated'),
                            "actual": times.get('actual')
                        }
                    
                    if 'landingTimes' in flight_info:
                        landing_times = flight_info['landingTimes']
                        if "horarios" not in result:
                            result["horarios"] = {}
                        result["horarios"]["aterrizaje"] = {
                            "programado": landing_times.get('scheduled'),
                            "estimado": landing_times.get('estimated'),
                            "actual": landing_times.get('actual')
                        }
                    
                    print(f"✅ Vuelo {codigo_vuelo} encontrado - {result.get('aerolinea', 'N/A')}")
                    return result
            
            # Si no se encuentra JSON, intentar parsear con BeautifulSoup como fallback
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Buscar título de la página para verificar si es un vuelo válido
            title = soup.find('title')
            if title and "Unknown Flight" not in title.get_text():
                # Extraer información básica del HTML
                result = {
                    "existe": True,
                    "codigo_vuelo": codigo_vuelo,
                    "url": url,
                    "timestamp": datetime.now().isoformat(),
                    "metodo": "html_fallback"
                }
                
                # Intentar extraer aerolínea del título
                title_text = title.get_text()
                if "Historial y rastreo de vuelos" in title_text:
                    airline_name = title_text.split("Historial y rastreo de vuelos")[0].strip()
                    result["aerolinea"] = airline_name
                
                print(f"✅ Vuelo {codigo_vuelo} encontrado (método HTML) - {result.get('aerolinea', 'N/A')}")
                return result
                
        except Exception as e:
            print(f"Error procesando la respuesta: {str(e)}")
        
        # Si llegamos aquí, no se pudo determinar si existe o no
        print(f"❓ No se pudo determinar la existencia del vuelo {codigo_vuelo}")
        return {"existe": False, "codigo_vuelo": codigo_vuelo, "error": "No se pudo procesar la respuesta"}

def main():
    # Crear instancia del scraper
    scraper = FlightAwareScraper()
    
    # Ejemplo de uso
    while True:
        print("\n" + "="*50)
        print("  VERIFICADOR DE VUELOS - FLIGHTAWARE")
        print("="*50)
        print("Ingrese un código de vuelo (ej: AA123, IB6451)")
        print("o escriba 'salir' para terminar\n")
        
        codigo = input("Código de vuelo: ").strip()
        
        if codigo.lower() in ['salir', 'exit', 'q']:
            print("\n¡Hasta luego!")
            break
            
        if not codigo:
            print("Por favor ingrese un código de vuelo válido")
            continue
            
        print("\nBuscando información...\n")
        
        # Verificar el vuelo
        resultado = scraper.verificar_vuelo(codigo)
        
        # Mostrar resultados
        if resultado.get("error"):
            print(f"❌ Error: {resultado['error']}")
        elif not resultado.get("existe"):
            print(f"✖ No se encontró información para el vuelo {codigo}")
        else:
            print("✅ VUELO ENCONTRADO")
            print("-" * 40)
            print(f"✈️  Vuelo: {resultado.get('codigo_vuelo', 'N/A')}")
            print(f"🏢 Aerolínea: {resultado.get('aerolinea', 'No disponible')}")
            
            if 'origen' in resultado:
                origen = resultado['origen']
                if isinstance(origen, dict):
                    print(f"🛫 Origen: {origen.get('codigo', 'N/A')} - {origen.get('nombre', 'N/A')}")
                else:
                    print(f"🛫 Origen: {origen}")
            
            if 'destino' in resultado:
                destino = resultado['destino']
                if isinstance(destino, dict):
                    print(f"🛬 Destino: {destino.get('codigo', 'N/A')} - {destino.get('nombre', 'N/A')}")
                else:
                    print(f"🛬 Destino: {destino}")
            
            if 'estado' in resultado:
                print(f"🔄 Estado: {resultado['estado']}")
            
            if 'tipo_aeronave' in resultado:
                print(f"✈️ Tipo de aeronave: {resultado['tipo_aeronave']}")
            
            if 'horarios' in resultado:
                horarios = resultado['horarios']
                print("⏰ Horarios:")
                if 'programado' in horarios:
                    print(f"   Programado: {horarios['programado']}")
                if 'estimado' in horarios:
                    print(f"   Estimado: {horarios['estimado']}")
                if 'actual' in horarios:
                    print(f"   Actual: {horarios['actual']}")
                if 'aterrizaje' in horarios:
                    aterrizaje = horarios['aterrizaje']
                    print("   Aterrizaje:")
                    if 'programado' in aterrizaje:
                        print(f"     Programado: {aterrizaje['programado']}")
                    if 'estimado' in aterrizaje:
                        print(f"     Estimado: {aterrizaje['estimado']}")
                    if 'actual' in aterrizaje:
                        print(f"     Actual: {aterrizaje['actual']}")
            
            print(f"\n🔗 Ver en FlightAware: {resultado.get('url')}")

if __name__ == "__main__":
    print("""
    🛫 VERIFICADOR DE VUELOS - FLIGHTAWARE
    =====================================
    Este script verifica códigos de vuelo en tiempo real
    utilizando el sitio web de FlightAware.
    
    Requisitos:
    - Python 3.6+
    - pip install requests beautifulsoup4
    
    Instrucciones:
    1. Ingresa un código de vuelo (ej: AA123, IB6451)
    2. El script verificará la información más reciente
    3. Verás los detalles del vuelo si existe
    """)
    
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n¡Hasta luego!")
    except Exception as e:
        print(f"\n❌ Error inesperado: {str(e)}")