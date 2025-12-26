"""
VERIFICADOR DE MATRÍCULAS DE AVIÓN USANDO LA API OFICIAL DE OPENSKY
Requisitos: pip install opensky-api
Documentación: https://openskynetwork.github.io/opensky-api/python.html
"""
from opensky_api import OpenSkyApi
import pandas as pd
import re
from datetime import datetime, timedelta
import time
import os

class OpenSkyAvionChecker:
    def __init__(self, username=None, password=None):
        """
        Inicializa el verificador con la API oficial de OpenSky
        
        Args:
            username: Opcional, para límites más altos
            password: Opcional, para límites más altos
        """
        self.api = OpenSkyApi(username=username, password=password)
        self.db_loaded = False
        self.aircraft_df = None
        
        # Patrones de matrículas internacionales
        self.patrones_matricula = {
            'ES': r'^EC-[A-Z]{3,4}$',      # España
            'US': r'^N[0-9]{1,5}[A-Z]?$',  # Estados Unidos
            'MX': r'^XA-[A-Z]{3}$',        # México
            'DE': r'^D-[A-Z]{4}$',         # Alemania
            'UK': r'^G-[A-Z]{4}$',         # Reino Unido
            'FR': r'^F-[A-Z]{4}$',         # Francia
            'BR': r'^PR-[A-Z]{3}$',        # Brasil
            'AR': r'^LV-[A-Z]{3}$',        # Argentina
            'CL': r'^CC-[A-Z]{3}$',        # Chile
            'CO': r'^HK-[0-9]{3}[A-Z]?$',  # Colombia
            'PE': r'^OB-[0-9]{4}$',        # Perú
            'VE': r'^YV[0-9]{3}[A-Z]?$',   # Venezuela
        }
    
    def validar_formato_matricula(self, matricula):
        """
        Valida el formato de una matrícula y devuelve el país probable
        """
        matricula = matricula.upper().strip()
        
        for pais, patron in self.patrones_matricula.items():
            if re.match(patron, matricula):
                return True, pais, matricula
        
        # Formato general como respaldo
        if re.match(r'^[A-Z]{1,3}-?[A-Z0-9]{1,6}$', matricula):
            return True, 'DESCONOCIDO', matricula
        
        return False, None, matricula
    
    def cargar_base_datos(self, ruta_csv='aircraftDatabase.csv'):
        """
        Carga la base de datos de aeronaves desde el archivo CSV
        """
        try:
            if os.path.exists(ruta_csv):
                print(f"📂 Cargando base de datos desde {ruta_csv}...")
                self.aircraft_df = pd.read_csv(
                    ruta_csv, 
                    dtype=str,
                    keep_default_na=False
                )
                self.db_loaded = True
                print(f"✅ Base de datos cargada: {len(self.aircraft_df)} registros")
                return True
            else:
                print(f"❌ Archivo {ruta_csv} no encontrado")
                print("💡 Descárgalo de: https://opensky-network.org/datasets/metadata/")
                return False
        except Exception as e:
            print(f"❌ Error cargando base de datos: {str(e)}")
            return False
    
    def buscar_en_base_datos(self, busqueda):
        """
        Busca en la base de datos local por matrícula, ICAO, modelo, etc.
        """
        if not self.db_loaded:
            print("⚠️  Base de datos no cargada. Usa cargar_base_datos() primero")
            return []
        
        busqueda = busqueda.upper().strip()
        
        try:
            # Crear máscaras de búsqueda para diferentes columnas
            masks = []
            
            # Buscar en matrícula (registration)
            if 'registration' in self.aircraft_df.columns:
                masks.append(self.aircraft_df['registration'].str.upper().str.contains(busqueda, na=False))
            
            # Buscar en ICAO24
            if 'icao24' in self.aircraft_df.columns:
                masks.append(self.aircraft_df['icao24'].str.upper().str.contains(busqueda, na=False))
           
            # Combinar todas las máscaras
            if masks:
                combined_mask = masks[0]
                for mask in masks[1:]:
                    combined_mask = combined_mask | mask
                
                resultados = self.aircraft_df[combined_mask]
                
                if not resultados.empty:
                    print(f"✅ Encontrados {len(resultados)} resultado(s) en base de datos")
                    return resultados
                else:
                    print("❌ No encontrado en base de datos")
                    return pd.DataFrame()
            else:
                print("⚠️  Columnas de búsqueda no encontradas en el CSV")
                return pd.DataFrame()
                
        except Exception as e:
            print(f"❌ Error en búsqueda: {str(e)}")
            return pd.DataFrame()
    
    def obtener_vuelos_actuales(self):
        """
        Obtiene todos los vuelos actuales usando la API
        """
        try:
            print("🛫 Obteniendo vuelos actuales...")
            states = self.api.get_states()
            
            if states and states.states:
                print(f"✅ {len(states.states)} aviones en vuelo ahora")
                return states.states
            else:
                print("ℹ️  No hay datos de vuelos actuales")
                return []
                
        except Exception as e:
            print(f"❌ Error obteniendo vuelos: {str(e)}")
            return []
    
    def buscar_avion_en_vuelo(self, busqueda):
        """
        Busca un avión específico en los vuelos actuales
        """
        try:
            states = self.api.get_states()
            
            if not states or not states.states:
                return []
            
            busqueda = busqueda.upper().strip()
            resultados = []
            
            for state in states.states:
                encontrado = False
                
                # Buscar por callsign (vuelo)
                if state.callsign and busqueda in state.callsign.strip().upper():
                    encontrado = True
                
                # Buscar por ICAO (si la búsqueda parece un código ICAO)
                if not encontrado and state.icao24 and busqueda == state.icao24.upper():
                    encontrado = True
                
                # Buscar por matrícula en base de datos (si está cargada)
                if not encontrado and self.db_loaded:
                    # Buscar el ICAO en nuestra base de datos
                    db_result = self.aircraft_df[
                        self.aircraft_df['icao24'].str.upper() == state.icao24.upper()
                    ]
                    if not db_result.empty:
                        matricula = db_result.iloc[0]['registration']
                        if matricula and busqueda in matricula.upper():
                            encontrado = True
                
                if encontrado:
                    resultados.append({
                        'icao24': state.icao24,
                        'callsign': state.callsign.strip() if state.callsign else 'N/A',
                        'matricula': self.obtener_matricula_de_icao(state.icao24),
                        'origen_pais': state.origin_country,
                        'posicion': (state.latitude, state.longitude) if state.latitude else None,
                        'altitud': state.baro_altitude,
                        'velocidad': state.velocity,
                        'rumbo': state.true_track,
                        'en_tierra': state.on_ground,
                        'timestamp': state.time_position,
                        'estado': state
                    })
            
            return resultados
            
        except Exception as e:
            print(f"❌ Error en búsqueda en vuelo: {str(e)}")
            return []
    
    def obtener_matricula_de_icao(self, icao24):
        """
        Obtiene la matrícula de un ICAO desde la base de datos
        """
        if self.db_loaded and icao24:
            resultado = self.aircraft_df[
                self.aircraft_df['icao24'].str.upper() == icao24.upper()
            ]
            if not resultado.empty:
                return resultado.iloc[0]['registration']
        return None
    
    def obtener_info_completa_icao(self, icao24):
        """
        Obtiene información completa de un avión por su ICAO24
        """
        if not self.db_loaded:
            return None
        
        resultado = self.aircraft_df[
            self.aircraft_df['icao24'].str.upper() == icao24.upper()
        ]
        
        if not resultado.empty:
            avion = resultado.iloc[0]
            info = {
                'icao24': avion.get('icao24', 'N/A'),
                'matricula': avion.get('registration', 'N/A'),
                'fabricante': avion.get('manufacturername', avion.get('manufacturericao', 'N/A')),
                'modelo': avion.get('model', 'N/A'),
                'tipo_codigo': avion.get('typecode', 'N/A'),
                'numero_serie': avion.get('serialnumber', 'N/A'),
                'operador': avion.get('operator', 'N/A'),
                'callsign_operador': avion.get('operatorcallsign', 'N/A'),
                'codigo_operador': avion.get('operatoricao', 'N/A'),
                'propietario': avion.get('owner', 'N/A'),
                'registrado': avion.get('registered', 'N/A'),
                'estatus': avion.get('status', 'N/A'),
                'construido': avion.get('built', 'N/A'),
                'tipo_aeronave': avion.get('icaoaircrafttype', 'N/A'),
                'motores': avion.get('engines', 'N/A'),
                'config_asientos': avion.get('seatconfiguration', 'N/A'),
                'modo_s': avion.get('modes', 'N/A'),
                'adsb': avion.get('adsb', 'N/A'),
                'categoria': avion.get('categoryDescription', 'N/A'),
                'notas': avion.get('notes', 'N/A')
            }
            return info
        
        return None
    
    def obtener_historial_vuelos(self, icao24, horas=24):
        """
        Obtiene el historial de vuelos de un avión
        Nota: Requiere autenticación para períodos largos
        """
        try:
            end = int(time.time())
            start = end - (horas * 3600)
            
            print(f"📅 Obteniendo historial de últimas {horas} horas para {icao24}...")
            flights = self.api.get_flights_by_aircraft(icao24, start, end)
            
            if flights:
                print(f"✅ {len(flights)} vuelo(s) encontrado(s)")
                vuelos_info = []
                for flight in flights:
                    vuelo_info = {
                        'callsign': flight.callsign,
                        'primer_contacto': datetime.fromtimestamp(flight.firstSeen).strftime('%Y-%m-%d %H:%M:%S'),
                        'ultimo_contacto': datetime.fromtimestamp(flight.lastSeen).strftime('%Y-%m-%d %H:%M:%S'),
                        'origen': flight.estDepartureAirport,
                        'destino': flight.estArrivalAirport
                    }
                    vuelos_info.append(vuelo_info)
                
                return vuelos_info
            else:
                print("ℹ️  No se encontraron vuelos en este período")
                return []
                
        except Exception as e:
            print(f"❌ Error obteniendo historial: {str(e)}")
            return []
    
    def buscar_vuelo_por_callsign(self, callsign, buscar_historico=False, horas_historia=24):
        """
        Busca vuelos por su código de vuelo (callsign) como 'BAW2203'.
        
        Args:
            callsign (str): Código de vuelo a buscar (ej: 'BAW2203').
            buscar_historico (bool): Si es True, también busca en vuelos históricos.
            horas_historia (int): Horas hacia atrás para buscar en el historial.
        
        Returns:
            dict: Diccionario con resultados de vuelos activos e históricos.
        """
        print(f"\n{'='*70}")
        print(f"🔍 BUSCANDO VUELO POR CALLSIGN: {callsign}")
        print('='*70)
        
        # Limpiar el callsign (sin espacios)
        callsign_limpio = callsign.strip().upper()
        resultados = {'activos': [], 'historicos': [], 'encontrado': False}
        
        # 1. BUSCAR EN VUELOS ACTIVOS (TIEMPO REAL)
        print("\n1. 🛫 Consultando vuelos activos...")
        try:
            # Usar la API para obtener todos los estados actuales
            states = self.api.get_states()
            
            if states and states.states:
                vuelos_activos = []
                for state in states.states:
                    if state.callsign and state.callsign.strip().upper() == callsign_limpio:
                        # Obtener matrícula del avión desde la base de datos
                        matricula = self.obtener_matricula_de_icao(state.icao24) if self.db_loaded else None
                        
                        vuelo_info = {
                            'callsign': state.callsign.strip(),
                            'icao24': state.icao24,
                            'matricula': matricula,
                            'origen_pais': state.origin_country,
                            'posicion': (state.latitude, state.longitude) if state.latitude else None,
                            'altitud_barometrica': state.baro_altitude,
                            'velocidad': state.velocity,
                            'rumbo': state.true_track,
                            'en_tierra': state.on_ground,
                            'timestamp_posicion': datetime.fromtimestamp(state.time_position).isoformat() 
                                                if state.time_position else None,
                            'ultimo_contacto': datetime.fromtimestamp(state.last_contact).isoformat() 
                                            if state.last_contact else None
                        }
                        vuelos_activos.append(vuelo_info)
                
                if vuelos_activos:
                    print(f"   ✅ VUELO ENCONTRADO EN TIEMPO REAL ({len(vuelos_activos)} instancia(s))")
                    for vuelo in vuelos_activos:
                        print(f"\n   ✈️  Callsign: {vuelo['callsign']}")
                        print(f"   📍 Posición: {vuelo['posicion'] if vuelo['posicion'] else 'No disponible'}")
                        print(f"   🏷️  Matrícula: {vuelo['matricula'] or 'No disponible'}")
                        print(f"   🏳️  País origen: {vuelo['origen_pais']}")
                        print(f"   📈 Altitud: {vuelo['altitud_barometrica']:.0f}m" if vuelo['altitud_barometrica'] else "   📈 Altitud: No disponible")
                        print(f"   🚀 Velocidad: {vuelo['velocidad']:.0f} m/s" if vuelo['velocidad'] else "   🚀 Velocidad: No disponible")
                        print(f"   🧭 Rumbo: {vuelo['rumbo']:.0f}°" if vuelo['rumbo'] else "   🧭 Rumbo: No disponible")
                        print(f"   ⏰ Última actualización: {vuelo['ultimo_contacto']}")
                    
                    resultados['activos'] = vuelos_activos
                    resultados['encontrado'] = True
                else:
                    print(f"   ℹ️  No se encontró el vuelo {callsign_limpio} en vuelos activos.")
            else:
                print("   ℹ️  No hay datos de vuelos activos disponibles.")
                
        except Exception as e:
            print(f"   ❌ Error consultando vuelos activos: {str(e)}")
        
        # 2. BUSCAR EN HISTORIAL (OPCIONAL - REQUIERE AUTENTICACIÓN)
        if buscar_historico:
            print(f"\n2. 📅 Consultando historial de últimas {horas_historia} horas...")
            try:
                # Calcular timestamps para el período histórico
                fin = int(time.time())
                inicio = fin - (horas_historia * 3600)
                
                # Obtener vuelos históricos
                # NOTA: Este endpoint podría requerir autenticación para períodos largos
                historial = self.api.get_flights_by_aircraft(
                    icao24=None,  # No filtramos por avión específico
                    begin=inicio,
                    end=fin
                )
                
                if historial:
                    vuelos_historicos = []
                    for flight in historial:
                        if flight.callsign and flight.callsign.strip().upper() == callsign_limpio:
                            vuelo_hist = {
                                'callsign': flight.callsign.strip(),
                                'primer_contacto': datetime.fromtimestamp(flight.firstSeen).isoformat(),
                                'ultimo_contacto': datetime.fromtimestamp(flight.lastSeen).isoformat(),
                                'aeropuerto_salida': flight.estDepartureAirport,
                                'aeropuerto_llegada': flight.estArrivalAirport,
                                'icao24': flight.icao24
                            }
                            vuelos_historicos.append(vuelo_hist)
                    
                    if vuelos_historicos:
                        print(f"   ✅ {len(vuelos_historicos)} vuelo(s) histórico(s) encontrado(s)")
                        for i, vuelo in enumerate(vuelos_historicos, 1):
                            print(f"\n   🗓️  Vuelo histórico {i}:")
                            print(f"   • Callsign: {vuelo['callsign']}")
                            print(f"   • Aeropuerto salida: {vuelo['aeropuerto_salida'] or 'Desconocido'}")
                            print(f"   • Aeropuerto llegada: {vuelo['aeropuerto_llegada'] or 'Desconocido'}")
                            print(f"   • Horario: {vuelo['primer_contacto']} a {vuelo['ultimo_contacto']}")
                            print(f"   • ICAO24: {vuelo['icao24']}")
                        
                        resultados['historicos'] = vuelos_historicos
                        resultados['encontrado'] = True
                    else:
                        print(f"   ℹ️  No se encontró el vuelo {callsign_limpio} en el historial consultado.")
                else:
                    print("   ℹ️  No hay datos históricos disponibles para el período.")
                    
            except Exception as e:
                print(f"   ❌ Error consultando historial: {str(e)}")
                print("   💡 Nota: El acceso a datos históricos puede requerir autenticación.")
        
        # Resumen final
        print(f"\n{'='*70}")
        if resultados['encontrado']:
            total = len(resultados['activos']) + len(resultados['historicos'])
            print(f"✅ BÚSQUEDA COMPLETADA: {total} resultado(s) para '{callsign_limpio}'")
        else:
            print(f"❌ No se encontró información para el vuelo '{callsign_limpio}'")
            print("\n💡 Posibles razones:")
            print("   • El vuelo no está activo en este momento")
            print("   • El código de vuelo podría haber cambiado (vuelos compartidos)")
            print("   • El vuelo no pasó por la cobertura de la red OpenSky")
            print("   • Para búsqueda histórica, necesita autenticación o el período es muy corto")
        
        return resultados
        
    def verificar_matricula(self, entrada):
        """
        Verificación completa de una matrícula o callsign
        """
        print(f"\n{'='*70}")
        print(f"🔍 VERIFICACIÓN: {entrada}")
        print('='*70)
        
        # Paso 1: Validar formato
        valido, pais, entrada_limpia = self.validar_formato_matricula(entrada)
        
        if valido:
            print(f"✅ Formato válido - País probable: {pais}")
        else:
            print("⚠️  Formato no reconocido, pero continuando búsqueda...")
        
        resultados = []
        
        # Paso 2: Buscar en base de datos (si está cargada)
        if self.db_loaded:
            print("\n📂 Buscando en base de datos local...")
            db_resultados = self.buscar_en_base_datos(entrada_limpia)
            
            if not db_resultados.empty:
                print("🎯 RESULTADOS EN BASE DE DATOS:")
                for idx, avion in db_resultados.head(3).iterrows():  # Mostrar máximo 3
                    print(f"\n  Avión {idx + 1}:")
                    print(f"  • Matrícula: {avion.get('registration', 'N/A')}")
                    print(f"  • ICAO24: {avion.get('icao24', 'N/A')}")
                    print(f"  • Modelo: {avion.get('model', 'N/A')}")
                    print(f"  • Fabricante: {avion.get('manufacturername', 'N/A')}")
                    print(f"  • Operador: {avion.get('operator', 'N/A')}")
                    print(f"  • Tipo: {avion.get('typecode', 'N/A')}")
                    
                    # Guardar para posible uso posterior
                    resultados.append({
                        'tipo': 'base_datos',
                        'icao24': avion.get('icao24'),
                        'matricula': avion.get('registration'),
                        'info': avion
                    })
        
        # Paso 3: Buscar en vuelos actuales
        print("\n🛫 Buscando en vuelos actuales...")
        vuelos = self.buscar_avion_en_vuelo(entrada_limpia)
        
        if vuelos:
            print(f"✈️  ENCONTRADO EN VUELO: {len(vuelos)} avión(es) activo(s)")
            
            for i, vuelo in enumerate(vuelos, 1):
                print(f"\n  Vuelo {i}:")
                print(f"  • Callsign: {vuelo['callsign']}")
                print(f"  • Matrícula: {vuelo['matricula'] or 'No disponible'}")
                print(f"  • ICAO24: {vuelo['icao24']}")
                print(f"  • País: {vuelo['origen_pais']}")
                
                if vuelo['posicion']:
                    lat, lon = vuelo['posicion']
                    print(f"  • Posición: {lat:.4f}°, {lon:.4f}°")
                
                if vuelo['altitud']:
                    print(f"  • Altitud: {vuelo['altitud']:.0f} m")
                
                if vuelo['velocidad']:
                    print(f"  • Velocidad: {vuelo['velocidad']:.0f} m/s ({(vuelo['velocidad'] * 1.94384):.0f} nudos)")
                
                if vuelo['rumbo']:
                    print(f"  • Rumbo: {vuelo['rumbo']:.0f}°")
                
                print(f"  • En tierra: {'Sí' if vuelo['en_tierra'] else 'No'}")
                
                # Obtener información adicional de la base de datos
                if vuelo['icao24']:
                    info_completa = self.obtener_info_completa_icao(vuelo['icao24'])
                    if info_completa:
                        print(f"  • Modelo: {info_completa.get('modelo', 'N/A')}")
                        print(f"  • Fabricante: {info_completa.get('fabricante', 'N/A')}")
                        print(f"  • Operador: {info_completa.get('operador', 'N/A')}")
        
        elif not resultados and not vuelos:
            print("\n❌ No se encontró información para esta búsqueda")
            print("\n💡 RECOMENDACIONES:")
            print("1. Asegúrate de que la base de datos esté cargada")
            print("2. Verifica el formato de la matrícula")
            print("3. El avión podría no estar en vuelo actualmente")
            print("4. Prueba con el callsign del vuelo (ej: IBE123)")
        
        return {
            'valido': valido,
            'pais': pais,
            'en_base_datos': len(resultados) > 0,
            'en_vuelo': len(vuelos) > 0,
            'resultados_db': resultados,
            'vuelos': vuelos
        }

# FUNCIÓN PRINCIPAL PARA USO FÁCIL
def main():
    """
    Interfaz principal del verificador de matrículas
    """
    print("""
    🛫 VERIFICADOR DE AVIÓN - OPENSKY NETWORK API
    ==============================================
    Usando la biblioteca oficial de OpenSky para Python
    
    Comandos disponibles:
    • [matrícula] - Buscar avión (ej: EC-MYT, N12345, IBE123)
    • cargar_db - Cargar base de datos de aeronaves
    • vuelos - Ver vuelos actuales
    • ayuda - Mostrar esta ayuda
    • salir - Terminar programa
    """)
    
    # Inicializar verificador
    checker = OpenSkyAvionChecker()
    
    while True:
        print("\n" + "─" * 70)
        comando = input("\n🔍 Ingrese comando o matrícula: ").strip()
        
        if comando.lower() == 'salir' or comando.lower() == 'exit':
            print("👋 ¡Hasta luego!")
            break
        
        elif comando.lower() == 'ayuda' or comando.lower() == 'help':
            print("""
            📋 COMANDOS DISPONIBLES:
            
            BÚSQUEDA:
              [matrícula]      - Buscar por matrícula (ej: EC-MYT)
              [callsign]       - Buscar por callsign de vuelo (ej: IBE123)
              [ICAO24]         - Buscar por código ICAO 24-bit (ej: 3c4b26)
            
            BASE DE DATOS:
              cargar_db        - Cargar archivo aircraftDatabase.csv
              buscar_db [txt]  - Buscar texto en base de datos
            
            INFORMACIÓN:
              vuelos           - Mostrar vuelos actuales
              estado           - Estado del sistema
              ayuda            - Mostrar esta ayuda
            
            SISTEMA:
              salir            - Terminar programa
            """)
        
        elif comando.lower() == 'cargar_db':
            ruta = input("Ruta del archivo CSV (Enter para aircraftDatabase.csv): ").strip()
            if not ruta:
                ruta = 'aircraftDatabase.csv'
            checker.cargar_base_datos(ruta)
        
        elif comando.lower() == 'vuelos':
            vuelos = checker.obtener_vuelos_actuales()
            if vuelos:
                print(f"\n✈️  Total de aviones en vuelo: {len(vuelos)}")
                # Mostrar algunos ejemplos
                print("\nEjemplos de vuelos activos:")
                for i, state in enumerate(vuelos[:5]):  # Mostrar solo 5
                    callsign = state.callsign.strip() if state.callsign else "N/A"
                    print(f"  {i+1}. {callsign} ({state.icao24}) - {state.origin_country}")
        
        elif comando.lower() == 'estado':
            print(f"\n📊 ESTADO DEL SISTEMA:")
            print(f"   • Base de datos cargada: {'✅ Sí' if checker.db_loaded else '❌ No'}")
            if checker.db_loaded:
                print(f"   • Registros en BD: {len(checker.aircraft_df)}")
            print(f"   • Autenticado en API: {'✅ Sí' if checker.api._password else '❌ No (límite 400/hora)'}")
        
        elif comando.lower().startswith('buscar_db '):
            busqueda = comando[9:].strip()
            if checker.db_loaded:
                resultados = checker.buscar_en_base_datos(busqueda)
                if not resultados.empty:
                    print(f"\n🔍 Resultados para '{busqueda}':")
                    for idx, row in resultados.head(5).iterrows():
                        print(f"\n  {idx+1}. {row.get('registration', 'N/A')} - {row.get('model', 'N/A')}")
                        print(f"     Operador: {row.get('operator', 'N/A')}")
                        print(f"     ICAO24: {row.get('icao24', 'N/A')}")
            else:
                print("⚠️  Primero carga la base de datos con 'cargar_db'")
        
        elif comando:
            # Búsqueda normal
            resultado = checker.verificar_matricula(comando)
            
            # Ofrecer opciones adicionales si se encontró algo
            if resultado.get('en_vuelo') or resultado.get('en_base_datos'):
                print("\n📌 OPCIONES ADICIONALES:")
                
                if resultado.get('vuelos'):
                    for vuelo in resultado['vuelos']:
                        if vuelo.get('icao24'):
                            print(f"  • Historial de {vuelo['icao24']}: historial {vuelo['icao24']}")
                
                if resultado.get('resultados_db'):
                    for res in resultado['resultados_db']:
                        if res.get('icao24'):
                            print(f"  • Info completa {res['icao24']}: info {res['icao24']}")

# EJECUCIÓN DIRECTA
if __name__ == "__main__":
    # Instrucciones de instalación
    print("📦 INSTALACIÓN REQUERIDA:")
    print("   pip install opensky-api pandas")
    print("\n📥 Descarga la base de datos de aeronaves:")
    print("   https://opensky-network.org/datasets/metadata/aircraftDatabase.csv")
    print("\n🚀 Iniciando verificador...\n")
    
    # Ejecutar interfaz principal
    try:
        checker = OpenSkyAvionChecker(username="gabrielpla-api-client", password="Iko6NSvhl0bZ0xJ3DrLkGjpBc7vqiiyd")
        resultados = checker.buscar_vuelo_por_callsign("BAW2203", True, 720)
        print(resultados)
    except KeyboardInterrupt:
        print("\n\n👋 Programa interrumpido por el usuario")
    except Exception as e:
        print(f"\n❌ Error inesperado: {str(e)}")