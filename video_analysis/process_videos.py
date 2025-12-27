import pandas as pd
import re
import os
from datetime import datetime
from opensky_verificator_api import OpenSkyAvionChecker
from flightaware_scraper import FlightAwareScraper
from paddleocr import PaddleOCR
import cv2

# Inicializar OCR
ocr = PaddleOCR(
    use_doc_orientation_classify=False,
    use_doc_unwarping=False,
    use_textline_orientation=False,
    lang='en')

def extract_datetime_from_filename(video_path):
    """
    Extrae la fecha y hora del nombre del archivo de video.
    
    Formato esperado: 2025-05-12T22_13_13Z_2025-05-12T22_28_13Z.mkv
    Extrae la primera fecha y hora del nombre del archivo.
    
    Args:
        video_path: ruta del archivo de video
        
    Returns:
        tuple: (date_str, time_str) o (None, None) si no se puede extraer
    """
    try:
        # Obtener solo el nombre del archivo
        filename = os.path.basename(video_path)
        
        # Patrón para extraer la primera fecha y hora: YYYY-MM-DDTHH_MM_SSZ
        pattern = r'(\d{4}-\d{2}-\d{2})T(\d{2}):(\d{2}):(\d{2})Z'
        match = re.search(pattern, filename)
        
        if match:
            date_str = match.group(1)  # YYYY-MM-DD
            time_str = f"{match.group(2)}:{match.group(3)}:{match.group(4)}"  # HH:MM:SS
            return date_str, time_str
        
        return None, None
    except Exception as e:
        print(f"Error extrayendo fecha del nombre del archivo: {e}")
        return None, None

def extract_text_from_video(video_path, frames_per_second_to_process=1, output_file='resultados.txt'):
    """
    Procesa video extrayendo texto aproximadamente una vez por segundo.

    Args:
        video_path: ruta del video
        frames_per_second_to_process: número de frames a procesar por cada segundo real del video (por defecto 1)
        output_file: archivo para guardar resultados
    """

    cap = cv2.VideoCapture(video_path)

    # Obtener la cantidad de frames por segundo (FPS) del video
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0:
        print("Error: No se pudo obtener el FPS del video. Asegúrate de que el archivo de video sea válido.")
        cap.release()
        return []

    # Calcular frame_skip para procesar aproximadamente 1 frame por segundo
    # Si frames_per_second_to_process es 1, frame_skip será igual al fps del video
    frame_skip = int(fps * frames_per_second_to_process)
    if frame_skip < 1:
        frame_skip = 1 # Asegurarse de procesar al menos 1 frame

    frame_count = 0
    results = []

    print(f"FPS del video: {fps:.2f}")
    print(f"Se procesará aproximadamente cada {frame_skip} frames para obtener {frames_per_second_to_process} frame(s) por segundo.")

    with open(output_file, 'a', encoding='utf-8') as f:
        f.write(video_path + '\n')
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1

            # Procesar solo cada N frames (para no sobrecargar)
            if frame_count % frame_skip != 0:
                continue

            # Convertir BGR a RGB (PaddleOCR espera RGB)
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Procesar con OCR
            result = ocr.predict(frame_rgb)

            # Extraer texto
            frame_tfext = []

            if result and result[0]:
                frame_tfext = result[0]["rec_texts"]

            if frame_tfext:
                timestamp = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000  # segundos
                text_line = f"[{timestamp:.2f}s] {' '.join(frame_tfext)}\n"
                f.write(text_line)
                results.append((timestamp, frame_tfext))

                print(f"Frame {frame_count} (Timestamp: {timestamp:.2f}s): {text_line.strip()}")

    cap.release()
    print(f"Procesamiento completado. Resultados en {output_file}")
    return results

def process_video_with_opensky(video_path: str, checker: OpenSkyAvionChecker, frames_per_second_to_process=1, output_csv='aviones_detectados.csv'):
    """
    Procesa video con OCR, busca matrículas en base de datos OpenSky y guarda resultados en CSV
    
    Args:
        video_path: ruta del video
        checker: instancia de OpenSkyAvionChecker
        frames_per_second_to_process: frames a procesar por segundo
        output_csv: archivo CSV para guardar resultados
    """
    
    # Extraer fecha y hora del nombre del archivo
    video_date, video_time = extract_datetime_from_filename(video_path)
    print(f"📅 Fecha del video: {video_date}, Hora: {video_time}")
    
    # 1. Procesar video con OCR
    print(f"🎬 Procesando video: {video_path}")
    resultados_ocr = extract_text_from_video(
        video_path, 
        frames_per_second_to_process=frames_per_second_to_process,
        output_file='ocr_resultados.txt'  # Archivo temporal para OCR
    )
    
    print(f"✅ OCR completado. Se procesaron {len(resultados_ocr)} frames")
    
    # 2. Verificar que la base de datos esté cargada
    if not checker.db_loaded:
        print("⚠️  La base de datos no está cargada. Intentando cargar aircraftDatabase.csv...")
        if not checker.cargar_base_datos():
            print("❌ No se pudo cargar la base de datos. Abortando.")
            return []
    
    # 3. Preparar lista para resultados
    aviones_detectados = []
    
    # 4. Patrones para buscar posibles matrículas/callsigns en el texto OCR
    patrones_avion = [
        # Patrones de matrículas internacionales (simplificados)
        r'[A-Z]{1,3}-?[A-Z0-9]{1,6}',  # Formato general matrículas
        r'[A-Z]{3}\d{1,4}',            # Ej: ABC123, IBE456
        r'[A-Z]{2,3}\d{2,4}[A-Z]?',    # Ej: EC123, N1234A
        r'\b[A-Z]{3,5}\b',             # Callsigns de 3-5 letras
        # Patrones específicos comunes
        r'EC-[A-Z]{3,4}',              # España
        r'N\d{1,5}[A-Z]?',             # USA
        r'[A-Z]{2,3}-\d{3,4}[A-Z]?',   # Formato con guión
    ]
    
    # Compilar patrones para mejor rendimiento
    regex_patrones = [re.compile(patron, re.IGNORECASE) for patron in patrones_avion]
    
    # 5. Procesar cada frame del OCR
    for timestamp, frame_texts in resultados_ocr:
        print(f"\n⏱️  Timestamp: {timestamp:.2f}s")
        
        # Unir todos los textos del frame
        texto_completo = ' '.join(frame_texts)
        
        # Buscar posibles matrículas/callsigns en el texto
        posibles_codigos = set()
        
        for regex in regex_patrones:
            matches = regex.findall(texto_completo)
            for match in matches:
                # Limpiar y normalizar el código
                codigo = match.upper().strip()
                if len(codigo) >= 3:  # Filtrar códigos muy cortos
                    posibles_codigos.add(codigo)
        
        print(f"  📝 Texto OCR: {texto_completo[:100]}...")
        
        if posibles_codigos:
            print(f"  🔍 Candidatos encontrados: {', '.join(posibles_codigos)}")
            
            # Buscar cada código en la base de datos
            for codigo in posibles_codigos:
                print(f"  👁️  Buscando: {codigo}")
                
                # Buscar en base de datos
                resultados_db = checker.buscar_en_base_datos(codigo)
                
                if not resultados_db.empty:
                    print(f"  ✅ Encontrado en BD: {len(resultados_db)} resultado(s)")
                    
                    # Para cada avión encontrado, agregar a la lista
                    for idx, avion in resultados_db.iterrows():
                        # Calcular fecha y hora reales sumando timestamp a la fecha/hora del video
                        real_datetime = None
                        real_date = video_date
                        real_time = video_time
                        
                        if video_date and video_time:
                            try:
                                # Convertir fecha y hora del video a datetime
                                video_datetime = datetime.strptime(f"{video_date} {video_time}", "%Y-%m-%d %H:%M:%S")
                                # Sumar el timestamp del frame
                                real_datetime = video_datetime + pd.Timedelta(seconds=timestamp)
                                real_date = real_datetime.strftime("%Y-%m-%d")
                                real_time = real_datetime.strftime("%H:%M:%S")
                            except Exception as e:
                                print(f"Error calculando fecha/hora real: {e}")
                        
                        # Extraer información relevante
                        info_avion = {
                            'codigo': codigo,
                            'icao24': avion.get('icao24', ''),
                            'date': real_date,
                            'time': real_time,
                            'registration': avion.get('registration', ''),
                            'manufacturericao': avion.get('manufacturericao', ''),
                            'manufacturername': avion.get('manufacturername', ''),
                            'model': avion.get('model', ''),
                            'operator': avion.get('operator', ''),
                            'operatorcallsign': avion.get('operatorcallsign', ''),
                            'typecode': avion.get('typecode', ''),
                            'origin_country': avion.get('origin_country', ''),
                        }
                        
                        # Verificar si ya detectamos este avión en timestamp similar
                        # (para evitar duplicados del mismo frame)
                        es_duplicado = False
                        for avion_existente in aviones_detectados:
                            if (avion_existente.get('icao24') == info_avion['icao24'] and 
                                avion_existente.get('date') == info_avion.get('date') and
                                avion_existente.get('time') == info_avion.get('time')):
                                es_duplicado = True
                                break
                        
                        if not es_duplicado:
                            aviones_detectados.append(info_avion)
                            print(f"    ✈️  Añadido: {info_avion['registration']} ({info_avion['model']})")
    
    # 6. Guardar resultados en CSV
    if aviones_detectados:
        print(f"\n💾 Guardando {len(aviones_detectados)} aviones detectados en {output_csv}")
        
        # Crear DataFrame
        df_resultados = pd.DataFrame(aviones_detectados)
        
        # Ordenar por timestamp
        df_resultados = df_resultados.sort_values('timestamp')
        
        # Guardar en CSV
        df_resultados.to_csv(output_csv, index=False, encoding='utf-8')
        
        # Mostrar resumen
        print("\n📊 RESUMEN DE DETECCIONES:")
        print(f"   • Total aviones detectados: {len(aviones_detectados)}")
        print(f"   • Archivo CSV guardado: {output_csv}")
        
        # Mostrar algunos ejemplos
        if len(aviones_detectados) > 0:
            print("\n📝 EJEMPLOS DETECTADOS:")
            for i, avion in enumerate(aviones_detectados[:5]):  # Mostrar primeros 5
                print(f"  {i+1}. {avion['timestamp_formateado']} - {avion['registration']} ({avion['model']}) - {avion['operator']}")
    
    else:
        print("\n❌ No se detectaron aviones en el video")
    
    return aviones_detectados

# Ejecutar si se llama directamente
def process_video_with_flightaware(video_path: str, frames_per_second_to_process=1, output_csv=None) -> pd.DataFrame:
    """
    Procesa video con OCR, busca códigos de vuelo en FlightAware y guarda resultados en CSV.
    Evita scrapeos innecesarios verificando códigos previamente procesados.
    
    Args:
        video_path: ruta del video
        frames_per_second_to_process: frames a procesar por segundo
        output_csv: archivo CSV para guardar resultados
    """
    
    # Inicializar el scraper de FlightAware
    scraper = FlightAwareScraper()
    
    # Diccionario para almacenar códigos ya procesados y sus resultados
    cache_vuelos = {}
    
    # Cargar datos existentes si el archivo de salida ya existe
    vuelos_detectados = []
    if output_csv and os.path.exists(output_csv):
        try:
            df_existente = pd.read_csv(output_csv)
            vuelos_detectados = df_existente.to_dict('records')
            # Crear caché de códigos ya procesados
            for vuelo in vuelos_detectados:
                codigo = vuelo['codigo_vuelo']
                if codigo not in cache_vuelos:
                    cache_vuelos[codigo] = {
                        'aerolinea': vuelo.get('aerolinea', 'Desconocida'),
                        'origen': vuelo.get('origen', ''),
                        'destino': vuelo.get('destino', '')
                    }
            print(f"📋 Cargados {len(cache_vuelos)} códigos de vuelo previamente procesados")
        except Exception as e:
            print(f"⚠️  No se pudo cargar el archivo existente: {e}")
    
    # Extraer fecha y hora del nombre del archivo
    video_date, video_time = extract_datetime_from_filename(video_path)
    
    # 1. Procesar video con OCR
    print(f"🎬 Procesando video: {video_path}")
    resultados_ocr = extract_text_from_video(
        video_path, 
        frames_per_second_to_process=frames_per_second_to_process,
        output_file='ocr_resultados.txt'  # Archivo temporal para OCR
    )
    
    # print(f"✅ OCR completado. Se procesaron {len(resultados_ocr)} frames")
    
    # 2. Preparar lista para resultados
    vuelos_detectados = []
    
    # 3. Patrones para buscar códigos de vuelo en el texto OCR
    patrones_vuelo = [
        r'\b[A-Z]{2}\d{2,4}\b',      # Ej: BA223, AA1234
        r'\b[A-Z]{3}\d{1,4}\b',      # Ej: IBE123, KLM45
        #r'\b[A-Z]\d{1,4}[A-Z]?\b',   # Ej: A123, B456C
        r'\b[A-Z]{2,3}\s?\d{2,4}\b', # Ej: AA 123, BA 4567
    ]
    
    # Compilar patrones para mejor rendimiento
    regex_patrones = [re.compile(patron, re.IGNORECASE) for patron in patrones_vuelo]
    
    # 4. Procesar cada frame del OCR
    for timestamp, frame_texts in resultados_ocr:
        # print(f"\n⏱️  Timestamp: {timestamp:.2f}s")
        
        # Unir todos los textos del frame
        texto_completo = ' '.join(frame_texts)
        
        # Buscar posibles códigos de vuelo en el texto
        posibles_codigos = set()
        
        for regex in regex_patrones:
            matches = regex.findall(texto_completo)
            for match in matches:
                # Limpiar y normalizar el código (eliminar espacios y convertir a mayúsculas)
                codigo = re.sub(r'\s+', '', match).upper()
                if 3 <= len(codigo) <= 8:  # Filtrar códigos por longitud razonable
                    posibles_codigos.add(codigo)
        
        # print(f"  📝 Texto OCR: {texto_completo[:100]}...")
        
        if posibles_codigos:
            print(f"  🔍 Candidatos encontrados: {', '.join(posibles_codigos)}")
            
            # Verificar cada código en FlightAware
            for codigo in posibles_codigos:
                # print(f"  ✈️  Verificando vuelo: {codigo}")
                
                

                # Verificar si ya tenemos este código en caché
                if codigo in cache_vuelos:
                    print(f"  🔄 Código {codigo} ya procesado anteriormente")
                    resultado = {
                        'existe': True,
                        'aerolinea': cache_vuelos[codigo]['aerolinea'],
                        'origen': {'codigo': cache_vuelos[codigo]['origen'].split(' - ')[0] if ' - ' in cache_vuelos[codigo]['origen'] else '?',
                                 'ciudad': cache_vuelos[codigo]['origen'].split(' - ')[1] if ' - ' in cache_vuelos[codigo]['origen'] else 'Desconocido'},
                        'destino': {'codigo': cache_vuelos[codigo]['destino'].split(' - ')[0] if ' - ' in cache_vuelos[codigo]['destino'] else '?',
                                  'ciudad': cache_vuelos[codigo]['destino'].split(' - ')[1] if ' - ' in cache_vuelos[codigo]['destino'] else 'Desconocido'}
                    }
                else:
                    # Si no está en caché, hacer scraping
                    print(f"  🔍 Verificando en FlightAware: {codigo}")
                    resultado = scraper.verificar_vuelo(codigo)
                
                if resultado and resultado.get('existe'):
                    # Actualizar caché si es un código nuevo
                    if codigo not in cache_vuelos:
                        cache_vuelos[codigo] = {
                            'aerolinea': resultado.get('aerolinea', 'Desconocida'),
                            'origen': f"{resultado.get('origen', {}).get('codigo', '?')} - {resultado.get('origen', {}).get('ciudad', 'Desconocido')}",
                            'destino': f"{resultado.get('destino', {}).get('codigo', '?')} - {resultado.get('destino', {}).get('ciudad', 'Desconocido')}"
                        }
                    
                    # Calcular fecha y hora reales sumando timestamp a la fecha/hora del video
                    real_datetime = None
                    real_date = video_date
                    real_time = video_time
                    
                    if video_date and video_time:
                        try:
                            # Convertir fecha y hora del video a datetime
                            video_datetime = datetime.strptime(f"{video_date} {video_time}", "%Y-%m-%d %H:%M:%S")
                            # Sumar el timestamp del frame
                            real_datetime = video_datetime + pd.Timedelta(seconds=timestamp)
                            real_date = real_datetime.strftime("%Y-%m-%d")
                            real_time = real_datetime.strftime("%H:%M:%S")
                        except Exception as e:
                            print(f"Error calculando fecha/hora real: {e}")


                    # Agregar a la lista de vuelos detectados
                    vuelo_info = {
                        'codigo_vuelo': codigo,
                        'date': real_date,
                        'time': real_time,
                        'aerolinea': resultado.get('aerolinea', 'Desconocida'),
                        'origen': f"{resultado.get('origen', {}).get('codigo', '?')} - {resultado.get('origen', {}).get('ciudad', 'Desconocido')}",
                        'destino': f"{resultado.get('destino', {}).get('codigo', '?')} - {resultado.get('destino', {}).get('ciudad', 'Desconocido')}",
                    }
                    
                    # Añadir horarios de takeoff y landing si están disponibles
                    if 'horarios' in resultado:
                        horarios = resultado['horarios']
                        
                        # Horarios de takeoff
                        if 'programado' in horarios:
                            vuelo_info['takeoff_programado'] = horarios['programado']
                        if 'estimado' in horarios:
                            vuelo_info['takeoff_estimado'] = horarios['estimado']
                        if 'actual' in horarios:
                            vuelo_info['takeoff_actual'] = horarios['actual']
                        
                        # Horarios de aterrizaje (landing)
                        if 'aterrizaje' in horarios:
                            aterrizaje = horarios['aterrizaje']
                            if 'programado' in aterrizaje:
                                vuelo_info['landing_programado'] = aterrizaje['programado']
                            if 'estimado' in aterrizaje:
                                vuelo_info['landing_estimado'] = aterrizaje['estimado']
                            if 'actual' in aterrizaje:
                                vuelo_info['landing_actual'] = aterrizaje['actual']
                    
                    vuelos_detectados.append(vuelo_info)
    
    # 5. Guardar resultados en CSV
    if vuelos_detectados:
        df = pd.DataFrame(vuelos_detectados)

        if output_csv:
            df.to_csv(output_csv, index=False, encoding='utf-8')
            print(f"\n✅ Resultados guardados en: {output_csv}")
            print(f"📊 Total de vuelos detectados: {len(vuelos_detectados)}")

        return df
    else:
        print("\n⚠️  No se detectaron vuelos en el video.")
    
    return vuelos_detectados


# Modificación en la función main original para incluir procesamiento de video
def main_opensky():
    """
    Menú principal completo con opción de procesamiento de video
    """
    print("""
    🛫 VERIFICADOR DE AVIÓN - OPENSKY NETWORK API
    ==============================================
    Usando la biblioteca oficial de OpenSky para Python
    
    Comandos disponibles:
    • [matrícula] - Buscar avión (ej: EC-MYT, N12345, IBE123)
    • cargar_db - Cargar base de datos de aeronaves
    • procesar_video - Procesar video para detectar aviones
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
            
            PROCESAMIENTO VIDEO:
              procesar_video   - Procesar video para detectar aviones con OCR
            
            BASE DE DATOS:
              cargar_db        - Cargar archivo aircraftDatabase.csv
              buscar_db [txt]  - Buscar texto en base de datos
            
            INFORMACIÓN:
              vuelos     video_path = "/home/gabo/Personal/Work/Lita/Proyectos/VideoSearch/vid-chronicle-main/data/videos/TWR-HAV/2025/05/12/+0,0/2025-05-12T22:13:13Z_2025-05-12T22:28:13Z.mkv"
      - Mostrar vuelos actuales
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
        
        elif comando.lower() == 'procesar_video':
            video_path = input("Ruta del video a procesar: ").strip()
            if not video_path:
                print("❌ Debes especificar una ruta de video")
                continue
            
            if not checker.db_loaded:
                print("⚠️  Cargando base de datos primero...")
                checker.cargar_base_datos()
            
            fps = input("Frames por segundo a procesar (1 por defecto): ").strip()
            fps_num = 1
            if fps:
                try:
                    fps_num = float(fps)
                except:
                    print("⚠️  Valor inválido, usando 1 fps")
            
            process_video_with_opensky(video_path, checker, fps_num)
        
        elif comando.lower() == 'vuelos':
            # ... (código existente para mostrar vuelos)
            vuelos = checker.obtener_vuelos_actuales()
            if vuelos:
                print(f"\n✈️  Total de aviones en vuelo: {len(vuelos)}")
                for i, state in enumerate(vuelos[:5]):
                    callsign = state.callsign.strip() if state.callsign else "N/A"
                    print(f"  {i+1}. {callsign} ({state.icao24}) - {state.origin_country}")
        
        else:
            # Búsqueda normal
            resultado = checker.verificar_matricula(comando)

def main_flightaware():
    """
    Menú principal para procesar videos con FlightAware
    """
    print("""
    ✈️  PROCESADOR DE VIDEOS CON FLIGHTAWARE
    ======================================
    Esta herramienta procesa videos, extrae texto con OCR
    y verifica códigos de vuelo en FlightAware.
    """)
    
    while True:
        print("\n" + "="*50)
        print("  MENÚ PRINCIPAL - FLIGHTAWARE")
        print("="*50)
        print("1. Procesar video")
        print("2. Salir")
        
        opcion = input("\nSeleccione una opción: ").strip()
        
        if opcion == '1':
            video_path = input("\nIngrese la ruta del video: ").strip('"')
            if not os.path.exists(video_path):
                print("❌ El archivo no existe. Intente nuevamente.")
                continue
                
            try:
                fps = float(input("Frames por segundo a procesar (recomendado: 1): ") or "1")
                process_video_with_flightaware(video_path, frames_per_second_to_process=fps)
            except Exception as e:
                print(f"❌ Error al procesar el video: {str(e)}")
                
        elif opcion == '2':
            print("\n¡Hasta luego! ✈️")
            break
            
        else:
            print("❌ Opción no válida. Intente nuevamente.")

if __name__ == "__main__":
    # Ejemplo de uso:
    # videos = find_video_files('/ruta/a/tu/directorio')
    # for video in videos:
    #     print(video)
    
    # Puedes elegir qué función ejecutar:
    # main_completo()  # Para el menú interactivo completo
    # main_video_processing()  # Para procesamiento directo de video
    
    video_path = "/home/gabo/Personal/Work/Lita/Proyectos/VideoSearch/vid-chronicle-main/data/videos/TWR-HAV/2025/05/12/+0,0/2025-05-12T22:13:13Z_2025-05-12T22:28:13Z.mkv"
    process_video_with_flightaware(video_path, frames_per_second_to_process=300)