import pandas as pd
import re
from datetime import datetime
from opensky_verificator_api import OpenSkyAvionChecker
from paddleocr import PaddleOCR
import cv2

# Inicializar OCR
ocr = PaddleOCR(
    use_doc_orientation_classify=False,
    use_doc_unwarping=False,
    use_textline_orientation=False,
    lang='en')

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

    with open(output_file, 'w', encoding='utf-8') as f:
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
                        # Extraer información relevante
                        info_avion = {
                            'timestamp': timestamp,
                            'timestamp_formateado': f"{int(timestamp//3600):02d}:{int((timestamp%3600)//60):02d}:{timestamp%60:06.3f}",
                            'codigo_buscado': codigo,
                            'icao24': avion.get('icao24', ''),
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
                            if (avion_existente['icao24'] == info_avion['icao24'] and 
                                abs(avion_existente['timestamp'] - timestamp) < 1.0):
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

# Función mejorada para buscar todos los posibles substrings
def buscar_todos_substrings(checker, texto, min_length=3, max_length=10):
    """
    Busca todos los posibles substrings en la base de datos
    
    Args:
        checker: instancia de OpenSkyAvionChecker
        texto: texto completo del OCR
        min_length: longitud mínima del substring
        max_length: longitud máxima del substring
    """
    resultados = []
    texto_upper = texto.upper()
    
    # Extraer todas las posibles combinaciones de letras/números
    palabras = re.findall(r'[A-Z0-9]{3,10}', texto_upper)
    
    for palabra in palabras:
        # Buscar substrings de diferentes longitudes
        for length in range(min_length, min(max_length, len(palabra)) + 1):
            for start in range(0, len(palabra) - length + 1):
                substring = palabra[start:start+length]
                
                # Buscar en base de datos
                resultados_db = checker.buscar_en_base_datos(substring)
                
                if not resultados_db.empty:
                    for _, avion in resultados_db.iterrows():
                        resultado = {
                            'substring': substring,
                            'icao24': avion.get('icao24', ''),
                            'registration': avion.get('registration', ''),
                            'model': avion.get('model', ''),
                            'operator': avion.get('operator', '')
                        }
                        resultados.append(resultado)
    
    return resultados

# Versión alternativa que usa la función de búsqueda por substrings
def process_video_with_opensky_substrings(video_path, checker, frames_per_second_to_process=1, output_csv='aviones_detectados.csv'):
    """
    Versión que busca todos los posibles substrings en el texto OCR
    """
    
    print(f"🎬 Procesando video: {video_path}")
    resultados_ocr = extract_text_from_video(
        video_path, 
        frames_per_second_to_process=frames_per_second_to_process,
        output_file='ocr_resultados.txt'
    )
    
    if not checker.db_loaded:
        print("⚠️  Cargando base de datos...")
        checker.cargar_base_datos()
    
    aviones_detectados = []
    
    for timestamp, frame_texts in resultados_ocr:
        texto_completo = ' '.join(frame_texts)
        
        print(f"\n⏱️  Timestamp: {timestamp:.2f}s")
        print(f"  📝 Texto: {texto_completo[:100]}...")
        
        # Buscar todos los substrings posibles
        resultados_substrings = buscar_todos_substrings(checker, texto_completo)
        
        for resultado in resultados_substrings:
            info_avion = {
                'timestamp': timestamp,
                'timestamp_formateado': f"{int(timestamp//3600):02d}:{int((timestamp%3600)//60):02d}:{timestamp%60:06.3f}",
                'substring_encontrado': resultado['substring'],
                'icao24': resultado['icao24'],
                'registration': resultado['registration'],
                'model': resultado['model'],
                'operator': resultado['operator'],
                'texto_ocr': texto_completo[:500]
            }
            
            # Evitar duplicados
            if not any(a['icao24'] == info_avion['icao24'] and 
                      abs(a['timestamp'] - timestamp) < 1.0 
                      for a in aviones_detectados):
                aviones_detectados.append(info_avion)
                print(f"  ✅ Detectado: {resultado['registration']} via '{resultado['substring']}'")
    
    # Guardar en CSV
    if aviones_detectados:
        df = pd.DataFrame(aviones_detectados)
        df.to_csv(output_csv, index=False, encoding='utf-8')
        print(f"\n💾 Guardados {len(aviones_detectados)} registros en {output_csv}")
    
    return aviones_detectados

# Función principal de ejemplo
def main_video_processing():
    """
    Ejemplo de uso de procesamiento de video con OpenSky
    """
    print("🎥 PROCESADOR DE VIDEO CON DETECCIÓN DE AVIÓN")
    print("=" * 50)
    
    # Inicializar verificador
    checker = OpenSkyAvionChecker()
    
    # Cargar base de datos
    print("📂 Cargando base de datos OpenSky...")
    if not checker.cargar_base_datos('aircraftDatabase.csv'):
        print("❌ No se pudo cargar la base de datos. Asegúrate de tener el archivo.")
        return
    
    # Solicitar ruta del video
    video_path = input("\n📁 Ruta del video a procesar: ").strip()
    if not video_path:
        video_path = "video_ejemplo.mp4"  # Ruta por defecto
    
    # Procesar video
    resultados = process_video_with_opensky(
        video_path=video_path,
        checker=checker,
        frames_per_second_to_process=1,
        output_csv='aviones_detectados.csv'
    )
    
    # Mostrar resumen final
    print("\n" + "=" * 50)
    print("🎬 PROCESAMIENTO COMPLETADO")
    if resultados:
        print(f"✅ Se detectaron {len(resultados)} aviones")
        print(f"📄 Resultados guardados en: aviones_detectados.csv")
        
        # Opción: Mostrar en vuelo actual
        respuesta = input("\n¿Buscar aviones detectados en vuelo actual? (s/n): ").lower()
        if respuesta == 's':
            for avion in resultados:
                if avion['icao24']:
                    print(f"\n🔍 Buscando {avion['registration']} ({avion['icao24']})...")
                    vuelos = checker.buscar_avion_en_vuelo(avion['icao24'])
                    if vuelos:
                        print(f"  ✈️  EN VUELO AHORA!")
                        for vuelo in vuelos:
                            print(f"  • Callsign: {vuelo['callsign']}")
                            if vuelo['posicion']:
                                lat, lon = vuelo['posicion']
                                print(f"  • Posición: {lat:.4f}°, {lon:.4f}°")
    else:
        print("❌ No se detectaron aviones en el video")

# Modificación en la función main original para incluir procesamiento de video
def main_completo():
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

# Ejecutar si se llama directamente
if __name__ == "__main__":
    # Puedes elegir qué función ejecutar:
    # main_completo()  # Para el menú interactivo completo
    # main_video_processing()  # Para procesamiento directo de video
    
    print("🎬 Ejecutar main_completo() para menú interactivo con video")
    print("🎥 Ejecutar main_video_processing() para procesar video directamente")