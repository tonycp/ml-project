import pandas as pd
import re
import os
import json
from datetime import datetime
from opensky_verificator_api import OpenSkyAvionChecker
from flightaware_scraper import FlightAwareScraper
from paddleocr import PaddleOCR
import cv2
import av


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


def extract_text_from_video_av(video_path, frames_per_second_to_process=1, output_file=None):
    print(f"🎬 Procesando video: {video_path}")

    try:
        container = av.open(video_path)
        stream = container.streams.video[0]
    except Exception as e:
        print(f"Error al abrir el video: {e}")
        return {}

    # FPS es necesario para calcular el número de frame informativo
    fps = float(stream.average_rate) if stream.average_rate else 30.0
    
    video_date, video_time = extract_datetime_from_filename(video_path)
    video_datetime = None
    if video_date and video_time:
        try:
            video_datetime = datetime.strptime(f"{video_date} {video_time}", "%Y-%m-%d %H:%M:%S")
        except: pass

    video_results = {"video_path": video_path, "fps": fps, "extracted_text": []}

    current_sec = 0.0
    intervalo = frames_per_second_to_process
    last_processed_ts = -1.0  # Para detectar si nos quedamos estancados

    while True:
        try:
            # Calculamos el timestamp en las unidades internas del stream
            target_ts = int(current_sec / stream.time_base)
            
            # Intentar el salto
            container.seek(target_ts, stream=stream)
            
            # Leer el siguiente frame tras el salto
            frame = next(container.decode(video=0))
            
            # Obtener el tiempo real del frame obtenido (a veces el seek no es exacto)
            actual_timestamp = float(frame.pts * stream.time_base)

            # --- CONDICIÓN DE SALIDA DE SEGURIDAD ---
            # Si el tiempo actual es igual al anterior, llegamos al final real
            if actual_timestamp <= last_processed_ts and current_sec > 0:
                print("Fin de video detectado (timestamp estancado).")
                break
            
            last_processed_ts = actual_timestamp
            
            # Convertir a RGB para el OCR
            frame_rgb = frame.to_ndarray(format='rgb24')
            result = ocr.predict(frame_rgb)
            
            # Usar la lógica de extracción que ya tenías
            frame_text = result[0]["rec_texts"] if (result and result[0]) else []

            if frame_text:
                # Calcular fechas...
                real_date, real_time, real_iso = video_date, video_time, None
                if video_datetime:
                    real_dt = video_datetime + pd.Timedelta(seconds=actual_timestamp)
                    real_date, real_time, real_iso = real_dt.strftime("%Y-%m-%d"), real_dt.strftime("%H:%M:%S"), real_dt.isoformat()

                video_results["extracted_text"].append({
                    "frame_number": int(actual_timestamp * fps),
                    "timestamp_seconds": round(actual_timestamp, 2),
                    "extracted_text": frame_text,
                    "combined_text": ' '.join(frame_text),
                    "date": real_date, "time": real_time, "timestamp_iso": real_iso
                })
                print(f"Tiempo: {actual_timestamp:.2f}s | Texto: {' '.join(frame_text)}")

            # Incrementamos el tiempo para el siguiente salto
            current_sec += intervalo

        except (StopIteration, av.EOFError):
            # Se alcanzó el final del video
            break
        except Exception as e:
            print(f"Fin de video o error en {current_sec}s: {e}")
            break

    container.close()
    # Guardar en JSON (tu lógica original)
    if output_file:
        with open(output_file, 'a', encoding='utf-8') as f:
            json.dump(video_results, f, ensure_ascii=False, indent=2)

    return video_results


def extract_text_from_video_cv2(video_path, frames_per_second_to_process=1, output_file=None) -> dict[str, any]:
    """
    Procesa video extrayendo texto aproximadamente una vez por segundo.

    Args:
        video_path: ruta del video
        frames_per_second_to_process: número de frames a procesar por cada segundo real del video (por defecto 1)
        output_file: archivo para guardar resultados (formato JSON)
    """

    print(f"🎬 Procesando video: {video_path}")

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

    print(f"FPS del video: {fps:.2f}. Se procesará aproximadamente cada {frame_skip} frames para obtener {frames_per_second_to_process} frame(s) por segundo.")

    # Extraer fecha y hora del nombre del archivo
    video_date, video_time = extract_datetime_from_filename(video_path)
    video_datetime = None
    if video_date and video_time:
        try:
            video_datetime = datetime.strptime(f"{video_date} {video_time}", "%Y-%m-%d %H:%M:%S")
        except Exception as e:
            print(f"Error al analizar la fecha/hora del video: {e}")

    # Estructura para almacenar los resultados en formato JSON
    video_results = {
        "video_path": video_path,
        "fps": fps,
        "extracted_text": []
    }

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Convertir BGR a RGB (PaddleOCR espera RGB)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Procesar con OCR
        result = ocr.predict(frame_rgb)

        # Extraer texto
        frame_text = []

        if result and result[0]:
            frame_text = result[0]["rec_texts"]

        if frame_text:
            timestamp = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000  # segundos
            
            # Calcular fecha y hora reales sumando timestamp a la fecha/hora del video
            real_date = video_date
            real_time = video_time
            real_iso = None
            
            if video_datetime:
                try:
                    real_datetime = video_datetime + pd.Timedelta(seconds=timestamp)
                    real_date = real_datetime.strftime("%Y-%m-%d")
                    real_time = real_datetime.strftime("%H:%M:%S")
                    real_iso = real_datetime.isoformat()
                except Exception as e:
                    print(f"Error calculando fecha/hora real: {e}")
            
            # Crear entrada JSON para este frame
            frame_entry = {
                "frame_number": frame_count,
                "timestamp_seconds": round(timestamp, 2),
                "extracted_text": frame_text,
                "combined_text": ' '.join(frame_text),
                "date": real_date,
                "time": real_time,
                "timestamp_iso": real_iso
            }
            
            video_results["extracted_text"].append(frame_entry)

            print(f"Frame {frame_count} (Timestamp: {timestamp:.2f}s): {' '.join(frame_text)}")

        frame_count += frame_skip
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_count)

    cap.release()
    
    # Guardar resultados en formato JSON
    if output_file:
        with open(output_file, 'a', encoding='utf-8') as f:
            json.dump(video_results, f, ensure_ascii=False, indent=2)
        
    print(f"Procesamiento completado")
    
    return video_results

# Ejecutar si se llama directamente
def process_text_video_with_flightaware(video_results: dict[str, any], output_csv: str = None) -> pd.DataFrame:
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
    vuelos_detectados = []


    # Patrones para buscar códigos de vuelo en el texto OCR
    patrones_vuelo = [
        r'\b[A-Z]{2}\d{2,4}\b',      # Ej: BA223, AA1234
        r'\b[A-Z]{3}\d{1,4}\b',      # Ej: IBE123, KLM45
        #r'\b[A-Z]\d{1,4}[A-Z]?\b',   # Ej: A123, B456C
        r'\b[A-Z]{2,3}\s?\d{2,4}\b', # Ej: AA 123, BA 4567
    ]
    
    # Compilar patrones para mejor rendimiento
    regex_patrones = [re.compile(patron, re.IGNORECASE) for patron in patrones_vuelo]
    
    # 4. Procesar cada frame del OCR
    for frame_entry in video_results.get("extracted_text", []):
        
        # Unir todos los textos del frame
        texto_completo = frame_entry.get("combined_text", "")
        print(f"Texto OCR: {texto_completo}")

        # Buscar posibles códigos de vuelo en el texto
        posibles_codigos = set()
        
        for regex in regex_patrones:
            matches = regex.findall(texto_completo)
            for match in matches:
                # Limpiar y normalizar el código (eliminar espacios y convertir a mayúsculas)
                codigo = re.sub(r'\s+', '', match).upper()
                if 3 <= len(codigo) <= 8:  # Filtrar códigos por longitud razonable
                    posibles_codigos.add(codigo)
      
        if posibles_codigos:
            print(f"Candidatos encontrados: {', '.join(posibles_codigos)}")
            
            for codigo in posibles_codigos:
                if codigo in cache_vuelos:
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
                    # print(f"  🔍 Verificando en FlightAware: {codigo}")
                    resultado = scraper.verificar_vuelo(codigo)
                           
                
                if resultado and resultado.get('existe'):
                    if codigo not in cache_vuelos:
                        cache_vuelos[codigo] = {
                            'aerolinea': resultado.get('aerolinea', 'Desconocida'),
                            'origen': f"{resultado.get('origen', {}).get('codigo', '?')} - {resultado.get('origen', {}).get('ciudad', 'Desconocido')}",
                            'destino': f"{resultado.get('destino', {}).get('codigo', '?')} - {resultado.get('destino', {}).get('ciudad', 'Desconocido')}"
                        }
                    # Calcular fecha y hora reales sumando timestamp a la fecha/hora del video
                    real_date = frame_entry.get("date", "")
                    real_time = frame_entry.get("time", "")
                    # Agregar a la lista de vuelos detectados
                    vuelo_info = {
                        'codigo_vuelo': codigo,
                        'date': real_date,
                        'time': real_time,
                        'aerolinea': resultado.get('aerolinea', 'Desconocida'),
                        'origen': f"{resultado.get('origen', {}).get('codigo', '?')} - {resultado.get('origen', {}).get('ciudad', 'Desconocido')}",
                        'destino': f"{resultado.get('destino', {}).get('codigo', '?')} - {resultado.get('destino', {}).get('ciudad', 'Desconocido')}",
                    }
                    
                    
                    vuelos_detectados.append(vuelo_info)
    
    # 5. Guardar resultados en CSV
    if vuelos_detectados:
        df = pd.DataFrame(vuelos_detectados)

        if output_csv:
            df.to_csv(output_csv, index=False, encoding='utf-8')
            print(f"\n✅ Resultados guardados en: {output_csv}")
        
        print(f"Total de vuelos detectados: {len(vuelos_detectados)}")

        return df
    else:
        print("\n⚠️  No se detectaron vuelos en el video.")
    
    return vuelos_detectados
