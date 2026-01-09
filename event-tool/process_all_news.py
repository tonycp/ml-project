#!/usr/bin/env python3
"""
Script para procesar todas las noticias de la base de datos y extraer eventos.

Uso:
    # Como script
    python process_all_news.py
    
    # Como módulo
    from process_all_news import extract_events_from_all_news
    events = extract_events_from_all_news()
"""

import sqlite3
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Dict, Any
import json
from tqdm import tqdm

from Event_extractor import EventExtractionPipeline, NewsContent, Event
from Event_extractor.classifiers.sentiment import (
    KeywordSentimentClassifier,
    SklearnSentimentClassifier,
    SentimentClassifier
)
from Event_extractor.classifiers.news_type import SklearnNewsClassifier


def load_news_from_database(
    database_path: str = "noticias.db",
    limit: Optional[int] = None,
    verbose: bool = True
) -> List[NewsContent]:
    """
    Carga noticias desde la base de datos SQLite y las convierte a objetos NewsContent.
    
    Args:
        database_path: Ruta a la base de datos SQLite
        limit: Número máximo de noticias a cargar (None = todas)
        verbose: Mostrar información de progreso
    
    Returns:
        List[NewsContent]: Lista de noticias cargadas
    
    Raises:
        FileNotFoundError: Si no existe la base de datos
        ValueError: Si la base de datos no contiene tablas o noticias
    
    Ejemplo:
        >>> news_list = load_news_from_database("noticias.db")
        >>> print(f"Cargadas {len(news_list)} noticias")
        
        >>> # Cargar solo las primeras 50
        >>> news_list = load_news_from_database(limit=50)
    """
    # Verificar que existe la base de datos
    db_path = Path(database_path)
    if not db_path.exists():
        raise FileNotFoundError(
            f"No se encuentra la base de datos: {database_path}\n"
            f"Asegúrate de que el archivo existe en el directorio actual."
        )
    
    if verbose:
        print(f"📂 Cargando noticias desde: {database_path}")
    
    # Conectar a la base de datos
    conn = sqlite3.connect(database_path)
    cursor = conn.cursor()
    
    # Inspeccionar estructura de la tabla
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
    tables = cursor.fetchall()
    
    if not tables:
        conn.close()
        raise ValueError("La base de datos no contiene ninguna tabla")
    
    table_name = tables[0][0]
    
    if verbose:
        print(f"📊 Tabla: {table_name}")
    
    # Cargar noticias
    query = f"SELECT * FROM {table_name}"
    if limit:
        query += f" LIMIT {limit}"
    
    cursor.execute(query)
    columns = [description[0] for description in cursor.description]
    rows = cursor.fetchall()
    
    conn.close()
    
    if not rows:
        if verbose:
            print("⚠️  No hay noticias en la base de datos")
        return []
    
    if verbose:
        print(f"✅ Cargadas {len(rows)} noticias")
        print(f"📋 Columnas: {', '.join(columns)}")
    
    # Convertir filas a objetos NewsContent
    news_list = []
    
    for i, row in enumerate(rows, 1):
        # Crear diccionario con los datos de la fila
        row_dict = dict(zip(columns, row))
        
        # Extraer campos relevantes (adaptable a diferentes esquemas)
        news_id = row_dict.get('id') or row_dict.get('msg_id') or str(i)
        text = row_dict.get('texto') or row_dict.get('text') or row_dict.get('content')
        date_str = row_dict.get('fecha') or row_dict.get('date')
        
        if not text:
            if verbose:
                print(f"⚠️  Noticia {i}: Sin texto, omitida")
            continue
        
        # Parsear fecha
        news_date = None
        if date_str:
            try:
                # Intentar varios formatos
                for fmt in ["%Y-%m-%d %H:%M:%S", "%Y-%m-%d", "%d/%m/%Y"]:
                    try:
                        news_date = datetime.strptime(str(date_str), fmt)
                        break
                    except ValueError:
                        continue
            except Exception:
                pass
        
        # Crear NewsContent
        news = NewsContent(
            id=str(news_id),
            text=text,
            publication_date=news_date
        )
        
        news_list.append(news)
    
    if verbose:
        print(f"✓ {len(news_list)} noticias válidas cargadas\n")
    
    return news_list


def extract_events_from_all_news(
    database_path: str = "noticias.db",
    use_sklearn_type_classifier: Optional[bool] = None,
    sklearn_model_path: Optional[str] = None,
    sentiment_classifier: Optional[str] = None,
    limit: Optional[int] = None,
    save_to_json: Optional[str] = None,
    save_to_csv: Optional[str] = None,
    verbose: bool = True,
    debug: bool = False
) -> List[Event]:
    """
    Procesa todas las noticias de la base de datos y extrae eventos.
    
    Args:
        database_path: Ruta a la base de datos SQLite (default: "noticias.db")
        use_sklearn_type_classifier: Si True, fuerza sklearn; si False, fuerza keyword; 
                                     si None (default), intenta usar sklearn si está disponible
        sklearn_model_path: Ruta al modelo sklearn entrenado (opcional)
        sentiment_classifier: Clasificador de sentimiento: None (auto-detecta), "keyword", "sklearn", 
                            o instancia de SentimentClassifier
        limit: Número máximo de noticias a procesar (None = todas)
        save_to_json: Ruta para guardar eventos en JSON (opcional)
        save_to_csv: Ruta para guardar eventos en CSV (opcional)
        verbose: Mostrar información de progreso
    
    Returns:
        List[Event]: Lista de todos los eventos extraídos
    
    Ejemplos:
        >>> # Uso básico (auto-detecta clasificadores disponibles)
        >>> events = extract_events_from_all_news()
        
        >>> # Forzar clasificadores específicos
        >>> events = extract_events_from_all_news(
        ...     use_sklearn_type_classifier=True,
        ...     sentiment_classifier="sklearn"
        ... )
        
        >>> # Guardar resultados
        >>> events = extract_events_from_all_news(
        ...     save_to_json="eventos.json",
        ...     save_to_csv="eventos.csv"
        ... )
    """
    
    # Cargar noticias desde la base de datos
    news_list = load_news_from_database(
        database_path=database_path,
        limit=limit,
        verbose=verbose
    )
    
    if not news_list:
        return []
    
    # Crear pipeline
    if verbose:
        print(f"\n🔧 Configurando pipeline...")
    
    # ========================================================================
    # Configurar clasificador de TIPO
    # ========================================================================
    actual_use_sklearn = False
    actual_sklearn_model_path = None
    
    # Buscar modelo por defecto
    default_type_model = Path("models/sklearn_spanish_svm.pkl")
    
    if use_sklearn_type_classifier is None:
        # Auto-detectar: usar sklearn si está disponible
        if sklearn_model_path:
            # Si se especificó una ruta, verificar si existe
            if Path(sklearn_model_path).exists():
                actual_use_sklearn = True
                actual_sklearn_model_path = sklearn_model_path
                if verbose:
                    print(f"   ✅ Clasificador de tipo: sklearn (modelo: {sklearn_model_path})")
            else:
                if verbose:
                    print(f"   ⚠️  Modelo sklearn no encontrado: {sklearn_model_path}")
                    print("   📝 Usando clasificador keyword para tipo de evento")
        elif default_type_model.exists():
            # Usar modelo por defecto si existe
            actual_use_sklearn = True
            actual_sklearn_model_path = str(default_type_model)
            if verbose:
                print(f"   ✅ Clasificador de tipo: sklearn (modelo: {actual_sklearn_model_path})")
        else:
            # No hay modelo disponible, usar keyword
            if verbose:
                print(f"   📝 Clasificador de tipo: keyword (sklearn no disponible)")
    
    elif use_sklearn_type_classifier:
        # Forzar uso de sklearn
        if sklearn_model_path:
            if Path(sklearn_model_path).exists():
                actual_use_sklearn = True
                actual_sklearn_model_path = sklearn_model_path
                if verbose:
                    print(f"   ✅ Clasificador de tipo: sklearn (modelo: {sklearn_model_path})")
            else:
                if verbose:
                    print(f"   ❌ Error: Modelo sklearn no encontrado: {sklearn_model_path}")
                    print("   📝 Usando clasificador keyword como fallback")
        elif default_type_model.exists():
            actual_use_sklearn = True
            actual_sklearn_model_path = str(default_type_model)
            if verbose:
                print(f"   ✅ Clasificador de tipo: sklearn (modelo: {actual_sklearn_model_path})")
        else:
            if verbose:
                print(f"   ⚠️  Modelo sklearn de tipo no encontrado en {default_type_model}")
                print("   📝 Usando clasificador keyword como fallback")
    else:
        # Forzar uso de keyword (use_sklearn_type_classifier=False)
        if verbose:
            print("   📝 Clasificador de tipo: keyword (forzado)")
    
    # ========================================================================
    # Configurar clasificador de SENTIMIENTO
    # ========================================================================
    sentiment_clf = None
    default_sentiment_model = Path("models/sklearn_tass_sentiment.pkl")
    
    if sentiment_classifier is None:
        # Auto-detectar: usar sklearn si está disponible
        if default_sentiment_model.exists():
            try:
                sentiment_clf = SklearnSentimentClassifier.load_model(str(default_sentiment_model))
                if verbose:
                    print(f"   ✅ Clasificador de sentimiento: sklearn (modelo: {default_sentiment_model})")
            except Exception as e:
                if verbose:
                    print(f"   ⚠️  Error cargando sklearn de sentimiento: {e}")
                    print("   📝 Usando clasificador keyword para sentimiento")
                sentiment_clf = KeywordSentimentClassifier()
        else:
            sentiment_clf = KeywordSentimentClassifier()
            if verbose:
                print(f"   📝 Clasificador de sentimiento: keyword (sklearn no disponible)")
    
    elif sentiment_classifier == "sklearn":
        # Forzar uso de sklearn
        if default_sentiment_model.exists():
            try:
                sentiment_clf = SklearnSentimentClassifier.load_model(str(default_sentiment_model))
                if verbose:
                    print(f"   ✅ Clasificador de sentimiento: sklearn (modelo: {default_sentiment_model})")
            except Exception as e:
                if verbose:
                    print(f"   ⚠️  Error cargando sklearn de sentimiento: {e}")
                    print("   📝 Usando clasificador keyword como fallback")
                sentiment_clf = KeywordSentimentClassifier()
        else:
            if verbose:
                print(f"   ⚠️  Modelo sklearn de sentimiento no encontrado en {default_sentiment_model}")
                print("   📝 Usando clasificador keyword como fallback")
            sentiment_clf = KeywordSentimentClassifier()
    
    elif sentiment_classifier == "keyword":
        # Forzar uso de keyword
        sentiment_clf = KeywordSentimentClassifier()
        if verbose:
            print("   📝 Clasificador de sentimiento: keyword (forzado)")
    
    elif isinstance(sentiment_classifier, SentimentClassifier):
        sentiment_clf = sentiment_classifier
        if verbose:
            print("   ✅ Clasificador de sentimiento: personalizado")
    
    else:
        sentiment_clf = KeywordSentimentClassifier()
        if verbose:
            print("   📝 Clasificador de sentimiento: keyword (fallback)")
    
    # Crear el pipeline con la configuración correcta
    pipeline = EventExtractionPipeline(
        use_sklearn_classifier=actual_use_sklearn,
        sklearn_model_path=actual_sklearn_model_path,
        sentiment_classifier=sentiment_clf
    )
    
    # Procesar noticias
    if verbose:
        print(f"\n⚙️  Procesando noticias...\n")
    
    all_events = []
    news_with_events = 0
    errors = 0
    error_messages = []  # Guardar primeros errores para mostrar
    
    # Usar tqdm para barra de progreso
    progress_bar = tqdm(
        enumerate(news_list, 1),
        total=len(news_list),
        desc="Procesando",
        unit="noticia",
        disable=not verbose
    )
    
    for i, news in progress_bar:
        # Extraer eventos
        try:
            events = pipeline.extract_events(news)
            
            if events:
                all_events.extend(events)
                news_with_events += 1
                progress_bar.set_postfix({
                    'eventos': len(all_events),
                    'con_eventos': news_with_events,
                    'errores': errors
                })
        
        except Exception as e:
            errors += 1
            
            # Guardar primeros 5 errores para debug
            if len(error_messages) < 5:
                error_messages.append(f"Noticia {i}: {type(e).__name__}: {str(e)}")
            
            # Mostrar error en modo debug
            if debug:
                print(f"\n❌ Error en noticia {i}: {e}")
                import traceback
                traceback.print_exc()
            
            progress_bar.set_postfix({
                'eventos': len(all_events),
                'con_eventos': news_with_events,
                'errores': errors
            })
            continue
            continue
    
    # Resumen
    if verbose:
        print(f"\n" + "="*60)
        print(f"📊 RESUMEN")
        print(f"="*60)
        print(f"Noticias procesadas: {len(news_list)}")
        print(f"Noticias con eventos: {news_with_events}")
        print(f"Errores encontrados: {errors}")
        print(f"Total de eventos extraídos: {len(all_events)}")
        if len(news_list) > 0:
            print(f"Promedio de eventos por noticia: {len(all_events)/len(news_list):.2f}")
        
        # Mostrar primeros errores si hubo
        if error_messages:
            print(f"\n⚠️  Primeros errores encontrados:")
            for msg in error_messages:
                print(f"   {msg}")
    
    # Guardar resultados si se especificó
    if save_to_json:
        _save_events_to_json(all_events, save_to_json, verbose)
    
    if save_to_csv:
        _save_events_to_csv(all_events, save_to_csv, verbose)
    
    return all_events


def _save_events_to_json(events: List[Event], filepath: str, verbose: bool = True):
    """Guarda los eventos en formato JSON."""
    
    eventos_json = []
    for event in events:
        evento_dict = {
            "fecha": event.date.isoformat(),
            "tipo": event.event_type.value,
            "sentimiento": event.sentiment.value,
            "confianza_tipo": event.confidence,
            "confianza_sentimiento": event.sentiment_confidence,
            "noticia_id": event.source_news_id,
            "entidades": event.entidades_asociadas or []
        }
        eventos_json.append(evento_dict)
    
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(eventos_json, f, indent=2, ensure_ascii=False)
    
    if verbose:
        print(f"\n💾 Eventos guardados en JSON: {filepath}")


def _save_events_to_csv(events: List[Event], filepath: str, verbose: bool = True):
    """Guarda los eventos en formato CSV."""
    import csv
    
    with open(filepath, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        
        # Header
        writer.writerow([
            "fecha",
            "tipo",
            "sentimiento",
            "confianza_tipo",
            "confianza_sentimiento",
            "noticia_id",
            "num_entidades"
        ])
        
        # Datos
        for event in events:
            writer.writerow([
                event.date.strftime("%Y-%m-%d"),
                event.event_type.value,
                event.sentiment.value,
                f"{event.confidence:.4f}",
                f"{event.sentiment_confidence:.4f}",
                event.source_news_id,
                len(event.entidades_asociadas) if event.entidades_asociadas else 0
            ])
    
    if verbose:
        print(f"💾 Eventos guardados en CSV: {filepath}")


def get_statistics(events: List[Event]) -> Dict[str, Any]:
    """
    Calcula estadísticas sobre los eventos extraídos.
    
    Args:
        events: Lista de eventos
    
    Returns:
        Dict con estadísticas: tipos, sentimientos, fechas, entidades
    """
    from collections import Counter
    
    if not events:
        return {
            "total": 0,
            "tipos": {},
            "sentimientos": {},
            "entidades_top": []
        }
    
    # Contar tipos
    tipos = Counter(e.event_type.value for e in events)
    
    # Contar sentimientos
    sentimientos = Counter(e.sentiment.value for e in events)
    
    # Extraer todas las entidades
    all_entities = []
    for event in events:
        if event.entidades_asociadas:
            for ent in event.entidades_asociadas:
                all_entities.append(ent.get('text', ''))
    
    entidades_top = Counter(all_entities).most_common(10)
    
    # Rango de fechas
    fechas = [e.date for e in events]
    fecha_min = min(fechas)
    fecha_max = max(fechas)
    
    # Confianza promedio
    confianza_tipo_promedio = sum(e.confidence for e in events) / len(events)
    confianza_sentimiento_promedio = sum(e.sentiment_confidence for e in events) / len(events)
    
    return {
        "total": len(events),
        "tipos": dict(tipos),
        "sentimientos": dict(sentimientos),
        "entidades_top": entidades_top,
        "fecha_min": fecha_min.isoformat(),
        "fecha_max": fecha_max.isoformat(),
        "confianza_tipo_promedio": confianza_tipo_promedio,
        "confianza_sentimiento_promedio": confianza_sentimiento_promedio,
        "noticias_unicas": len(set(e.source_news_id for e in events))
    }


def print_statistics(events: List[Event]):
    """Imprime estadísticas formateadas de los eventos."""
    
    stats = get_statistics(events)
    
    print("\n" + "="*60)
    print("📊 ESTADÍSTICAS DETALLADAS")
    print("="*60)
    
    print(f"\n📈 General:")
    print(f"   Total de eventos: {stats['total']}")
    print(f"   Noticias únicas: {stats['noticias_unicas']}")
    print(f"   Rango de fechas: {stats['fecha_min'][:10]} a {stats['fecha_max'][:10]}")
    
    print(f"\n🏷️  Distribución por tipo:")
    for tipo, count in sorted(stats['tipos'].items(), key=lambda x: x[1], reverse=True):
        porcentaje = count / stats['total'] * 100
        print(f"   {tipo:15s}: {count:3d} ({porcentaje:5.1f}%)")
    
    print(f"\n😊 Distribución por sentimiento:")
    for sent, count in sorted(stats['sentimientos'].items(), key=lambda x: x[1], reverse=True):
        porcentaje = count / stats['total'] * 100
        print(f"   {sent:10s}: {count:3d} ({porcentaje:5.1f}%)")
    
    print(f"\n💯 Confianza promedio:")
    print(f"   Tipo: {stats['confianza_tipo_promedio']*100:.1f}%")
    print(f"   Sentimiento: {stats['confianza_sentimiento_promedio']*100:.1f}%")
    
    if stats['entidades_top']:
        print(f"\n👥 Top 10 entidades más mencionadas:")
        for i, (entidad, count) in enumerate(stats['entidades_top'], 1):
            print(f"   {i:2d}. {entidad:30s}: {count:3d} menciones")


def main():
    """Función principal cuando se ejecuta como script."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Procesa todas las noticias y extrae eventos. "
                   "Por defecto, auto-detecta los mejores clasificadores disponibles."
    )
    parser.add_argument(
        "--database",
        default="noticias.db",
        help="Ruta a la base de datos SQLite (default: noticias.db)"
    )
    parser.add_argument(
        "--force-sklearn-type",
        action="store_true",
        help="Forzar uso de clasificador sklearn para tipo (default: auto-detecta)"
    )
    parser.add_argument(
        "--force-keyword-type",
        action="store_true",
        help="Forzar uso de clasificador keyword para tipo (default: auto-detecta)"
    )
    parser.add_argument(
        "--sklearn-model",
        help="Ruta al modelo sklearn entrenado (opcional)"
    )
    parser.add_argument(
        "--force-sklearn-sentiment",
        action="store_true",
        help="Forzar uso de clasificador sklearn para sentimiento (default: auto-detecta)"
    )
    parser.add_argument(
        "--force-keyword-sentiment",
        action="store_true",
        help="Forzar uso de clasificador keyword para sentimiento (default: auto-detecta)"
    )
    parser.add_argument(
        "--limit",
        type=int,
        help="Número máximo de noticias a procesar"
    )
    parser.add_argument(
        "--json",
        help="Guardar eventos en archivo JSON"
    )
    parser.add_argument(
        "--csv",
        help="Guardar eventos en archivo CSV"
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="No mostrar progreso detallado"
    )
    parser.add_argument(
        "--stats",
        action="store_true",
        help="Mostrar estadísticas detalladas"
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Mostrar errores detallados durante el procesamiento"
    )
    
    args = parser.parse_args()
    
    # Determinar qué clasificadores usar basado en los argumentos
    use_sklearn_type = None  # Auto-detecta por defecto
    if args.force_sklearn_type:
        use_sklearn_type = True
    elif args.force_keyword_type:
        use_sklearn_type = False
    
    sentiment_classifier = None  # Auto-detecta por defecto
    if args.force_sklearn_sentiment:
        sentiment_classifier = "sklearn"
    elif args.force_keyword_sentiment:
        sentiment_classifier = "keyword"
    
    # Procesar noticias
    events = extract_events_from_all_news(
        database_path=args.database,
        use_sklearn_type_classifier=use_sklearn_type,
        sklearn_model_path=args.sklearn_model,
        sentiment_classifier=sentiment_classifier,
        limit=args.limit,
        save_to_json=args.json,
        save_to_csv=args.csv,
        verbose=not args.quiet,
        debug=args.debug
    )
    
    # Mostrar estadísticas si se solicitó
    if args.stats and events:
        print_statistics(events)
    
    return events


if __name__ == "__main__":
    main()
