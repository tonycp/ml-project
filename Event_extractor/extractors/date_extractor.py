"""
Extractor de fechas de texto en español.
"""

from typing import List, Optional
from datetime import datetime, timedelta
import re
import spacy
from dateutil import parser
from dateutil.relativedelta import relativedelta


class DateExtractor:
    """
    Extractor de fechas de texto en español utilizando reglas y NLP.
    
    Identifica fechas explícitas, expresiones relativas y rangos de fechas.
    Cuando se detecta un rango de fechas (inicio y fin), se extraen ambas
    como fechas separadas.
    """
    
    def __init__(self, reference_date: Optional[datetime] = None):
        """
        Inicializa el extractor de fechas.
        
        Args:
            reference_date: Fecha de referencia para resolver fechas relativas.
                          Ha de ser la fecha de publicación de la noticia (en la metadata de la noticia).
                          Si no se proporciona, NO se procesarán fechas relativas ni fechas sin año
                          para evitar guardar fechas erróneas.
        """
        self.reference_date = reference_date
        self._has_reference_date = reference_date is not None
        
        # Mapeo de meses
        self.meses_map = {
            'enero': 1, 'febrero': 2, 'marzo': 3, 'abril': 4,
            'mayo': 5, 'junio': 6, 'julio': 7, 'agosto': 8,
            'septiembre': 9, 'octubre': 10, 'noviembre': 11, 'diciembre': 12
        }
        self.weekdays_map = {
            'lunes': 0, 'martes': 1, 'miércoles': 2, 'miercoles': 2,
            'jueves': 3, 'viernes': 4, 'sábado': 5, 'sabado': 5,
            'domingo': 6
        }
    
    def extract_dates(self, tokens: List[str]) -> List[datetime]:
        """
        Extrae todas las fechas de una lista de tokens preprocesados.
        
        Args:
            tokens: Lista de tokens (palabras) del texto preprocesado
            
        Returns:
            Lista de objetos datetime con las fechas encontradas
        """
        dates = []
        
        # Extraer fechas de rangos (ej: "1 enero 5 enero 2024")
        dates.extend(self._extract_date_ranges(tokens))
        
        # extraer fechas de rangos con "del ... al ..." (ej: "del 10 al 15 de enero de 2025")
        dates.extend(self._extract_simple_range(tokens))
        
        # Extraer fechas explícitas completas ("25 diciembre 2024")
        dates.extend(self._extract_explicit_dates(tokens))
        
        # Extraer fechas numéricas ("25/12/2024")
        dates.extend(self._extract_numeric_dates(tokens))
        
        # Extraer fechas relativas solo si hay reference_date
        if self._has_reference_date:
            dates.extend(self._extract_relative_dates(tokens))
        
        unique_dates = list(set(dates))
        unique_dates.sort()
        
        return unique_dates
    
    def _extract_date_ranges(self, tokens: List[str]) -> List[datetime]:
        """
        Extrae rangos de fechas y devuelve la fecha de inicio y fin como fechas separadas.
        
        Args:
            tokens: Lista de tokens del texto preprocesado
            
        Returns:
            Lista con las fechas de inicio y fin
        """
        dates = []
        
        # Buscar patrones tipo: 1 enero 5 enero 2024
        for i in range(len(tokens) - 4):
            try:
                if (tokens[i].isdigit() and tokens[i+1] in self.meses_map and
                    tokens[i+2].isdigit() and tokens[i+3] in self.meses_map):
                    dia_inicio = int(tokens[i])
                    mes_inicio_str = tokens[i+1]
                    dia_fin = int(tokens[i+2])
                    mes_fin_str = tokens[i+3]
                    anio = None
                    if tokens[i+4].isdigit() and len(tokens[i+4]) == 4:
                        anio = int(tokens[i+4])
                    elif self._has_reference_date and self.reference_date is not None:
                        anio = self.reference_date.year
                    else:
                        continue
                    
                    mes_inicio = self.meses_map.get(mes_inicio_str)
                    mes_fin = self.meses_map.get(mes_fin_str)
                    
                    if mes_inicio and mes_fin:
                        # Fecha de inicio
                        fecha_inicio = datetime(anio, mes_inicio, dia_inicio)
                        dates.append(fecha_inicio)
                        
                        # Fecha de fin
                        fecha_fin = datetime(anio, mes_fin, dia_fin)
                        dates.append(fecha_fin)
            except Exception:
                continue
        
        return dates
    
    def _extract_simple_range(self, tokens: List[str]) -> List[datetime]:
        """
        Extrae fechas de la forma: <num> <num> <mes> <año> (ej: 10 15 enero 2025, equivalente a "del 10 al 15 de enero de 2025" en el texto sin preprocesar).
        Devuelve ambas fechas como objetos datetime.
        """
        dates = []
        for i in range(len(tokens) - 3):
            try:
                if tokens[i].isdigit() and tokens[i+1].isdigit() and tokens[i+2] in self.meses_map:
                    dia_inicio = int(tokens[i])
                    dia_fin = int(tokens[i+1])
                    mes_str = tokens[i+2]
                    anio = None
                    if i+3 < len(tokens) and tokens[i+3].isdigit() and len(tokens[i+3]) == 4:
                        anio = int(tokens[i+3])
                    elif self._has_reference_date and self.reference_date is not None:
                        anio = self.reference_date.year
                    else:
                        continue
                    mes = self.meses_map.get(mes_str)
                    if mes:
                        fecha_inicio = datetime(anio, mes, dia_inicio)
                        fecha_fin = datetime(anio, mes, dia_fin)
                        dates.append(fecha_inicio)
                        dates.append(fecha_fin)
            except Exception:
                continue
        return dates
    
    def _extract_explicit_dates(self, tokens: List[str]) -> List[datetime]:
        """Extrae fechas en formato "DD MMMM YYYY" y "DD MMMM"."""
        dates = []
        
        # Buscar patrones tipo: 25 diciembre 2024
        for i in range(len(tokens) - 2):
            try:
                if tokens[i].isdigit() and tokens[i+1] in self.meses_map and tokens[i+2].isdigit() and len(tokens[i+2]) == 4:
                    dia = int(tokens[i])
                    mes_str = tokens[i+1]
                    anio = int(tokens[i+2])
                    
                    mes = self.meses_map.get(mes_str)
                    if mes:
                        fecha = datetime(anio, mes, dia)
                        dates.append(fecha)
            except Exception:
                continue
        
        # Fechas sin año (solo si hay reference_date)
        if self._has_reference_date and self.reference_date is not None:
            for i in range(len(tokens) - 1):
                try:
                    if tokens[i].isdigit() and tokens[i+1] in self.meses_map:
                        dia = int(tokens[i])
                        mes_str = tokens[i+1]
                        mes = self.meses_map.get(mes_str)
                        if mes:
                            fecha = datetime(self.reference_date.year, mes, dia)
                            dates.append(fecha)
                except Exception:
                    continue
        
        return dates
    
    def _extract_numeric_dates(self, tokens: List[str]) -> List[datetime]:
        """Extrae fechas en formato numérico DD/MM/YYYY o DD-MM-YYYY."""
        dates = []
        for token in tokens:
            if re.match(r"\d{1,2}[/-]\d{1,2}[/-]\d{4}", token):
                try:
                    parts = re.split(r"[/-]", token)
                    dia = int(parts[0])
                    mes = int(parts[1])
                    anio = int(parts[2])
                    
                    fecha = datetime(anio, mes, dia)
                    dates.append(fecha)
                except Exception:
                    continue
        
        return dates
    
    def _extract_relative_dates(self, tokens: List[str]) -> List[datetime]:
        """Extrae fechas relativas (hoy, mañana, etc.) de una lista de tokens. Requiere reference_date."""
        dates = []
        if not self._has_reference_date or self.reference_date is None:
            return dates
        for i, token in enumerate(tokens):
            expr = token.lower().strip()
            if expr == "hoy":
                dates.append(self.reference_date)
            elif expr == "mañana":
                dates.append(self.reference_date + timedelta(days=1))
            elif expr == "ayer":
                dates.append(self.reference_date - timedelta(days=1))
            elif expr == "pasado" and i+1 < len(tokens) and tokens[i+1] == "mañana":
                dates.append(self.reference_date + timedelta(days=2))
            elif expr == "próxima" and i+1 < len(tokens) and tokens[i+1] == "semana":
                dates.append(self.reference_date + timedelta(weeks=1))
            elif expr == "próximo" and i+1 < len(tokens) and tokens[i+1] == "mes":
                dates.append(self.reference_date + relativedelta(months=1))
            elif expr == "próximo" and i+1 < len(tokens) and tokens[i+1] == "año":
                dates.append(self.reference_date + relativedelta(years=1))
            elif expr == "última" and i+1 < len(tokens) and tokens[i+1] == "semana":
                dates.append(self.reference_date - timedelta(weeks=1))
            elif expr == "último" and i+1 < len(tokens) and tokens[i+1] == "mes":
                dates.append(self.reference_date - relativedelta(months=1))
            elif expr == "último" and i+1 < len(tokens) and tokens[i+1] == "año":
                dates.append(self.reference_date - relativedelta(years=1))
            elif expr == "próximo" and i+1 < len(tokens) and tokens[i+1] in self.meses_map:
                mes_str = tokens[i+1]
                mes = self.meses_map.get(mes_str)
                year = self.reference_date.year
                if mes < self.reference_date.month:
                    year += 1
                try:
                    fecha = datetime(year, mes, 1)
                    dates.append(fecha)
                except Exception:
                    continue
            elif expr == "este" and i+1 < len(tokens) and tokens[i+1] in self.meses_map:
                mes_str = tokens[i+1]
                mes = self.meses_map.get(mes_str)
                year = self.reference_date.year
                try:
                    fecha = datetime(year, mes, 1)
                    dates.append(fecha)
                except Exception:
                    continue
            elif expr == "último" and i+1 < len(tokens) and tokens[i+1] in self.meses_map:
                mes_str = tokens[i+1]
                mes = self.meses_map.get(mes_str)
                year = self.reference_date.year
                if mes > self.reference_date.month:
                    year -= 1
                try:
                    fecha = datetime(year, mes, 1)
                    dates.append(fecha)
                except Exception:
                    continue
            elif expr == "próximo" and i+2 < len(tokens) and tokens[i+1].isdigit() and tokens[i+2] in self.meses_map:
                dia = int(tokens[i+1])
                mes_str = tokens[i+2]
                mes = self.meses_map.get(mes_str)
                year = self.reference_date.year
                if mes < self.reference_date.month or (mes == self.reference_date.month and dia < self.reference_date.day):
                    year += 1
                try:
                    fecha = datetime(year, mes, dia)
                    dates.append(fecha)
                except Exception:
                    continue
            elif expr == "este" and i+2 < len(tokens) and tokens[i+1].isdigit() and tokens[i+2] in self.meses_map:
                dia = int(tokens[i+1])
                mes_str = tokens[i+2]
                mes = self.meses_map.get(mes_str)
                year = self.reference_date.year
                try:
                    fecha = datetime(year, mes, dia)
                    dates.append(fecha)
                except Exception:
                    continue
            elif expr == "último" and i+2 < len(tokens) and tokens[i+1].isdigit() and tokens[i+2] in self.meses_map:
                dia = int(tokens[i+1])
                mes_str = tokens[i+2]
                mes = self.meses_map.get(mes_str)
                year = self.reference_date.year
                if mes > self.reference_date.month or (mes == self.reference_date.month and dia > self.reference_date.day):
                    year -= 1
                try:
                    fecha = datetime(year, mes, dia)
                    dates.append(fecha)
                except Exception:
                    continue
            #Con los dias de la semana
            elif expr == "próximo" and i+1 < len(tokens) and tokens[i+1] in self.weekdays_map:
                reference_week_day = self.reference_date.weekday()
                target_week_day = self.weekdays_map[tokens[i+1]]
                if target_week_day > reference_week_day:
                    days_ahead = target_week_day - reference_week_day
                else:
                    days_ahead = 7 - (reference_week_day - target_week_day)
                fecha = self.reference_date + timedelta(days=days_ahead)
                dates.append(fecha)
            elif expr == "último" and i+1 < len(tokens) and tokens[i+1] in self.weekdays_map:
                reference_week_day = self.reference_date.weekday()
                target_week_day = self.weekdays_map[tokens[i+1]]
                if target_week_day < reference_week_day:
                    days_behind = reference_week_day - target_week_day
                else:
                    days_behind = 7 - (target_week_day - reference_week_day)
                fecha = self.reference_date - timedelta(days=days_behind)
                dates.append(fecha)
            elif expr in self.weekdays_map:
                reference_week_day = self.reference_date.weekday()
                target_week_day = self.weekdays_map[expr]
                days_ahead = (target_week_day - reference_week_day) % 7
                fecha = self.reference_date + timedelta(days=days_ahead)
                dates.append(fecha)
            elif expr == "este" and i+1 < len(tokens) and tokens[i+1] in self.weekdays_map:
                reference_week_day = self.reference_date.weekday()
                target_week_day = self.weekdays_map[tokens[i+1]]
                days_ahead = (target_week_day - reference_week_day) % 7
                fecha = self.reference_date + timedelta(days=days_ahead)
                dates.append(fecha)
            # caso "siguiente mes" o "mes siguiente" (lo mismo con anio)
            elif expr == "siguiente" and i+1 < len(tokens) and tokens[i+1] == "mes":
                fecha = self.reference_date + relativedelta(months=1)
                dates.append(fecha)
            elif expr == "mes" and i+1 < len(tokens) and tokens[i+1] == "siguiente":
                fecha = self.reference_date + relativedelta(months=1)
                dates.append(fecha)
            elif expr == "siguiente" and i+1 < len(tokens) and tokens[i+1] == "año":
                fecha = self.reference_date + relativedelta(years=1)
                dates.append(fecha)
            elif expr == "año" and i+1 < len(tokens) and tokens[i+1] == "siguiente":
                fecha = self.reference_date + relativedelta(years=1)
                dates.append(fecha)
            # anterior mes / año
            elif expr == "anterior" and i+1 < len(tokens) and tokens[i+1] == "mes":
                fecha = self.reference_date - relativedelta(months=1)
                dates.append(fecha)
            elif expr == "mes" and i+1 < len(tokens) and tokens[i+1] == "anterior":
                fecha = self.reference_date - relativedelta(months=1)
                dates.append(fecha)
            elif expr == "anterior" and i+1 < len(tokens) and tokens[i+1] == "año":
                fecha = self.reference_date - relativedelta(years=1)
                dates.append(fecha)
            elif expr == "año" and i+1 < len(tokens) and tokens[i+1] == "anterior":
                fecha = self.reference_date - relativedelta(years=1)
                dates.append(fecha)
        return dates
