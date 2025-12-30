# Guía de Instalación - Event Extractor

Esta guía te ayudará a instalar y configurar el paquete Event Extractor.

## Requisitos Previos

- Python 3.8 o superior
- pip (gestor de paquetes de Python)
- git (opcional, para clonar el repositorio)

## Instalación Paso a Paso

### 1. Clonar el Repositorio (opcional)

Si quieres instalar desde el código fuente:

```bash
git clone https://github.com/tonycp/ml-project.git
cd ml-project
```

### 2. Crear un Entorno Virtual (recomendado)

Es recomendable usar un entorno virtual para evitar conflictos de dependencias:

```bash
# En Linux/Mac
python3 -m venv venv
source venv/bin/activate

# En Windows
python -m venv venv
venv\Scripts\activate
```

### 3. Instalar el Paquete

#### Opción A: Instalación en modo desarrollo (recomendada para desarrollo)

```bash
pip install -e .
```

Esta opción instala el paquete en modo "editable", lo que significa que los cambios en el código fuente se reflejarán inmediatamente sin necesidad de reinstalar.

#### Opción B: Instalación normal

```bash
pip install .
```

#### Opción C: Instalación con dependencias de desarrollo

```bash
pip install -e ".[dev]"
```

### 4. Instalar el Modelo de spaCy

El paquete requiere el modelo de spaCy para español:

```bash
python -m spacy download es_core_news_sm
```

### 5. Verificar la Instalación

Ejecuta Python y verifica que puedes importar el paquete:

```python
python3 -c "from Event_extractor import EventExtractionPipeline; print('✓ Instalación exitosa')"
```

## Instalación de Dependencias Manualmente

Si prefieres instalar las dependencias manualmente:

```bash
pip install -r requirements.txt
python -m spacy download es_core_news_sm
```

## Solución de Problemas

### Error: "No module named 'spacy'"

Solución:
```bash
pip install spacy
```

### Error: "Can't find model 'es_core_news_sm'"

Solución:
```bash
python -m spacy download es_core_news_sm
```

### Error de permisos al instalar

Solución en Linux/Mac:
```bash
pip install --user -e .
```

Solución en Windows (ejecutar como administrador):
```bash
pip install -e .
```

### Error: "ModuleNotFoundError: No module named 'Event_extractor'"

Asegúrate de:
1. Estar en el directorio correcto (ml-project/)
2. Haber activado el entorno virtual
3. Haber ejecutado `pip install -e .`

## Probar la Instalación

Una vez instalado, ejecuta los ejemplos:

```bash
cd examples
python basic_usage.py
```

Si ves la salida con eventos extraídos, ¡la instalación fue exitosa!

## Desinstalación

Para desinstalar el paquete:

```bash
pip uninstall event-extractor
```

## Actualización

Para actualizar a la última versión:

```bash
git pull origin main  # Si clonaste el repositorio
pip install -e . --upgrade
```

## Soporte

Si encuentras problemas durante la instalación, por favor:
1. Revisa esta guía nuevamente
2. Verifica que cumples con todos los requisitos previos
3. Abre un issue en: https://github.com/tonycp/ml-project/issues
