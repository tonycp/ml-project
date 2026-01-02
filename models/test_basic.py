#!/usr/bin/env python3
"""
Tests básicos para verificar el funcionamiento del sistema de forecasting.
"""

import sys
from pathlib import Path

# Añadir el directorio padre al path
sys.path.append(str(Path(__file__).parent.parent))

from models import ModelConfig, ATCAircraftDataLoader, RandomForestModel


def test_data_loading():
    """Test básico de carga de datos."""
    print("Testing data loading...")

    try:
        config = ModelConfig()
        loader = ATCAircraftDataLoader(config)

        # Test info de datos
        info = loader.get_data_info()
        assert 'daily_atc' in info, "No se encontró información de datos diarios"

        if 'records' in info['daily_atc']:
            records = info['daily_atc']['records']
            assert records > 0, f"No hay registros en datos diarios: {records}"
            print(f"✓ Datos diarios: {records} registros")
        else:
            print("⚠️ Datos diarios no disponibles para test")
            return False

        # Test carga de datos
        df = loader.get_training_data('daily_atc')
        assert len(df) > 0, "No se cargaron datos de entrenamiento"
        assert 'total' in df.columns, "Columna 'total' no encontrada"
        print(f"✓ Datos de entrenamiento cargados: {len(df)} registros")

        return True

    except Exception as e:
        print(f"✗ Error en carga de datos: {e}")
        return False


def test_config():
    """Test de configuración."""
    print("Testing configuration...")

    try:
        config = ModelConfig()

        # Verificar rutas
        assert config.data_dir.exists(), f"Directorio de datos no existe: {config.data_dir}"

        # Verificar configuración de modelos
        assert 'arima' in config.models, "Configuración ARIMA no encontrada"
        assert 'prophet' in config.models, "Configuración Prophet no encontrada"

        print("✓ Configuración válida")
        return True

    except Exception as e:
        print(f"✗ Error en configuración: {e}")
        return False


def test_imports():
    """Test de imports."""
    print("Testing imports...")

    try:
        from models import (
            ATCAircraftDataLoader,
            AircraftDataPreprocessor,
            AircraftFeatureEngineer,
            AircraftForecaster,
            ARIMAModel,
            ProphetModel,
            RandomForestModel
        )
        print("✓ Todos los imports exitosos")
        return True

    except ImportError as e:
        print(f"✗ Error de import: {e}")
        return False


def main():
    """Función principal de tests."""
    print("🧪 Tests Básicos - Sistema de Forecasting de Aeronaves")
    print("=" * 60)

    tests = [
        ("Imports", test_imports),
        ("Configuración", test_config),
        ("Carga de Datos", test_data_loading),
    ]

    passed = 0
    total = len(tests)

    for test_name, test_func in tests:
        print(f"\n🔍 {test_name}")
        print("-" * 30)
        if test_func():
            passed += 1
        print()

    print("=" * 60)
    print(f"📊 Resultados: {passed}/{total} tests pasaron")

    if passed == total:
        print("✅ Todos los tests básicos pasaron exitosamente")
        print("\n🚀 El sistema está listo para usar")
        print("Ejecutar 'python models/example_usage.py' para un ejemplo completo")
        return 0
    else:
        print("❌ Algunos tests fallaron")
        print("Revisar los errores arriba y verificar la instalación")
        return 1


if __name__ == "__main__":
    sys.exit(main())