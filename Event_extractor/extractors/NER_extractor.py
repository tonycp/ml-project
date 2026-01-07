"""
Extractor de entidades nombradas (NER) para eventos.
"""
from utils.text_preprocessor import get_nlp

class NERExtractor:
    def __init__(self, model_path: str):
        """
        Inicializa el extractor NER con el modelo especificado.

        :param model_path: Ruta al modelo NER preentrenado.
        """
        self.model_path = model_path
        self.nlp = get_nlp()
        
        # Aquí se cargaría el modelo NER desde la ruta proporcionada
        # self.model = load_model(model_path)

    def extract_entities(self, text: str) -> list:
        """
        Extrae entidades nombradas del texto dado.

        :param text: Texto del cual extraer entidades.
        :return: Lista de entidades nombradas extraídas.
        """
        # Aquí se implementaría la lógica para extraer entidades usando el modelo cargado
        # entities = self.model.predict(text)
        # return entities
        return []  # Placeholder return statement