# Description: This file contains the functions to process the texts.
import os
import spacy as sp
from tqdm import tqdm
from typing import List

# from constants import TEXT_ENCODE
TEXT_ENCODE = "utf-8"

# Initialize the SpaCy model globally
_nlp = sp.load("es_core_news_lg")

PROCESSED_TEXT = None

def get_nlp():
    """Provides the globally initialized SpaCy model."""
    return _nlp


def tokenize_texts_batch(texts: List[str], batch_size: int = 100) -> List[List[str]]:
    """
    Tokeniza múltiples textos de manera eficiente usando nlp.pipe().
    
    Args:
        texts: Lista de textos a tokenizar
        batch_size: Tamaño del lote para procesamiento
        
    Returns:
        Lista de listas de tokens (uno por texto)
    """
    nlp = get_nlp()
    all_tokens = []
    
    # Procesar en lotes para eficiencia
    for doc in nlp.pipe(texts, batch_size=batch_size, disable=["parser", "ner"]):
        # Filter out punctuation and stop words, and lemmatize
        tokens = [token.lemma_.lower() for token in doc if not token.is_punct and not token.is_stop]
        # Conservar palabras y números
        words = [token for token in tokens if token != "_" and all(char.isalnum() or char == "_" for char in token)]
        all_tokens.append(words)
    
    return all_tokens

def get_tokens(path :str):
    """Returns the tokens of the texts in the given path.

    Args:
        path (str): path to the files.

    Returns:
        list[list[str]]: list of lists of tokens. Each element of the list is a list of tokens of a different text.
    """    
    tokens = []
    print("Reading texts...")
    texts = _read_texts(path)
    for text in tqdm(texts):
        tokens.append(_tokenize_text(text))
    print("Texts tokenized.")
    return tokens

def get_processed_text(text, force = False):
    """
    Provides the processed SpaCy document for the given text.

    Args:
        text (str): Text to be processed.
        force (bool, optional): If True, forces re-processing even if a processed document is cached. Defaults to False.

    Returns:
        spacy.tokens.Doc: Processed SpaCy document.
    """
    global PROCESSED_TEXT
    if force or PROCESSED_TEXT is None:
        PROCESSED_TEXT = process_text(text)
    return PROCESSED_TEXT
    
def process_text(text: str):
    return _nlp(text)
    
def _tokenize_text(text :str):
    """Tokenizes the given text.

    Args:
        text (str): text to be tokenized.

    Returns:
        list[str]: list of tokens. Each element of the list is a word of the text. The words are in lowercase, without punctuation and lemmatized.
    """    
    # IMPORTANTE: No usar get_processed_text() porque cachea el resultado
    # Procesar directamente el texto para evitar problemas con múltiples textos
    doc = process_text(text)
    
    # Filter out punctuation and stop words, and lemmatize
    tokens = [token.lemma_.lower() for token in doc if not token.is_punct and not token.is_stop]
    # Conservar palabras y números
    words = [token for token in tokens if token != "_" and all(char.isalnum() or char == "_" for char in token)]
    return words

def _read_texts(path : str):
    """Extracts the text of the .txt files in the given path.
    
    Parameters:
    path:str - path to the files

    Returns:
    doc_list:list[str] - list of the texts stored in the files of the given path. Each element of the list is the text of a different file.
    """
    doc_list = []
    print("Searching text files...")
    for filename in os.listdir(path):
        if filename.endswith(".txt"):
            with open(os.path.join(path, filename), 'r', encoding=TEXT_ENCODE, errors='ignore') as f:
                text = f.read().replace('\n', ' ')
                doc_list.append(text)
        print("Files found: " + str(len(doc_list)), end="\r")
    print("Texts found: " + str(len(doc_list)))
    return doc_list

def extract_svo(doc):
    """Extracts the subject-verb-object (SVO) triples from the given SpaCy document.

    Args:
        doc (spacy.tokens.Doc): SpaCy document to extract SVO triples from.

    Returns:
        list[tuple[str, str, str]]: list of SVO triples. Each element of the list is a tuple (subject, verb, object).
    """    
    resultados = []
    for token in doc:
        if token.pos_ == "VERB":
            sujetos = [child for child in token.children if child.dep_ in ("nsubj", "nsubjpass")]
            objetos = [child for child in token.children if child.dep_ in ("dobj", "obj", "iobj")]
            for subj in sujetos:
                for obj in objetos:
                    s_text = " ".join([t.text for t in subj.subtree])
                    o_text = " ".join([t.text for t in obj.subtree])
                    resultados.append((s_text, token.text, o_text))
    return resultados