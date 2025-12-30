# Description: This file contains the functions to process the texts.
import os
import spacy as sp
from tqdm import tqdm

# from constants import TEXT_ENCODE
TEXT_ENCODE = "utf-8"

nlp = sp.load("es_core_news_sm")

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

def _tokenize_text(text :str):
    """Tokenizes the given text.

    Args:
        text (str): text to be tokenized.

    Returns:
        list[str]: list of tokens. Each element of the list is a word of the text. The words are in lowercase, without punctuation and lemmatized.
    """    
    # Divide the text into fragments
    max_length = nlp.max_length
    fragments = []
    current_fragment = ""
    for word in text.split():
        if len(current_fragment) + len(word) + 1 > max_length:
            fragments.append(current_fragment)
            current_fragment = ""
        current_fragment += word + " "
    if current_fragment:
        fragments.append(current_fragment)
 
    # Tokenize each fragment separately using nlp.pipe
    token_list = [token for doc in nlp.pipe(fragments, disable=["parser", "ner"]) for token in doc]
        
    # Filter out punctuation and stop words, and lemmatize
    tokens = [token.lemma_.lower() for token in token_list if not token.is_punct or not token.is_stop]
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
