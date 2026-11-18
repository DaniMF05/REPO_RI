import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer


try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)
    nltk.download('punkt_tab', quiet=True) 

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords', quiet=True)

# Inicializar componentes globales fuera de la función.
stop_words = set(stopwords.words('english'))
stemmer = PorterStemmer()

def preprocess(text: str) -> list[str]:
    """
    Normaliza, tokeniza, elimina stopwords y aplica stemming a un texto.
    
    Args:
        text (str): El texto original a procesar.
        
    Returns:
        list[str]: Lista de tokens procesados.
    """
    if not isinstance(text, str) or not text.strip():
        return []
        
    # 1. Normalización: Pasar a minúsculas
    text = text.lower()
    
    # 2. Normalización: Eliminar puntuación
    text = re.sub(r'[^a-z0-9\s]', ' ', text)
    
    # 3. Tokenización
    tokens = word_tokenize(text)
    
    # 4 y 5. Eliminación de stopwords y Stemming
    processed_tokens = [
        stemmer.stem(word) 
        for word in tokens 
        if word not in stop_words and len(word) > 1
    ]
    
    return processed_tokens