from Instructions.GenerativeAI.vectordb import vectordb
from Instructions.GenerativeAI.rag import rag
from Instructions.GenerativeAI.foundation_model import foundation_model_explanation


GENAI_MAPPER = {
    "Foundation Model": [foundation_model_explanation, './Instructions/GenerativeAI/foundation.png'],
    "Vector Database": [vectordb, './Instructions/GenerativeAI/vectordb.png'],
    "Retrieval Augmented Generation": [rag, './Instructions/GenerativeAI/rag.png'],
}
