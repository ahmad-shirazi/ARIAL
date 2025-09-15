import torch
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

class TextRetriever:
    """
    Retrieves relevant text segments based on a query.
    """
    def __init__(self, model_name='all-MiniLM-L6-v2', device=None):
        if device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device
        self.model = SentenceTransformer(model_name, device=self.device)

    def find(self, query: str, ocr_results: list[dict], top_k: int = 5) -> list[dict]:
        """
        Finds the most relevant text segments from OCR results.
        
        Args:
            query: The input query string.
            ocr_results: A list of OCR result dictionaries.
            top_k: The number of top results to return.
            
        Returns:
            A filtered list of OCR result dictionaries, ranked by relevance.
        """
        if not ocr_results:
            return []

        query_embedding = self.model.encode([query])
        
        segment_texts = [res['text'] for res in ocr_results]
        segment_embeddings = self.model.encode(segment_texts)
        
        similarities = cosine_similarity(query_embedding, segment_embeddings)[0]
        
        query_words = set(query.lower().split())
        lexical_scores = []
        for text in segment_texts:
            text_words = set(text.lower().split())
            score = len(query_words.intersection(text_words)) / len(query_words) if query_words else 0
            lexical_scores.append(score)
        
        lexical_scores = np.array(lexical_scores)
        
        combined_scores = 0.7 * similarities + 0.3 * lexical_scores
        
        top_indices = np.argsort(combined_scores)[::-1][:top_k]
        
        return [ocr_results[i] for i in top_indices]
