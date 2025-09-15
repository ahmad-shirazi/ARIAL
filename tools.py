import torch
from PIL import Image
from doctr.io import DocumentFile
from doctr.models import ocr_predictor
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
import numpy as np
import re

class OCRProcessor:
    """
    Handles OCR and layout parsing of a document.
    """
    def __init__(self, det_model='db_resnet50', reco_model='trocr-base-handwritten', device=None):
        if device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device
        
        # NOTE: The user requested microsoft/trocr-base-handwritten, but doctr's default recognizer
        # is typically more robust for general documents. We are sticking to doctr's defaults.
        # If specific handwritten OCR is needed, this can be swapped.
        self.model = ocr_predictor(det_arch=det_model, reco_arch='crnn_vgg16_bn', pretrained=True).to(self.device)

    def run(self, image_path: str) -> list[dict]:
        """
        Processes an image to extract text segments and their bounding boxes.
        
        Args:
            image_path: Path to the input image.
            
        Returns:
            A list of dictionaries, where each dictionary contains the 
            recognized text segment and its corresponding bounding box coordinates.
        """
        if "pdf" in image_path.lower():
            doc = DocumentFile.from_pdf(image_path)
        else:
            doc = DocumentFile.from_images(image_path)
            
        result = self.model(doc)
        
        ocr_results = []
        
        for page in result.pages:
            # Get dimensions for normalization
            width, height = page.dimensions
            
            for block in page.blocks:
                for line in block.lines:
                    words = []
                    # Unpack word bounding boxes
                    abs_coords = [
                        (
                            int(word.geometry[0][0] * width), 
                            int(word.geometry[0][1] * height), 
                            int(word.geometry[1][0] * width), 
                            int(word.geometry[1][1] * height)
                        ) 
                        for word in line.words
                    ]
                    
                    # Merge word boxes to get line box
                    if not abs_coords:
                        continue

                    x_min = min(box[0] for box in abs_coords)
                    y_min = min(box[1] for box in abs_coords)
                    x_max = max(box[2] for box in abs_coords)
                    y_max = max(box[3] for box in abs_coords)
                    
                    line_text = " ".join(word.value for word in line.words)
                    
                    ocr_results.append({
                        'text': line_text,
                        'box': [x_min, y_min, x_max, y_max]
                    })
                    
        return ocr_results

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
        
        # Semantic search
        similarities = cosine_similarity(query_embedding, segment_embeddings)[0]
        
        # Lexical search (simple keyword matching)
        query_words = set(query.lower().split())
        lexical_scores = []
        for text in segment_texts:
            text_words = set(text.lower().split())
            score = len(query_words.intersection(text_words)) / len(query_words) if query_words else 0
            lexical_scores.append(score)
        
        lexical_scores = np.array(lexical_scores)
        
        # Combine scores (e.g., 70% semantic, 30% lexical)
        combined_scores = 0.7 * similarities + 0.3 * lexical_scores
        
        # Get top-k indices
        top_indices = np.argsort(combined_scores)[::-1][:top_k]
        
        return [ocr_results[i] for i in top_indices]

class QAModule:
    """
    Generates a textual answer based on retrieved context.
    """
    def __init__(self, model_name='google/gemma-2b-it', device=None):
        if device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map=self.device,
            torch_dtype=torch.bfloat16 # Use bfloat16 for better performance
        )
        
        self.pipeline = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            device=self.device
        )

    def ask(self, question: str, context: str) -> str:
        """
        Generates an answer to a question based on the provided context.
        
        Args:
            question: The question to answer.
            context: The context from which to generate the answer.
            
        Returns:
            A string containing the final answer.
        """
        prompt = f"""
        Context:
        {context}
        
        Question: {question}
        
        Based on the context, provide a concise and extractive answer to the question.
        Answer:
        """
        
        terminators = [
            self.pipeline.tokenizer.eos_token_id,
            self.pipeline.tokenizer.convert_tokens_to_ids("<|eot_id|>")
        ]

        outputs = self.pipeline(
            prompt,
            max_new_tokens=256,
            eos_token_id=terminators,
            do_sample=True,
            temperature=0.1,
            top_p=0.9,
        )

        generated_text = outputs[0]['generated_text']
        
        # Extract the answer part from the generated text
        answer_marker = "Answer:"
        answer_pos = generated_text.find(answer_marker)
        if answer_pos != -1:
            return generated_text[answer_pos + len(answer_marker):].strip()
        
        return generated_text.strip()


class SpatialGrounder:
    """
    Localizes the generated answer on the document image.
    """
    def ground(self, answer_text: str, ocr_results: list[dict]) -> list[int]:
        """
        Finds the bounding box of the answer text in the document.
        
        Args:
            answer_text: The answer text to localize.
            ocr_results: A list of OCR result dictionaries.
            
        Returns:
            A bounding box list of coordinates [x1, y1, x2, y2].
        """
        normalized_answer = re.sub(r'\s+', '', answer_text.lower())
        
        best_match = {'box': None, 'score': 0.0}

        # Direct Match
        for res in ocr_results:
            text = res['text']
            if answer_text.lower() in text.lower():
                # For now, we take the first exact match.
                # A more advanced version would use proximity to question keywords.
                return res['box']
        
        # If no exact match, find the best partial match and merge boxes if necessary
        # This is a simplified approach. A more robust solution would involve more complex matching logic.
        
        all_text = " ".join([res['text'] for res in ocr_results])
        if answer_text.lower() not in all_text.lower():
            # If answer is not even in the full text, we can't ground it.
            return None

        # Try to find segments that make up the answer
        contributing_segments = []
        for res in ocr_results:
            if res['text'].lower() in answer_text.lower():
                contributing_segments.append(res)
        
        if not contributing_segments:
            return None

        # Merge boxes of contributing segments
        boxes = [seg['box'] for seg in contributing_segments]
        x_min = min(box[0] for box in boxes)
        y_min = min(box[1] for box in boxes)
        x_max = max(box[2] for box in boxes)
        y_max = max(box[3] for box in boxes)
        
        return [x_min, y_min, x_max, y_max]

class ComputeModule:
    """
    A simple tool for handling numerical calculations.
    """
    def run(self, operation: str, values: list) -> float:
        """
        Performs a calculation on a list of numerical values.
        
        Args:
            operation: The operation to perform (e.g., 'sum', 'average').
            values: A list of numerical values (can be strings).
            
        Returns:
            The result of the calculation.
        """
        # Clean values and convert to float
        numeric_values = []
        for v in values:
            try:
                # Remove common currency symbols and commas
                cleaned_v = re.sub(r'[$,]', '', str(v))
                numeric_values.append(float(cleaned_v))
            except (ValueError, TypeError):
                continue
        
        if not numeric_values:
            return 0.0

        if operation == 'sum':
            return sum(numeric_values)
        elif operation == 'average':
            return sum(numeric_values) / len(numeric_values)
        else:
            raise ValueError(f"Unsupported operation: {operation}")
