import re

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

        for res in ocr_results:
            text = res['text']
            if answer_text.lower() in text.lower():
                return res['box']
        
        all_text = " ".join([res['text'] for res in ocr_results])
        if answer_text.lower() not in all_text.lower():
            return None

        contributing_segments = []
        for res in ocr_results:
            if res['text'].lower() in answer_text.lower():
                contributing_segments.append(res)
        
        if not contributing_segments:
            return None

        boxes = [seg['box'] for seg in contributing_segments]
        x_min = min(box[0] for box in boxes)
        y_min = min(box[1] for box in boxes)
        x_max = max(box[2] for box in boxes)
        y_max = max(box[3] for box in boxes)
        
        return [x_min, y_min, x_max, y_max]
