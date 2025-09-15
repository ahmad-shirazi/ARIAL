import re
import math

class SpatialGrounder:
    """
    Localizes the generated answer on the document image.
    """
    def ground(self, answer_text: str, ocr_results: list[dict], question: str = None) -> list[int]:
        """
        Finds the bounding box of the answer text in the document.
        
        Args:
            answer_text: The answer text to localize.
            ocr_results: A list of OCR result dictionaries.
            question: Optional question for contextual disambiguation.
            
        Returns:
            A bounding box list of coordinates [x1, y1, x2, y2].
        """
        # Direct exact match
        for res in ocr_results:
            if answer_text.lower() in res['text'].lower():
                return res['box']
        
        # Check if answer exists in the document at all
        all_text = " ".join([res['text'] for res in ocr_results])
        if answer_text.lower() not in all_text.lower():
            return None

        # Find multiple potential matches for disambiguation
        potential_matches = []
        for res in ocr_results:
            if answer_text.lower() in res['text'].lower():
                potential_matches.append(res)
        
        # If multiple matches found, use contextual cues for disambiguation
        if len(potential_matches) > 1 and question:
            best_match = self._disambiguate_with_context(potential_matches, question, ocr_results)
            if best_match:
                return best_match['box']

        # Try to find segments that make up the answer (for multi-segment answers)
        contributing_segments = []
        answer_words = set(answer_text.lower().split())
        
        for res in ocr_results:
            text_words = set(res['text'].lower().split())
            if answer_words.intersection(text_words):
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

    def _disambiguate_with_context(self, matches: list[dict], question: str, all_ocr_results: list[dict]) -> dict:
        """
        Uses contextual cues (proximity to question keywords) to select the best match.
        """
        question_keywords = set(re.findall(r'\w+', question.lower()))
        question_keywords = {word for word in question_keywords if len(word) > 2}  # Filter short words
        
        best_match = None
        best_score = -1
        
        for match in matches:
            score = self._calculate_context_score(match, question_keywords, all_ocr_results)
            if score > best_score:
                best_score = score
                best_match = match
                
        return best_match
    
    def _calculate_context_score(self, match: dict, question_keywords: set, all_ocr_results: list[dict]) -> float:
        """
        Calculates a context score based on proximity to question keywords.
        """
        match_box = match['box']
        match_center = [(match_box[0] + match_box[2]) / 2, (match_box[1] + match_box[3]) / 2]
        
        total_score = 0.0
        keyword_count = 0
        
        for ocr_result in all_ocr_results:
            text_words = set(re.findall(r'\w+', ocr_result['text'].lower()))
            common_keywords = question_keywords.intersection(text_words)
            
            if common_keywords:
                # Calculate distance between this OCR result and the match
                other_box = ocr_result['box']
                other_center = [(other_box[0] + other_box[2]) / 2, (other_box[1] + other_box[3]) / 2]
                
                distance = math.sqrt(
                    (match_center[0] - other_center[0])**2 + 
                    (match_center[1] - other_center[1])**2
                )
                
                # Closer keywords get higher scores (inverse distance)
                if distance > 0:
                    proximity_score = 1.0 / (1.0 + distance / 100.0)  # Normalize distance
                    total_score += proximity_score * len(common_keywords)
                    keyword_count += len(common_keywords)
        
        return total_score / max(1, keyword_count)  # Average score
