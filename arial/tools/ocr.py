import torch
from doctr.io import DocumentFile
from doctr.models import ocr_predictor

class OCRProcessor:
    """
    Handles OCR and layout parsing of a document.
    """
    def __init__(self, det_model='db_resnet50', reco_model='trocr-base-handwritten', device=None):
        if device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device
        
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
            width, height = page.dimensions
            
            for block in page.blocks:
                for line in block.lines:
                    abs_coords = [
                        (
                            int(word.geometry[0][0] * width), 
                            int(word.geometry[0][1] * height), 
                            int(word.geometry[1][0] * width), 
                            int(word.geometry[1][1] * height)
                        ) 
                        for word in line.words
                    ]
                    
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
