import torch
from PIL import Image, ImageOps, ImageEnhance
from doctr.io import DocumentFile
from doctr.models import ocr_predictor
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
import numpy as np

class OCRProcessor:
    """
    Handles OCR and layout parsing of a document.
    Implements two-stage OCR: text detection followed by recognition.
    """
    def __init__(self, det_model='db_resnet50', reco_model='microsoft/trocr-base-handwritten', device=None):
        if device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device
        
        # Stage 1: Text Detection using doctr
        self.detector = ocr_predictor(det_arch=det_model, reco_arch='crnn_vgg16_bn', pretrained=True).to(self.device)
        
        # Stage 2: Text Recognition using TrOCR
        self.trocr_processor = TrOCRProcessor.from_pretrained(reco_model)
        self.trocr_model = VisionEncoderDecoderModel.from_pretrained(reco_model).to(self.device)

    def preprocess_image(self, image: Image.Image) -> Image.Image:
        """
        Applies preprocessing steps like resizing, denoising, and contrast enhancement.
        """
        # Convert to RGB if necessary
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Resize to uniform resolution (maintaining aspect ratio)
        max_size = 2048
        if max(image.size) > max_size:
            image.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)
        
        # Enhance contrast
        enhancer = ImageEnhance.Contrast(image)
        image = enhancer.enhance(1.2)
        
        # Enhance sharpness
        enhancer = ImageEnhance.Sharpness(image)
        image = enhancer.enhance(1.1)
        
        return image

    def run(self, image_path: str) -> list[dict]:
        """
        Processes an image to extract text segments and their bounding boxes.
        Implements two-stage OCR: detection then recognition with TrOCR.
        
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
            
        # Stage 1: Text Detection
        detection_result = self.detector(doc)
        
        ocr_results = []
        
        # Get the original image for cropping
        if "pdf" in image_path.lower():
            # For PDFs, we'd need to convert to images first
            original_image = None  # This would need PDF to image conversion
        else:
            original_image = Image.open(image_path)
            original_image = self.preprocess_image(original_image)
        
        for page in detection_result.pages:
            width, height = page.dimensions
            
            for block in page.blocks:
                for line in block.lines:
                    # Get line bounding box
                    line_coords = []
                    for word in line.words:
                        line_coords.extend([
                            [word.geometry[0][0] * width, word.geometry[0][1] * height],
                            [word.geometry[1][0] * width, word.geometry[1][1] * height]
                        ])
                    
                    if not line_coords:
                        continue

                    # Calculate line bounding box
                    x_coords = [coord[0] for coord in line_coords]
                    y_coords = [coord[1] for coord in line_coords]
                    x_min, x_max = int(min(x_coords)), int(max(x_coords))
                    y_min, y_max = int(min(y_coords)), int(max(y_coords))
                    
                    # Stage 2: Text Recognition with TrOCR
                    if original_image:
                        # Crop the text region
                        # Add padding to improve recognition
                        padding = 5
                        crop_box = (
                            max(0, x_min - padding),
                            max(0, y_min - padding),
                            min(original_image.width, x_max + padding),
                            min(original_image.height, y_max + padding)
                        )
                        
                        cropped_image = original_image.crop(crop_box)
                        
                        # Use TrOCR for recognition
                        pixel_values = self.trocr_processor(cropped_image, return_tensors="pt").pixel_values.to(self.device)
                        generated_ids = self.trocr_model.generate(pixel_values)
                        recognized_text = self.trocr_processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
                    else:
                        # Fallback to doctr's recognition for PDFs
                        recognized_text = " ".join(word.value for word in line.words)
                    
                    ocr_results.append({
                        'text': recognized_text,
                        'box': [x_min, y_min, x_max, y_max]
                    })
                    
        return ocr_results
