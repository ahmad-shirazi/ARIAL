#!/usr/bin/env python3
"""
Example usage of the ARIAL framework for Document Visual Question Answering.
"""
import os
from arial.agent.planner_agent import PlannerAgent
from arial.tools import (
    OCRProcessor,
    TextRetriever,
    QAModule,
    SpatialGrounder,
    ComputeModule,
)

def main():
    """
    Example usage of the ARIAL framework.
    """
    # Set up document and questions
    document_path = "documents/sample_document.pdf"
    questions = [
        "What is the total amount due?",
        "When is the invoice date?",
        "What is the invoice number?",
        "What is the sum of all line items?"
    ]
    
    print("ARIAL Framework Example")
    print("=" * 50)
    
    if not os.path.exists(document_path):
        print(f"Error: Sample document not found at {document_path}")
        print("Please ensure you have a document in the documents/ folder.")
        return
    
    # Initialize all components
    print("Initializing ARIAL components...")
    
    try:
        ocr_processor = OCRProcessor()
        text_retriever = TextRetriever()
        qa_module = QAModule()
        spatial_grounder = SpatialGrounder()
        compute_module = ComputeModule()
        
        tools = {
            "RunOCR": ocr_processor,
            "FindText": text_retriever,
            "AskQA": qa_module,
            "GroundAnswer": spatial_grounder,
            "Compute": compute_module
        }
        
        agent = PlannerAgent(tools)
        print("All components initialized successfully!")
        
    except Exception as e:
        print(f"Error initializing components: {e}")
        return
    
    # Process each question
    for i, question in enumerate(questions, 1):
        print(f"\n{'='*50}")
        print(f"Question {i}: {question}")
        print('='*50)
        
        try:
            answer, bounding_box = agent.run(document_path, question)
            
            print(f"\nResult:")
            print(f"Answer: {answer}")
            print(f"Bounding Box: {bounding_box}")
            
        except Exception as e:
            print(f"Error processing question: {e}")
        
        print("\n" + "-" * 50)

if __name__ == "__main__":
    main()
