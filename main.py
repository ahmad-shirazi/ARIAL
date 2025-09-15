import argparse
import os
from dotenv import load_dotenv

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
    Main function to run the ARIAL pipeline.
    """
    load_dotenv()

    parser = argparse.ArgumentParser(description="ARIAL Framework for Document VQA")
    parser.add_argument("--document_path", type=str, default="documents/sample_document.pdf", help="Path to the document image or PDF.")
    parser.add_argument("--question", type=str, required=True, help="Question to ask about the document.")
    
    args = parser.parse_args()

    if not os.path.exists(args.document_path):
        print(f"Error: Document not found at {args.document_path}")
        return

    # 1. Initialize all modules
    print("Initializing ARIAL components...")
    ocr_processor = OCRProcessor()
    text_retriever = TextRetriever()
    qa_module = QAModule()
    spatial_grounder = SpatialGrounder()
    compute_module = ComputeModule()
    print("Components initialized.")

    tools = {
        "RunOCR": ocr_processor,
        "FindText": text_retriever,
        "AskQA": qa_module,
        "GroundAnswer": spatial_grounder,
        "Compute": compute_module
    }

    # 2. Instantiate the Planner Agent with its tools
    print("Initializing Planner Agent...")
    try:
        agent = PlannerAgent(tools)
        print("Agent initialized.")
    except Exception as e:
        print(f"Failed to initialize the Planner Agent: {e}")
        print("Please ensure you have set your HUGGING_FACE_HUB_TOKEN in the .env file and have accepted the license for the agent model on the Hugging Face Hub.")
        return

    # 3. Execute a query
    print(f"\nExecuting query for document: '{args.document_path}'")
    print(f"Question: '{args.question}'")
    
    final_answer, bounding_box = agent.run(args.document_path, args.question)

    # 4. Print the result
    print("\n--- Execution Finished ---")
    if final_answer:
        print(f"Answer: {final_answer}")
        print(f"Bounding Box: {bounding_box}")
    else:
        print("Could not determine the answer.")

if __name__ == "__main__":
    main()