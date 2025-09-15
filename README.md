# ARIAL Framework Implementation

This project is a Python implementation of the ARIAL (Agentic Reasoning for Interpretable Answer Localization) framework for Document Visual Question Answering (DocVQA).

## Features

- **End-to-End DocVQA**: Takes a document image and a question, and returns a textual answer with its bounding box.
- **Modular Architecture**: Built with separate components for OCR, text retrieval, question answering, spatial grounding, and computation.
- **Agentic Reasoning**: An LLM-powered planner agent orchestrates the workflow, deciding which tool to use at each step.
- **Powered by Hugging Face**: Leverages pre-trained models from the Hugging Face Hub for its core functionalities.

## Project Structure

```
.
├── README.md
├── main.py                 # Main script to run the ARIAL pipeline
├── planner_agent.py        # Contains the PlannerAgent class
├── tools.py                # Implementation of all tools (OCR, QA, etc.)
├── requirements.txt        # Python dependencies
└── .env.example            # Example for environment variables
```

## Getting Started

### Prerequisites

- Python 3.9+
- An NVIDIA GPU is highly recommended for running the models.

### Installation

1.  **Clone the repository:**
    ```bash
    git clone <repository_url>
    cd <repository_name>
    ```

2.  **Create a virtual environment:**
    ```bash
    python -m venv venv
    source venv/bin/activate
    ```

3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Set up environment variables:**

    Create a `.env` file by copying the example:
    ```bash
    cp .env.example .env
    ```

    Open the `.env` file and add your Hugging Face Hub token. This is required to download and use certain models like Llama 3.
    ```
    HUGGING_FACE_HUB_TOKEN="your_hf_token_here"
    ```

### Running the Application

Place your document images in a directory (e.g., `documents/`).

Run the `main.py` script with the path to your document and your question:

```bash
python main.py --document_path "path/to/your/document.png" --question "What is the total amount due?"
```

The script will download the necessary models on the first run, which might take some time.

## How It Works

The ARIAL framework operates in three stages:

1.  **Stage 1: Input Processing**: The `OCRProcessor` uses `doctr` to extract text segments and their bounding boxes from the input document.
2.  **Stage 2: Agentic Reasoning**: The `PlannerAgent`, powered by a large language model, uses a set of tools to answer the question. It follows a "Sense-Think-Act" loop to decide which tool to call next. The available tools are:
    -   `TextRetriever`: Finds relevant text segments using semantic and lexical search.
    -   `QAModule`: Generates a textual answer from the context.
    -   `SpatialGrounder`: Locates the answer's bounding box on the document.
    -   `ComputeModule`: Performs numerical calculations.
3.  **Stage 3: Output Generation**: The final answer and its bounding box are returned.
