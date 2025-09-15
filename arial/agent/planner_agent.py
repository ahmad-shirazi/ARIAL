import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import re
import json

class PlannerAgent:
    """
    The central orchestrator that uses an LLM to decide which tool to call.
    """
    def __init__(self, tools: dict, model_name='meta-llama/Llama-3-8B-Instruct', device=None):
        if device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device
            
        self.tools = tools
        self.history = []

        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16
        )

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map="auto",
            quantization_config=quantization_config,
        )

    def _build_prompt(self, question: str, document_path: str) -> str:
        """
        Builds the prompt for the agent LLM with few-shot examples.
        """
        tool_descriptions = """
- RunOCR(document_path: str) -> list[dict]: Runs OCR on the document to extract text and bounding boxes.
- FindText(query: str) -> list[dict]: Finds the most relevant text segments from the OCR results based on a query. The OCR results are managed internally.
- AskQA(question: str, context: str) -> str: Answers a question based on the provided context.
- GroundAnswer(answer_text: str) -> list[float]: Finds the bounding box of the answer text in the document. The OCR results are managed internally.
- Compute(operation: str, values: list) -> float: Performs a calculation (e.g., 'sum', 'average').
"""
        
        few_shot_example = """
Example reasoning trace:
Question: "What is the total amount due?"
Thought: I need to first extract text from the document, then find relevant segments about amounts or totals, then answer the question and locate the answer on the document.
Action: {"RunOCR": {"document_path": "invoice.pdf"}}
Observation: [{"text": "Invoice #12345", "box": [10, 20, 200, 40]}, {"text": "Amount: $150.00", "box": [10, 100, 180, 120]}, {"text": "Tax: $15.00", "box": [10, 130, 150, 150]}, {"text": "Total: $165.00", "box": [10, 160, 180, 180]}]
Thought: Now I need to find text segments related to "total amount" to focus on the relevant parts.
Action: {"FindText": {"query": "total amount due"}}
Observation: [{"text": "Total: $165.00", "box": [10, 160, 180, 180]}, {"text": "Amount: $150.00", "box": [10, 100, 180, 120]}]
Thought: I have the relevant context. Now I can answer the question using the QA module.
Action: {"AskQA": {"question": "What is the total amount due?", "context": "Total: $165.00 Amount: $150.00"}}
Observation: "$165.00"
Thought: I have the answer. Now I need to ground it in the document to find its bounding box.
Action: {"GroundAnswer": {"answer_text": "$165.00"}}
Observation: [10, 160, 180, 180]
Thought: Perfect! I have both the answer and its location.
Final Answer: {"answer": "$165.00", "bounding_box": [10, 160, 180, 180]}
"""
        
        history_str = "\n".join(self.history)
        
        prompt = f"""
You are a helpful assistant that answers questions based on a document. You have access to the following tools:
{tool_descriptions}

{few_shot_example}

Your task is to answer the following question: "{question}" for the document at "{document_path}".

Follow this process:
1. ALWAYS start by running OCR to extract text from the document.
2. Think about what you need to do to answer the question. Your thought process should be in a 'Thought:' block.
3. Based on your thought, choose one of the available tools and specify the arguments in a single-line JSON 'Action:' block.
4. Observe the result of the tool, which will be provided to you.
5. Repeat until you have the final answer and its bounding box.
6. Once you have the final answer and its bounding box, respond with a 'Final Answer:' block containing a JSON object with 'answer' and 'bounding_box'.

Here is the history of your work so far:
{history_str}
"""
        return prompt

    def _parse_action(self, llm_output: str) -> (str, dict):
        """
        Parses the LLM output to get the tool name and arguments.
        """
        action_match = re.search(r"Action:\s*({.*})", llm_output)
        if action_match:
            try:
                action_json = action_match.group(1)
                action_data = json.loads(action_json)
                tool_name = list(action_data.keys())[0]
                args = action_data[tool_name]
                return tool_name, args
            except (json.JSONDecodeError, IndexError, TypeError) as e:
                print(f"Error parsing action: {e}")
                return None, None
        return None, None
        
    def _parse_final_answer(self, llm_output: str) -> dict:
        """
        Parses the LLM output for the final answer.
        """
        answer_match = re.search(r"Final Answer:\s*({.*})", llm_output, re.DOTALL)
        if answer_match:
            try:
                answer_json = answer_match.group(1)
                return json.loads(answer_json)
            except json.JSONDecodeError as e:
                print(f"Error parsing final answer: {e}")
                return None
        return None

    def run(self, document_path: str, question: str):
        """
        Executes the agentic loop to answer the question.
        """
        self.history = []
        state = {} # To hold OCR results, retrieved text, etc.
        
        # Automatically run OCR as the first step
        print("\n--- Auto-initializing with OCR ---")
        try:
            ocr_tool = self.tools['RunOCR']
            ocr_results = ocr_tool.run(document_path)
            state['ocr_results'] = ocr_results
            self.history.append(f"Action: {{'RunOCR': {{'document_path': '{document_path}'}}}}")
            self.history.append(f"Observation: Found {len(ocr_results)} text segments")
            print(f"OCR completed. Found {len(ocr_results)} text segments.")
        except Exception as e:
            print(f"Error during OCR initialization: {e}")
            return None, None
        
        max_turns = 10
        for i in range(max_turns):
            print(f"\n--- Turn {i+1} ---")
            
            prompt = self._build_prompt(question, document_path)
            
            input_ids = self.tokenizer.encode(prompt, return_tensors="pt").to(self.device)
            outputs = self.model.generate(
                input_ids,
                max_new_tokens=512,
                eos_token_id=self.tokenizer.eos_token_id,
                pad_token_id=self.tokenizer.eos_token_id
            )
            
            llm_response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Get only the newly generated part
            llm_output = llm_response[len(prompt):].strip()
            
            self.history.append(f"Thought: {llm_output}")
            print(f"Agent Thought: {llm_output}")
            
            # Check for final answer
            final_answer = self._parse_final_answer(llm_output)
            if final_answer:
                print("\n--- Final Answer Found ---")
                return final_answer.get('answer'), final_answer.get('bounding_box')

            # Parse and execute action
            tool_name, args = self._parse_action(llm_output)
            
            if tool_name and tool_name in self.tools:
                print(f"Action: {tool_name}({args})")
                tool = self.tools[tool_name]
                
                try:
                    # Special handling for tools that need internal state
                    if tool_name == 'FindText':
                        if 'ocr_results' not in state:
                            raise ValueError("RunOCR must be called before FindText.")
                        result = tool.find(query=args['query'], ocr_results=state['ocr_results'])
                        state['retrieved_text'] = result
                    elif tool_name == 'AskQA':
                        if 'retrieved_text' not in state:
                            # If no retrieval was done, use all OCR text as context
                            context_text = " ".join([res['text'] for res in state.get('ocr_results', [])])
                        else:
                            context_text = " ".join([res['text'] for res in state['retrieved_text']])
                        result = tool.ask(question=args.get('question', question), context=context_text)
                        state['answer_text'] = result
                    elif tool_name == 'GroundAnswer':
                        if 'ocr_results' not in state or 'answer_text' not in state:
                            raise ValueError("OCR and QA must be run before grounding.")
                        result = tool.ground(
                            answer_text=state['answer_text'], 
                            ocr_results=state['ocr_results'],
                            question=question
                        )
                    else:
                        result = tool.run(**args)

                    if tool_name == 'RunOCR':
                        state['ocr_results'] = result
                        
                    observation = f"Observation: {json.dumps(result, indent=2)}"
                    self.history.append(f"Action: {json.dumps({tool_name: args})}")
                    self.history.append(observation)
                    print(observation)

                except Exception as e:
                    error_message = f"Error executing tool {tool_name}: {e}"
                    self.history.append(f"Error: {error_message}")
                    print(error_message)
            else:
                message = "No valid action found in the response."
                self.history.append(f"System: {message}")
                print(message)

        print("\n--- Max turns reached ---")
        return None, None
