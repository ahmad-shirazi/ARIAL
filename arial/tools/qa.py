import torch
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM

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
            torch_dtype=torch.bfloat16
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
        
        answer_marker = "Answer:"
        answer_pos = generated_text.find(answer_marker)
        if answer_pos != -1:
            return generated_text[answer_pos + len(answer_marker):].strip()
        
        return generated_text.strip()
