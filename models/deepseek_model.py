import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import os
import gc


def clear_gpu_memory():
    gc.collect()
    torch.cuda.empty_cache()

class DeepSeekModel:
    def __init__(self, model_path, device="cuda"):

        # Initialises the DeepSeek model
        self.device = device if torch.cuda.is_available() else "cpu"
        self.model_path = model_path

        # Load tokenizer and model 
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path, use_fast=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            device_map="cpu", 
            trust_remote_code=True,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True
        )
        self.model.eval()  

    def refine_response(self, question, response_text, max_new_tokens=500, temperature=0.7, top_p=0.9):

        # Refines a given response using DeepSeek 
        try:
            if not response_text.strip():
                return "Error: No valid response provided for refinement."

            prompt = f"""Refine the following medical response to be more detailed, structured, and evidence-based:

### Original Answer:
{response_text}

### Refined Answer:
"""
            self.model.to(self.device)

            # Tokenize input prompt
            inputs = self.tokenizer(
                prompt,
                return_tensors="pt",
                padding="max_length",
                truncation=True,
                max_length=2048
            ).to(self.device)

            with torch.amp.autocast("cuda"):
                outputs = self.model.generate(
                    input_ids=inputs["input_ids"],
                    attention_mask=inputs["attention_mask"],
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id
                )

            # Decode and return refined text
            refined_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True).strip()

            # Check if the model generated a meaningful response
            if not refined_text or refined_text.startswith("Refine the following") or "<think>" in refined_text:
                return "Error: DeepSeek failed to refine the response."

            return refined_text

        except RuntimeError as e:
            if "out of memory" in str(e):
                clear_gpu_memory()
                return "Error: GPU memory exceeded. Try reducing input length or parameters."
            raise e
        except Exception as e:
            print(f"Error in DeepSeek refinement: {str(e)}")
            return "Error refining response."

    def cleanup(self):
        self.model.to("cpu")
        clear_gpu_memory()
