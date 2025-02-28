import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import gc


def clear_gpu_memory():
    gc.collect()
    torch.cuda.empty_cache()

class TransformersModel:
    def __init__(self, model_name="facebook/opt-1.3b", device="cuda"):

        # Initialises a generic transformer model for language modeling tasks.
        self.device = device if torch.cuda.is_available() else "cpu"
        self.model_name = model_name

        # Load tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
            low_cpu_mem_usage=True
        )

        self.model.to(self.device)
        self.model.eval()

    def compute_perplexity(self, text):

        # Computes uncertainty score of a the model response.
        try:
            if not text.strip():
                print("[WARNING] Empty response passed for uncertainty calculation.")
                return float("inf")

            inputs = self.tokenizer(text, return_tensors="pt").to(self.device)

            with torch.no_grad():
                outputs = self.model(**inputs, labels=inputs.input_ids)

            if outputs.loss is None or torch.isnan(outputs.loss):
                print("[WARNING] Perplexity calculation failed. Returning high uncertainty.")
                return float("inf")

            loss = outputs.loss.item()
            return torch.exp(torch.tensor(loss)).item()  # Convert to perplexity score

        except RuntimeError as e:
            if "out of memory" in str(e):
                clear_gpu_memory()
                return "Error: GPU memory exceeded. Try reducing input length or parameters."
            raise e
        except Exception as e:
            print(f"❌ Error in compute_perplexity: {str(e)}")
            return float("inf")

    def cleanup(self):
        self.model.to("cpu")
        clear_gpu_memory()
