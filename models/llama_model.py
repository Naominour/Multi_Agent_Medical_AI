import torch
import gc
from llama31 import Llama
from transformers import AutoTokenizer



def clear_gpu_memory():
    gc.collect()
    torch.cuda.empty_cache()

class LlamaModel:
    def __init__(self, ckpt_dir, tokenizer_path, max_seq_len=2048, max_batch_size=1, device="cuda"):

        # Initialises the LLaMA model with memory optimisations.
        
        self.device = device if torch.cuda.is_available() else "cpu"
        self.ckpt_dir = ckpt_dir
        self.tokenizer_path = tokenizer_path
        self.max_seq_len = max_seq_len
        self.max_batch_size = max_batch_size

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_path, use_fast=True)

        # Load model
        self.model = Llama.build(
            ckpt_dir=self.ckpt_dir,
            tokenizer_path=self.tokenizer_path,
            max_seq_len=self.max_seq_len,
            max_batch_size=self.max_batch_size,
            flash=False
        )

        self.model.to(self.device)
        self.model.eval() 

    def load_checkpoint(self, checkpoint_path):
        
        # Loads a trained checkpoint into the LLaMA model.
        try:
            clear_gpu_memory()
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            self.model.load_state_dict(checkpoint["model_state_dict"])
            print(f"✅ Successfully loaded LLaMA checkpoint from {checkpoint_path}")
        except Exception as e:
            print(f"❌ Error loading checkpoint: {str(e)}")

    def format_few_shot_prompt(self, question):
        """
        Formats a few-shot learning prompt to improve response quality.
        """
        few_shot_examples = """
        Example 1:
        Q: What are the symptoms of diabetes?
        A: Diabetes symptoms include frequent urination, excessive thirst, fatigue, and blurred vision.

        Example 2:
        Q: How is hypertension treated?
        A: Hypertension is treated with lifestyle changes, medication, and regular monitoring.

        Example 3:
        Q: What is the treatment for pneumonia?
        A: Pneumonia treatment depends on the cause but often includes antibiotics, rest, and hydration.

        User Question:""".strip()
        return f"{few_shot_examples}\nQ: {question}\nA:"

    def generate_response(self, input_text, max_gen_len=300, temperature=0.6, top_p=0.9, sample_rng=None, formatted=False):
        
        # Generates a response from LLaMA 
        try:
            if not formatted:
                prompt = self.format_few_shot_prompt(input_text)
            else:
                prompt = input_text

            if sample_rng is None:
                sample_rng = torch.Generator(device=self.device)

            clear_gpu_memory()
            with torch.cuda.amp.autocast():
                output = self.model.text_completion(
                    [prompt],
                    sample_rng=sample_rng,
                    max_gen_len=max_gen_len,
                    temperature=temperature,
                    top_p=top_p,
                    echo=False
                )[0]

            full_text = output.get('generation', '')
            cleaned_text = full_text.split("User Question:")[0].strip()
            return {'generation': cleaned_text}

        except Exception as e:
            print(f"❌ Error in generate_response: {str(e)}")
            return {'generation': 'Error generating response'}

    def cleanup(self):

        # Moves the model to CPU and clears GPU memory
        self.model.to("cpu")
        clear_gpu_memory()
