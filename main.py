from random import sample
from llama31 import Llama
from sentence_transformers import SentenceTransformer, util
from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoModelForCausalLM, AutoTokenizer
from contextlib import contextmanager

import torch
import torch.nn.functional as F
import numpy as np
import requests
import xml.etree.ElementTree as ET
import os
import gc

# Memory management configuration
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:512,expandable_segments:True'
os.environ["WANDB_DISABLED"] = "true"

def clear_gpu_memory():
    gc.collect()
    torch.cuda.empty_cache()

# Configuration
pubmed_api_key = 'b8f335b181c463ae3f16b3678edcda622008'
ckpt_dir = "/content/drive/MyDrive/MedAI/models/Llama3.1-8B"
tokenizer_path = "/content/drive/MyDrive/MedAI/models/Llama3.1-8B/tokenizer.model"
checkpoint_path = "/content/drive/MyDrive/test/llama/output_data/trained_model.pth"
deepseek_model_path = "/content/drive/MyDrive/MedAI/DeepSeek-model/merged_model"

# Initialize sentence transformer
st_model = SentenceTransformer('all-MiniLM-L6-v2', device='cuda')

# Initialize LLaMA
llama = Llama.build(
    ckpt_dir=ckpt_dir,
    tokenizer_path=tokenizer_path,
    max_seq_len=2048,
    max_batch_size=1,
    flash=False
)

# Load LLaMA checkpoint with memory optimization
clear_gpu_memory()
checkpoint = torch.load(checkpoint_path, map_location="cuda", weights_only=True)
llama.model.load_state_dict(checkpoint["model_state_dict"])
llama.model.to("cuda")
llama.model.eval()

@contextmanager
def use_gpu(model):
    model.to("cuda")
    try:
        yield
    finally:
        model.to("cpu")
        clear_gpu_memory()

def format_few_shot_prompt(question):
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

def generate_llama_response(input_text, max_gen_len=300, temperature=0.6, top_p=0.9, sample_rng=None, formatted=False):
    """
    Generates a response from LLaMA with memory optimization.
    """
    try:
        if not formatted:
            prompt = format_few_shot_prompt(input_text)
        else:
            prompt = input_text

        if sample_rng is None:
            sample_rng = torch.Generator(device='cuda')
            
        with use_gpu(llama.model):
            with torch.cuda.amp.autocast():  # Enable automatic mixed precision
                output = llama.text_completion(
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
        print(f"Error in generate_llama_response: {str(e)}")
        return {'generation': 'Error generating response'}

def compute_relevance_score(question, response):
    try:
        actual_response = response.get('generation', '')
        with torch.cuda.amp.autocast():
            q_embeddings = st_model.encode([question], device='cuda')
            r_embeddings = st_model.encode([actual_response], device='cuda')
            relevance_score = cosine_similarity(q_embeddings, r_embeddings)[0][0]
        return relevance_score
    except Exception as e:
        print(f"Error in compute_relevance_score: {str(e)}")
        return 0.0

def mc_dropout_samples(question, prompt, num_samples=5, max_gen_len=300, temperature=0.6, top_p=0.9):
    samples = []
    try:
        llama.model.train() 
        sample_rng = torch.Generator(device='cuda')
        
        for i in range(num_samples):
            response = generate_llama_response(
                prompt,
                max_gen_len=max_gen_len,
                temperature=temperature,
                top_p=top_p,
                sample_rng=sample_rng,
                formatted=True 
            )
            samples.append(response.get('generation', ''))
            clear_gpu_memory()
        
        llama.model.eval() 
        return samples
    except Exception as e:
        print(f"Error in mc_dropout_samples: {str(e)}")
        return ['Error generating samples'] * num_samples

def compute_mc_uncertainty_score(samples):
    try:
        with torch.cuda.amp.autocast():
            samples_embedding = st_model.encode(samples)
            pairwise_sim = []
            num_samples = len(samples_embedding)
            
            for i in range(num_samples):
                for j in range(i + 1, num_samples):
                    sim = cosine_similarity([samples_embedding[i]], [samples_embedding[j]])[0][0]
                    pairwise_sim.append(sim)
            
            avg_similarity = np.mean(pairwise_sim)
            std_similarity = np.std(pairwise_sim)
            return avg_similarity, std_similarity
    except Exception as e:
        print(f"Error in compute_mc_uncertainty_score: {str(e)}")
        return 0.0, 0.0

def fetch_pubmed_info(question, api_key, retmax=3):
    try:
        esearch_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
        esearch_params = {
            "db": "pubmed",
            "term": question,
            "retmode": "json",
            "retmax": retmax,
            "api_key": api_key
        }
        esearch_response = requests.get(esearch_url, params=esearch_params)
        esearch_data = esearch_response.json()
        id_list = esearch_data.get("esearchresult", {}).get("idlist", [])
        
        if not id_list:
            return ["No relevant PubMed articles found."]
        
        efetch_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
        efetch_params = {
            "db": "pubmed",
            "id": ",".join(id_list),
            "retmode": "xml",
            "api_key": api_key
        }
        efetch_response = requests.get(efetch_url, params=efetch_params)
        root = ET.fromstring(efetch_response.content)
        
        articles_info = []
        for article in root.findall(".//PubmedArticle"):
            title_elem = article.find(".//ArticleTitle")
            title = title_elem.text if title_elem is not None else "No title available"
            abstract_texts = []
            for abstract in article.findall(".//Abstract/AbstractText"):
                if abstract.text:
                    abstract_texts.append(abstract.text)
            abstract = " ".join(abstract_texts) if abstract_texts else "No abstract available."
            info = f"Title: {title}\nAbstract: {abstract}"
            articles_info.append(info)
        
        return articles_info
    except Exception as e:
        print(f"Error in fetch_pubmed_info: {str(e)}")
        return ["Error fetching PubMed information"]

def generate_augmented_prompt(question, response, api_key):
    try:
        pubmed_references = fetch_pubmed_info(question, api_key)
        augmented_prompt = f"""
        User Question: {question}

        Relevant References:
        """
        for idx, ref in enumerate(pubmed_references, start=1):
            augmented_prompt += f"[Reference {idx}]: {ref}\n\n"

        augmented_prompt += "Based on the above references, provide a concise and evidence-based answer:\nAnswer:"
        final_answer = generate_llama_response(augmented_prompt, formatted=True)
        return final_answer
    except Exception as e:
        print(f"Error in generate_augmented_prompt: {str(e)}")
        return {'generation': 'Error generating augmented prompt'}

def compute_uncertainty_augmented_answer(response_text):
    """
    Computes an uncertainty score (perplexity) using Facebook's OPT-1.3B model.
    """
    try:
        if not response_text.strip(): 
            print("[WARNING] Empty response passed to uncertainty calculation.")
            return float("inf")

        tokenizer = AutoTokenizer.from_pretrained("facebook/opt-1.3b")
        model = AutoModelForCausalLM.from_pretrained("facebook/opt-1.3b")
        
        with use_gpu(model):
            with torch.cuda.amp.autocast():
                inputs = tokenizer(response_text, return_tensors="pt").to("cuda")
                with torch.no_grad():
                    outputs = model(**inputs, labels=inputs.input_ids)

        if outputs.loss is None or torch.isnan(outputs.loss):
            print("[WARNING] Uncertainty calculation failed. Returning high uncertainty.")
            return float("inf")  

        loss = outputs.loss.item()
        uncertainty_score = np.exp(loss)
        if np.isnan(uncertainty_score) or uncertainty_score > 1e6: 
            print("[WARNING] Uncertainty calculation produced NaN or overflow.")
            return float("inf")
        return uncertainty_score
    except Exception as e:
        print(f"Error in compute_uncertainty_augmented_answer: {str(e)}")
        return float("inf")

# Loading DeepSeek Model with optimizations
clear_gpu_memory()
deepseek_tokenizer = AutoTokenizer.from_pretrained(deepseek_model_path, use_fast=True)
deepseek_model = AutoModelForCausalLM.from_pretrained(
    deepseek_model_path,
    device_map="cpu",
    trust_remote_code=True,
    torch_dtype=torch.float16,
    low_cpu_mem_usage=True
)
deepseek_model.eval()

def refine_with_deepseek(question, response, max_new_tokens=2000, temperature=0.7, top_p=0.9):
    try:
        # Extract the response text
        if not isinstance(response, dict):
            response_text = str(response)
        else:
            response_text = response.get('generation', '')
            
        prompt = f"""User Question: {question}
                     Final Response: {response_text}

                     Please refine the above answer to be more detailed, evidence-based, and well-structured. Provide specific improvements or additional insights.
                     Refined Answer:"""
        
        with use_gpu(deepseek_model):
            # Tokenize the prompt
            inputs = deepseek_tokenizer(prompt, return_tensors="pt", padding=True, truncation=True)
            inputs = {k: v.to("cuda") for k, v in inputs.items()}
            
            # Generate new tokens using max_new_tokens
            with torch.cuda.amp.autocast():
                outputs = deepseek_model.generate(
                    input_ids=inputs["input_ids"],
                    attention_mask=inputs["attention_mask"],
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    do_sample=True,
                    pad_token_id=deepseek_tokenizer.eos_token_id
                )
            refined_text = deepseek_tokenizer.decode(outputs[0], skip_special_tokens=True)
            return refined_text
            
    except RuntimeError as e:
        if "out of memory" in str(e):
            clear_gpu_memory()
            return "Error: GPU memory exceeded. Try reducing input length or model parameters."
        raise e
    except Exception as e:
        print(f"Error in refine_with_deepseek: {str(e)}")
        return "Error refining response"


# Agent Classes
class BaseAgent:
    def __init__(self, name):
        self.name = name
        self.state = {}  

    def process(self, input_data):
        raise NotImplementedError("Each agent must implement its own process method.")

    def communicate(self, message):
        print(f"{self.name} received: {message}")

class EvidenceRetrievalAgent(BaseAgent):
    def __init__(self, name="EvidenceRetrievalAgent", api_key=None):
        super().__init__(name)
        self.api_key = api_key

    def process(self, query):
        """
        Retrieve evidence and return it as formatted text.
        """
        try:
            evidence = fetch_pubmed_info(query, self.api_key)
            evidence_text = "\n\n".join([f"[Reference {i+1}]: {ref}" for i, ref in enumerate(evidence)])
            return evidence_text
        except Exception as e:
            print(f"Error in EvidenceRetrievalAgent: {str(e)}")
            return "Error retrieving evidence"

class ClinicalReasoningAgent(BaseAgent):
    def __init__(self, name="ClinicalReasoningAgent"):
        super().__init__(name)
    
    def process(self, question, evidence=None):
        """
        Generate an answer using chain-of-thought (CoT) reasoning.
        """
        try:
            prompt = f"Question: {question}\n"
            if evidence:
                prompt += f"Evidence:\n{evidence}\n"
            prompt += """Let's reason through the problem step by step:
            Step 1: Analyze the question and evidence
            Step 2: Identify key medical concepts
            Step 3: Form a structured response
            
            Based on the above steps, here is the final answer:\n"""
            
            response = generate_llama_response(prompt, formatted=True)
            
            if not response or not response.get('generation'):
                return {'generation': 'No valid response generated. Please try again.'}
                
            return response
            
        except Exception as e:
            print(f"Error in ClinicalReasoningAgent: {str(e)}")
            return {'generation': 'An error occurred while processing the request.'}

class AgentOrchestrator:
    def __init__(self, reasoning_agent, retrieval_agent):
        self.reasoning_agent = reasoning_agent
        self.retrieval_agent = retrieval_agent

    def handle_query(self, question):
        try:
            # Get evidence
            evidence = self.retrieval_agent.process(question)
            if not evidence or evidence == "Error retrieving evidence":
                print("Warning: Failed to retrieve evidence")
                evidence = "No evidence available"

            # Get reasoning response
            reasoning_response = self.retrieval_agent.process(question)  # Use evidence if needed
            reasoning_response = self.reasoning_agent.process(question, evidence)
            if not reasoning_response or not reasoning_response.get('generation'):
                print("Warning: Failed to generate reasoning response")
                return {'generation': 'Failed to generate a response'}

            # Compute scores
            relevance_score = compute_relevance_score(question, reasoning_response)
            uncertainty_score = compute_uncertainty_augmented_answer(reasoning_response['generation'])

            # If response is uncertain, try again
            if relevance_score < 0.5 or uncertainty_score > 10:
                print("🔄 Uncertain response, requesting better response")
                evidence = self.retrieval_agent.process(question)  
                reasoning_response = self.reasoning_agent.process(question, evidence)

            return reasoning_response

        except Exception as e:
            print(f"Error in AgentOrchestrator: {str(e)}")
            return {'generation': 'Error in processing query'}

if __name__ == "__main__":
    try:
        # Initial memory cleanup
        clear_gpu_memory()
        
        user_question = "What are the early signs of Alzheimer's disease?"
        
        # Generate base response
        print("\nGenerating base LLaMA response...")
        base_response = generate_llama_response(user_question)
        print("Base Response:", base_response)
        
        # Compute relevance
        print("\nComputing relevance score...")
        relevance_score = compute_relevance_score(user_question, base_response)
        print(f"Relevance Score: {relevance_score:.3f}")
        
        # Generate MC dropout samples
        print("\nGenerating multiple responses using MC dropout...")
        prompt = format_few_shot_prompt(user_question)
        mc_samples = mc_dropout_samples(user_question, prompt, num_samples=5)
        for idx, sample in enumerate(mc_samples, 1):
            print(f"Sample {idx}: {sample}")
        
        # Compute uncertainty metrics
        print("\nComputing uncertainty metrics...")
        avg_sim, std_sim = compute_mc_uncertainty_score(mc_samples)
        print(f"Average pairwise similarity: {avg_sim:.3f}")
        print(f"Standard deviation of similarities: {std_sim:.3f}")
        
        # Generate augmented response
        print("\nGenerating augmented response...")
        clear_gpu_memory()
        augmented_answer = generate_augmented_prompt(user_question, base_response, pubmed_api_key)
        print("Augmented Response:", augmented_answer.get('generation', ''))
        
        # Compute uncertainty for augmented response
        print("\nComputing uncertainty for augmented response...")
        uncertainty_entropy = compute_uncertainty_augmented_answer(augmented_answer.get('generation', ''))
        print(f"Perplexity (Uncertainty): {uncertainty_entropy:.3f}")
        
        # Use agents via orchestrator
        print("\n--- Using Agents via the Orchestrator ---")
        clear_gpu_memory()
        clinical_agent = ClinicalReasoningAgent()
        evidence_agent = EvidenceRetrievalAgent(api_key=pubmed_api_key)
        orchestrator = AgentOrchestrator(clinical_agent, evidence_agent)
        
        # Get final response
        print("\nGetting final response from agents...")
        final_response = orchestrator.handle_query(user_question)
        print("Final Response:", final_response)
        
        # Refine with DeepSeek if we have a valid response
        if final_response and final_response.get('generation'):
            print("\nRefining with DeepSeek...")
            clear_gpu_memory()
            deepseek_response = refine_with_deepseek(user_question, final_response)
            print("DeepSeek Response:", deepseek_response)
        else:
            print("\nSkipping DeepSeek refinement due to invalid response")
            
        # Final memory cleanup
        clear_gpu_memory()
        
    except Exception as e:
        clear_gpu_memory()
