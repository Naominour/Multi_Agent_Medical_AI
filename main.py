from random import sample
from llama31 import Llama
from sentence_transformers import SentenceTransformer, util
from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoModelForCausalLM, AutoTokenizer
from contextlib import contextmanager
from bias_detection import detect_lexical_bias, analyse_sentiment
from datetime import datetime

import torch
import torch.nn.functional as F
import numpy as np
import requests
import xml.etree.ElementTree as ET
import os
import gc
import json 

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

def save_result_to_json(results):
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    filename = f"/content/drive/MyDrive/MedAI/output_{timestamp}.json"

    with open(filename, "w") as json_file:
         json.dump(results, json_file, indent=4)
    print(f"Results saved to {filename}")


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

def refine_with_deepseek(question, response, max_new_tokens=500, temperature=0.7, top_p=0.9):
    try:
        # Extract response text
        response_text = response.get('generation', '') if isinstance(response, dict) else str(response)

        if not response_text.strip():
            return "Error: No valid response provided for refinement."

        # Improved prompt formatting
        prompt = f"""Refine the following medical response to be more detailed, structured, and evidence-based:

### Original Answer:
{response_text}

### Refined Answer:
"""

        with use_gpu(deepseek_model):
            # Check available GPU memory before running DeepSeek
            if torch.cuda.is_available():
                free_mem = torch.cuda.memory_allocated() / (1024**3)  # Convert to GB
                if free_mem > 10:
                    print("⚠ Warning: GPU memory is running low. DeepSeek may fail.")
                    return "Error: GPU memory exceeded. Try reducing input length or parameters."

            
            inputs = deepseek_tokenizer(prompt, return_tensors="pt", padding="max_length", truncation=True, max_length=2048)
            inputs = {k: v.to("cuda") for k, v in inputs.items()}

            with torch.amp.autocast("cuda"):
                outputs = deepseek_model.generate(
                    input_ids=inputs["input_ids"],
                    attention_mask=inputs["attention_mask"],
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    do_sample=True,
                    pad_token_id=deepseek_tokenizer.eos_token_id
                )

            refined_text = deepseek_tokenizer.decode(outputs[0], skip_special_tokens=True).strip()

            # Check if DeepSeek failed
            if not refined_text or refined_text.startswith("Refine the following") or "<think>" in refined_text:
                print("⚠ DeepSeek did not generate a valid response. Retrying...")
                return "Error: DeepSeek failed to refine the response."

            # Run bias checks
            lexical_bias = detect_lexical_bias(refined_text)
            sentiment_bias = analyse_sentiment(refined_text)

            if lexical_bias or "Potential" in sentiment_bias:
                print("⚠ Bias/Misinformation detected! Returning original response.")
                return response_text  # Return original response if bias check fails

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

class HumanExpertReviewAgent(BaseAgent):
    def __init__(self, name="HumanExpertReviewAgent", threshold_uncertainty=10.0, api_key=None):
        super().__init__(name)
        self.threshold_uncertainty = threshold_uncertainty
        self.api_key = api_key

    def process(self, question, response, api_key):
        current_answer = response
        while True:
            print("\n--- Human Expert Review ---")
            print("Current Answer:")
            print(current_answer)
            expert_available = input("\nIs a human expert available to revise the final answer? (y/n): ").strip().lower() == 'y'
            if not expert_available:
                # If no expert is available, just return the current answer.
                return current_answer

            expert_approval = input("Do you approve the LLM's final answer? (y/n): ").strip().lower()
            if expert_approval == "y":
                print("Human expert approved the answer.")
                return current_answer
            else:
                expert_comment = input("Human expert, please provide your comment (e.g., what's missing or incorrect):\n")
                # Build a new prompt that incorporates the expert's feedback.
                new_prompt = f"""Refine the following medical response based on the human expert's feedback.

### Original Answer:
{current_answer}

### Expert Comment:
{expert_comment}

### Question:
{question}

### Revised Answer:"""
                # Refine again with DeepSeek.
                new_response = refine_with_deepseek(question, {"generation": new_prompt})
                # Optionally, fetch additional evidence if needed.
                pubmed_data = generate_augmented_prompt(question, new_response, self.api_key)
                # Compute uncertainty on the augmented response.
                new_uncertainty = compute_uncertainty_augmented_answer(pubmed_data.get('generation', new_response))
                print(f"New refined answer uncertainty (perplexity): {new_uncertainty:.3f}")
                # If uncertainty is acceptable, update and ask for expert approval again.
                if new_uncertainty < self.threshold_uncertainty:
                    current_answer = pubmed_data.get('generation', new_response)
                    print("The refined answer has acceptable uncertainty. Please review again.")
                else:
                    print("The refined answer uncertainty is still high. Trying again...")
                    current_answer = new_response

class AgentOrchestrator(BaseAgent):
    def __init__(self, reasoning_agent, retrieval_agent, human_agent=None):
        self.reasoning_agent = reasoning_agent
        self.retrieval_agent = retrieval_agent
        self.human_agent = human_agent

    def handle_query(self, question, expert_available=False):
        try:
            evidence = self.retrieval_agent.process(question)
            if not evidence or evidence == "Error retrieving evidence":
                print("Warning: Failed to retrieve evidence")
                evidence = "No evidence available"

            # Get reasoning response
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

            final_response = reasoning_response.get('generation', '')

            lexical_bias = detect_lexical_bias(final_response)
            sentiment_bias = analyse_sentiment(final_response)

            if lexical_bias or "Potential" in sentiment_bias:
                print("⚠ Bias detected in the refined response. Expert review is recommended.")
            
            # If a human expert is available, use the human expert agent.
            if expert_available and self.human_agent:
                print("\n--- Initiating Human Expert Review ---")
                final_response = self.human_agent.process(question, final_response)

            return {'generation': final_response}
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
        
        results = {
            "User Question": user_question,
            "Base Response": base_response,
            "Relevance Score": float(relevance_score),
            "Pairwise similarity": float(avg_sim),
            "Standard deviation": float(std_sim),
            "Augmented Response": augmented_answer,
            "Perplexity": float(uncertainty_entropy),
        }
        save_result_to_json(results)


        # Use agents via orchestrator
        print("\n--- Using Agents via the Orchestrator ---")
        clear_gpu_memory()
        clinical_agent = ClinicalReasoningAgent()
        evidence_agent = EvidenceRetrievalAgent(api_key=pubmed_api_key)
        human_agent = HumanExpertReviewAgent(threshold_uncertainty=10.0, api_key=pubmed_api_key)
        orchestrator = AgentOrchestrator(clinical_agent, evidence_agent, human_agent)
        
        clinical_response = clinical_agent.process(
        question=user_question,
        evidence=augmented_answer.get('generation', '')
        )
        print("\nClinical Agent Response:")
        print(clinical_response.get('generation', ''))

        # Optionally, you can also retrieve extra evidence with the evidence agent.
        additional_evidence = evidence_agent.process(user_question)
        print("\nAdditional Evidence Retrieved:")
        print(additional_evidence)

        # Combine the clinical response and any additional evidence if desired.
        combined_response_text = (
            f"{clinical_response.get('generation', '')}\n\nAdditional Evidence:\n{additional_evidence}"
        )

        deepseek_refined_response = refine_with_deepseek(
        question=user_question,
        response={"generation": combined_response_text},
        max_new_tokens=500, 
        temperature=0.7,
        top_p=0.9
        )
        print("\nDeepSeek Refined Response:")
        print(deepseek_refined_response)

        expert_flag = input("\nIs a human expert available for review? (y/n): ").strip().lower() == 'y'

        clear_gpu_memory()
        if expert_flag:  # expert_flag is set based on user input.
            final_response = human_agent.process(
                question=user_question,
                response=deepseek_refined_response,
                api_key=pubmed_api_key
            )
            print("\nFinal Response after Human Expert Review:")
            print(final_response)
        else:
            final_response = deepseek_refined_response
            print("\nFinal Response (No Human Expert Review):")
            print(final_response)

        print("\n--- Using Agents via the Orchestrator ---")
        final_response = orchestrator.handle_query(user_question, expert_available=expert_flag)
        print("\nFinal Response from Agents:", final_response.get('generation', ''))
        
        # Finally, check bias of the deepseek refined response.
        print("\nBias Detection Results:")
        print("Lexical Bias:", detect_lexical_bias(final_response))
        print("Sentiment Bias:", analyse_sentiment(final_response))
        
        # Final memory cleanup
        clear_gpu_memory()

    except Exception as e:
        clear_gpu_memory()
        print("An error occurred:", e)
