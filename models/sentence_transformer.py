import torch
from sentence_transformers import SentenceTransformer, util
import numpy as np
import gc


def clear_gpu_memory():
    gc.collect()
    torch.cuda.empty_cache()

class SentenceTransformerModel:
    def __init__(self, model_name="all-MiniLM-L6-v2", device="cuda"):

        # Initialises a sentence embedding model for computing similarity.
        self.device = device if torch.cuda.is_available() else "cpu"
        self.model_name = model_name

        self.model = SentenceTransformer(self.model_name, device=self.device)

    def compute_similarity(self, sentence1, sentence2):

        # Computes cosine similarity between two sentences.
        try:
            embeddings = self.model.encode([sentence1, sentence2], convert_to_tensor=True)
            similarity = util.pytorch_cos_sim(embeddings[0], embeddings[1]).item()
            return similarity
        except Exception as e:
            print(f"❌ Error in compute_similarity: {str(e)}")
            return 0.0

    def compute_mc_uncertainty(self, samples):

        # Computes uncertainty metrics using Monte Carlo (MC) dropout.
        try:
            embeddings = self.model.encode(samples, convert_to_tensor=True)
            num_samples = len(embeddings)
            pairwise_similarities = []

            for i in range(num_samples):
                for j in range(i + 1, num_samples):
                    similarity = util.pytorch_cos_sim(embeddings[i], embeddings[j]).item()
                    pairwise_similarities.append(similarity)

            avg_similarity = np.mean(pairwise_similarities)
            std_similarity = np.std(pairwise_similarities)
            return avg_similarity, std_similarity
        except Exception as e:
            print(f"❌ Error in compute_mc_uncertainty: {str(e)}")
            return 0.0, 0.0

    def cleanup(self):
        self.model.to("cpu")
        clear_gpu_memory()
