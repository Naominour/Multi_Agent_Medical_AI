from agents.base_agent import BaseAgent
from utils.pubmed_api import fetch_pubmed_info

class EvidenceRetrievalAgent(BaseAgent):
    def __init__(self, name="EvidenceRetrievalAgent", api_key=None):
        # Agent that fetches medical evidence from PubMed.
        super().__init__(name)
        self.api_key = api_key

    def process(self, query):
        # Retrieves medical literature from PubMed.
        try:
            evidence = fetch_pubmed_info(query, self.api_key)
            evidence_text = "\n\n".join([f"[Reference {i+1}]: {ref}" for i, ref in enumerate(evidence)])
            return evidence_text
        except Exception as e:
            print(f"Error in EvidenceRetrievalAgent: {str(e)}")
            return "Error retrieving evidence"
