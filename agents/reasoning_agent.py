from agents.base_agent import BaseAgent
from models.llama_model import LlamaModel

class ClinicalReasoningAgent(BaseAgent):
    def __init__(self, name="ClinicalReasoningAgent", llama_model=None):
        # Agent that generates clinical reasoning responses.
        super().__init__(name)
        self.llama = llama_model

    def process(self, question, evidence=None):
        # Generates a structured medical answer using LLaMA.
        try:
            prompt = f"Question: {question}\n"
            if evidence:
                prompt += f"Evidence:\n{evidence}\n"

            prompt += """Let's reason through the problem step by step:
            Step 1: Analyse the question and evidence
            Step 2: Identify key medical concepts
            Step 3: Form a structured response

            Based on the above steps, here is the final answer:\n"""

            response = self.llama.generate_response(prompt, formatted=True)
            return response if response else {'generation': 'No valid response generated.'}

        except Exception as e:
            print(f"Error in ClinicalReasoningAgent: {str(e)}")
            return {'generation': 'Error in processing query.'}
