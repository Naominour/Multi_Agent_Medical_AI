from agents.base_agent import BaseAgent
from utils.bias_detection import detect_lexical_bias, analyse_sentiment
from models.deepseek_model import DeepSeekModel

class HumanExpertReviewAgent(BaseAgent):
    def __init__(self, name="HumanExpertReviewAgent", deepseek_model=None):
        # Agent that refines responses with human oversight.
        super().__init__(name)
        self.deepseek = deepseek_model

    def process(self, question, response):
        # Allows human review and refines the response with DeepSeek.
        current_answer = response
        while True:
            print("\n--- Human Expert Review ---")
            print("Current Answer:")
            print(current_answer)

            expert_approval = input("Do you approve the LLM's final answer? (y/n): ").strip().lower()
            if expert_approval == "y":
                return current_answer

            expert_comment = input("Provide expert comment for refinement:\n")
            new_prompt = f"""Refine the following medical response based on expert feedback.

            ### Original Answer:
            {current_answer}

            ### Expert Comment:
            {expert_comment}

            ### Question:
            {question}

            ### Revised Answer:"""

            new_response = self.deepseek.refine_response(question, new_prompt)
            current_answer = new_response
