from agents.reasoning_agent import ClinicalReasoningAgent
from agents.retrieval_agent import EvidenceRetrievalAgent
from agents.human_review_agent import HumanExpertReviewAgent

class AgentOrchestrator:
    def __init__(self, reasoning_agent, retrieval_agent, human_agent=None):
        # Manages AI agents to generate high-quality responses.
        self.reasoning_agent = reasoning_agent
        self.retrieval_agent = retrieval_agent
        self.human_agent = human_agent

    def handle_query(self, question, expert_available=False):
        try:
            evidence = self.retrieval_agent.process(question)
            reasoning_response = self.reasoning_agent.process(question, evidence)
            final_response = reasoning_response.get('generation', '')

            if expert_available and self.human_agent:
                final_response = self.human_agent.process(question, final_response)

            return {'generation': final_response}
        except Exception as e:
            print(f"Error in Orchestrator: {str(e)}")
            return {'generation': 'Error processing query'}
