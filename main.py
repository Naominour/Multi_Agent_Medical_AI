from agents.orchestrator import AgentOrchestrator
from agents.reasoning_agent import ClinicalReasoningAgent
from agents.retrieval_agent import EvidenceRetrievalAgent
from agents.human_review_agent import HumanExpertReviewAgent
from models.llama_model import LlamaModel
from models.deepseek_model import DeepSeekModel
from utils.memory_management import clear_gpu_memory
import json

llama = LlamaModel(
    ckpt_dir="data/llama/",
    tokenizer_path="data/llama/tokenizer.model"
)

deepseek = DeepSeekModel(model_path="data/deepseek/merged_model")

# Load AI agents
reasoning_agent = ClinicalReasoningAgent(llama_model=llama)
retrieval_agent = EvidenceRetrievalAgent(api_key="YOUR_PUBMED_API_KEY")
human_agent = HumanExpertReviewAgent(deepseek_model=deepseek)

orchestrator = AgentOrchestrator(reasoning_agent, retrieval_agent, human_agent)

# Take user input
user_question = input("\nEnter your medical question: ")

# Process query
expert_flag = input("\nIs a human expert available for review? (y/n): ").strip().lower() == 'y'
final_response = orchestrator.handle_query(user_question, expert_available=expert_flag)

print("\nFinal Response:")
print(final_response.get('generation', ''))

# Save response as JSON
result_data = {"User Question": user_question, "Final Response": final_response}
with open("data/output/result.json", "w") as file:
    json.dump(result_data, file, indent=4)
print("\n✅ Response saved in data/output/result.json")

clear_gpu_memory()
