# 🏥 Multi Agent Medical AI 🤖  
🚀 This is an AI-driven **medical question-answering system** integrating **LLaMA 3.1, DeepSeek R1, and Transformer models** to provide **accurate, evidence-based medical responses**.  
It is powered by **3 AI Agents**, **Evidence Retrieval AI Agent**, **Clinical Reasoning AI Agent**, **Uncertainty & Refinement AI Agent**, and features **Human-in-the-Loop (HITL) Review** for ensuring reliability and reducing AI bias.


---
## 🔥 What's Special About MedAI?
✅ **Fine-Tuned LLaMA & DeepSeek on 20,000+ Medical Data Points** 📊  
   - Trained on a **large-scale medical dataset** for **domain-specific accuracy**.  
   - Ensures **high-quality, expert-verified medical answers**.  

📌 **GitHub Link to Fine-Tuned Models:**  
🔗 **[Fine-Tuned LLaMA 3.1 Model](https://github.com/Naominour/Fine_Tuning_LLaMA_Model)**  
🔗 **[Fine-Tuned DeepSeek R1 Model](https://github.com/Naominour/Fine-tuning-DeepSeek-R1)**  

---

## 🤖 AI Agent Architecture (Human-in-the-Loop)
MedAI's response pipeline includes **3 AI agents** working together, with optional **human expert review**.

### 🔹 **1️⃣ Evidence Retrieval Agent (AI)**
   - **Fetches medical literature** from **PubMed** and **other sources**.
   - Ensures responses are **evidence-based**.

### 🔹 **2️⃣ Clinical Reasoning Agent (AI)**
   - **Uses fine-tuned LLaMA 3** to generate a **structured medical response**.
   - Applies **chain-of-thought (CoT) reasoning**.

### 🔹 **3️⃣ Uncertainty & Refinement Agent (AI)**
   - **Fine-Tuned DeepSeek model refines** AI-generated responses.
   - **Bias detection & uncertainty scoring** (Monte Carlo Dropout, Perplexity).

### 🔹 **4️⃣ Human Expert Review (Optional)**
   - **Doctors or experts** review and refine responses.
   - Ensures **AI reliability** in **critical medical queries**.

> 💡 **Human-in-the-Loop (HITL) ensures AI-assisted, expert-reviewed responses!**

---

## 📌 Features
✅ **Fine-Tuned LLaMA & DeepSeek on 20,000+ Medical Data**  
✅ **AI-Powered Medical Answer Generation** (LLaMA 3 + DeepSeek)  
✅ **Evidence-Based Retrieval** (PubMed API)  
✅ **Bias & Uncertainty Detection** (Lexical & Sentiment Bias, Perplexity)  
✅ **Monte Carlo Dropout (MC-D) for Stability**  
✅ **Human Expert Review (HITL) for Refinement**  
✅ **Memory-Efficient Execution** (Prevents GPU OOM errors)  

---

## ⚡ Installation
### 🔹 1️⃣ Clone the Repository
```bash
git clone https://github.com/Naominour/Multi_Agent_Medical_AI.git
cd Multi_Agent_Medical_AI
```

### 🔹 2️⃣ Install Dependencies
```bash
pip install -r requirements.txt
```

### 🔹 3️⃣ Download Fine-Tuned LLaMA & DeepSeek Models 
Download the fine-tuned models from:  
🔗 **[Fine-Tuned Model Repository](https://github.com/your-finetuned-models-repo)**

Then, place them in:
```bash
data/llama/
data/deepseek/
```
📌 **Note: Use Hugging Face's transformers library to download models automatically.**

# 🚀 Running the Project

Once dependencies and models are set up, run:

```bash
python main.py
```

## ✅ Example Input:
```bash
Enter your medical question: What are the early signs of Alzheimer's disease?
Is a human expert available for review? (y/n): y
```
## ✅ Example Output:
```bash
{
  "User Question": "What are the early signs of Alzheimer's disease?",
  "Final Response": "Early signs of Alzheimer's include memory loss, confusion, difficulty in problem-solving, and changes in mood or personality."
}
```
## ✅ Response saved in: data/output/result.json

---

## 💡 How It Works
1️⃣ **Retrieve Evidence (AI) 🏛️**  
   - **PubMed API fetches** medical literature.  
   - Provides **peer-reviewed references** for credibility.  

2️⃣ **Generate Response (AI) 🤖**  
   - **Fine-Tuned LLaMA 3 generates structured clinical answers**.  
   - Applies **Chain-of-Thought (CoT) reasoning** for logical step-by-step answers.  

3️⃣ **Compute Relevance & Uncertainty (AI) 📊**  
   - **Sentence Transformer checks response relevance**.  
   - **Monte Carlo Dropout (MC-D) ensures stability**.  
   - **OPT-1.3B Model calculates perplexity (uncertainty score)**.  

4️⃣ **Refine with DeepSeek (AI) 🛠️**  
   - **Fine-Tuned DeepSeek refines responses for clarity & accuracy**.  
   - **Now enhanced with Chain-of-Thought (CoT) refinement** to ensure logical reasoning.  

5️⃣ **Human Expert Review (Optional) 🩺**  
   - **Doctors or professionals review AI-generated answers**.  
   - Ensures **safe, expert-verified responses**.
     
---

## 📊 Performance Metrics

   ✔️ **Relevance Score (Cosine Similarity)**
   ✔️ **Perplexity Score (Uncertainty)**
   ✔️ **Bias Detection (Lexical & Sentiment Bias)**
   ✔️ **Monte Carlo Dropout Stability**

---

## 🛠️ Configuration

Modify main.py to customize **API keys, models, or parameters**
```bash
retrieval_agent = EvidenceRetrievalAgent(api_key="YOUR_PUBMED_API_KEY")
llama = LlamaModel(ckpt_dir="data/llama/", tokenizer_path="data/llama/tokenizer.model")
deepseek = DeepSeekModel(model_path="data/deepseek/merged_model")
```

---

## 📌 Example Use Cases

  ✅ **Medical Question Answering – Get accurate, evidence-based medical responses.**
  ✅ **Clinical Decision Support – Doctors & researchers can validate AI-generated insights.**
  ✅ **Human-AI Collaboration – HITL ensures safer AI-assisted medicine.**

---

## 📜 License

This project is licensed under the **MIT License.**
