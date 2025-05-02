from dotenv import load_dotenv
import os
from pypdf import PdfReader
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.metrics import (
    answer_correctness,
    faithfulness,
    context_precision,
    context_recall,
    answer_relevancy,
    answer_similarity,
    AspectCritic,
)
from ragas import evaluate, EvaluationDataset
import pandas as pd
from datasets import Dataset
from docx import Document
# Step 1: Load environment and keys
load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
import pandas as pd

def extract_text_from_excel(excel_path, sheet_name=0):
    df = pd.read_excel(excel_path, sheet_name=sheet_name)
    return "\n".join(df.astype(str).apply(lambda row: " | ".join(row), axis=1))
def extract_text_from_docx(docx_path):
    doc = Document(docx_path)
    return "\n".join([para.text for para in doc.paragraphs if para.text.strip()])
def extract_text_from_txt(txt_path):
    with open(txt_path, 'r', encoding='utf-8') as file:
        return file.read()
# Step 2: PDF reader utility
def extract_text_from_pdf(pdf_path):
    reader = PdfReader(pdf_path)
    return "\n".join([page.extract_text() for page in reader.pages if page.extract_text()])

text = extract_text_from_pdf("input/diabetes_treatment.pdf")  # Replace with your PDF path
#text = extract_text_from_txt("input/diabetes_treatment.txt") 
#text = extract_text_from_docx("input/diabetes_treatment.docx")
#text = extract_text_from_excel("input/diabetes_treatment.xlsx") 
# Simulated QA dataset
qa_data = [
    {
        "question": "What is the main topic of the PDF?",
        "user_input": "What is the main topic of the PDF?",
        "reference": "The document outlines possible lines of treatment for diabetes.",
        "answer": "The document outlines possible lines of treatment for diabetes.",
        "context": text,
        "retrieved_contexts": [text],
        "response": "The document discusses treatment options for diabetes."
    },
    {
        "question": "What lifestyle modifications are recommended for diabetes?",
        "user_input": "What lifestyle modifications are recommended for diabetes?",
        "answer": "Balanced diet, regular exercise, and weight management.",
        "reference": "Balanced diet, regular exercise, and weight management."  , 
        "context": text,
        "retrieved_contexts": [text],
        "response": "Healthy eating, exercise, and managing body weight are suggested."
    },
    {
        "question": "What is Metformin used for?",
        "answer": "Metformin lowers glucose production in the liver and is a first-line medication.",
        "user_input": "What is Metformin used for?",
        "reference": "Metformin lowers glucose production in the liver and is a first-line medication.",
        "context": text,
        "retrieved_contexts": [text],
        "response": "It is a first-line drug that reduces liver glucose production."
    },
    {
        "question": "What is CGM in the context of diabetes?",
        "user_input": "What is CGM in the context of diabetes?",
        "reference": "CGM stands for Continuous Glucose Monitoring, which tracks blood sugar levels throughout the day.",
        "answer": "CGM stands for Continuous Glucose Monitoring, which tracks blood sugar levels throughout the day.",
        "context": text,
        "retrieved_contexts": [text],
        "response": "CGM tracks blood sugar levels all day."
    },
    {
        "question": "What are examples of emerging therapies for diabetes?",
        "answer": "SGLT2 inhibitors and GLP-1 receptor agonists.",
        "user_input": "What are examples of emerging therapies for diabetes?",
        "reference": "SGLT2 inhibitors and GLP-1 receptor agonists.",
        "context": text,
        "retrieved_contexts": [text],
        "response": "New treatments include SGLT2 inhibitors and GLP-1 drugs."
    }
]


# Step 4: Convert to HF dataset
hf_dataset = Dataset.from_list(qa_data)
eval_dataset = EvaluationDataset.from_hf_dataset(hf_dataset)

# Step 5: LLM and Embedding setup
llm = LangchainLLMWrapper(ChatOpenAI(model="gpt-4"))
embedding = LangchainEmbeddingsWrapper(OpenAIEmbeddings())

# Optional: AspectCritic with a custom definition
aspect_critic_metric = AspectCritic(
    name="summary_accuracy",
    llm=llm,
    definition="Does the answer accurately reflect the key information?"
)

# Step 6: All available RAGAS metrics
all_metrics = [
    answer_correctness,
    faithfulness,
    context_precision,
    context_recall,
    answer_relevancy,
    answer_similarity,
    aspect_critic_metric
]

# Step 7: Evaluate
results = evaluate(
    dataset=eval_dataset,
    metrics=all_metrics,
    llm=llm,
    embeddings=embedding
)

# Step 8: Save results
df = results.to_pandas()
df.to_csv("ragas_full_evaluation_results.csv", index=False)
print("✅ Full evaluation complete. Results saved to ragas_full_evaluation_results.csv")
