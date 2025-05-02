from dotenv import load_dotenv
import os

# Load environment variables from .env file
load_dotenv()

# Access environment variables
openai_key = os.getenv("OPENAI_API_KEY")
from langchain_openai.chat_models import AzureChatOpenAI
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings
from ragas.metrics import AspectCritic
from datasets import load_dataset
from ragas import EvaluationDataset
from ragas import evaluate
evaluator_llm = LangchainLLMWrapper(ChatOpenAI(model="gpt-4o"))
evaluator_embeddings = LangchainEmbeddingsWrapper(OpenAIEmbeddings())

eval_dataset = load_dataset("explodinggradients/earning_report_summary",split="train")
eval_dataset = EvaluationDataset.from_hf_dataset(eval_dataset)
print("Features in dataset:", eval_dataset.features())
print("Total samples in dataset:", len(eval_dataset))

metric = AspectCritic(name="summary_accuracy",llm=evaluator_llm, definition="Verify if the summary is accurate.")
results = evaluate(eval_dataset, metrics=[metric])
print(results)
# Convert to pandas and save to CSV
df = results.to_pandas()
print(df)

# Export to CSV
csv_file = "evaluation_results.csv"
df.to_csv(csv_file, index=False)
print(f"✅ Results exported to {csv_file}")