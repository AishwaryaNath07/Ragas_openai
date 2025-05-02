
from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings
import numpy as np
from dotenv import load_dotenv
import os
from ragas import evaluate
from ragas.llms import LangchainLLMWrapper
from ragas import EvaluationDataset
from ragas.metrics import LLMContextRecall, Faithfulness, FactualCorrectness
# Load environment variables from .env file
load_dotenv()

# Access environment variables
openai_key = os.getenv("OPENAI_API_KEY")
llm = ChatOpenAI(model="gpt-4o")
embeddings = OpenAIEmbeddings()
sample_docs = [
    "Jonas Salk developed the first successful polio vaccine, which has saved millions of lives worldwide.",
    "The double helix model of DNA, proposed by James Watson and Francis Crick, revealed the structure of genetic material and revolutionized molecular biology.",
    "Antoine Lavoisier is considered the father of modern chemistry for establishing the law of conservation of mass and helping to systematize chemical nomenclature.",
    "Galileo Galilei used his telescope to observe celestial bodies, discovering the moons of Jupiter, the phases of Venus, and supporting the heliocentric theory.",
    "Werner Heisenberg formulated the uncertainty principle, a fundamental concept in quantum mechanics stating that certain pairs of physical properties cannot be known simultaneously.",
    "The periodic table organizes all known chemical elements in a systematic way based on atomic number and properties, enabling prediction of element behavior.",
    "Tim Berners-Lee invented the World Wide Web in 1989, allowing people to access and share information globally using hyperlinks and web browsers.",
    "Nikola Tesla made pioneering contributions to the development of alternating current (AC), wireless transmission, and electromagnetic devices.",
    "Alexander Fleming discovered penicillin in 1928, marking the beginning of modern antibiotics and revolutionizing treatment of bacterial infections.",
    "Schrödinger's cat is a famous thought experiment that illustrates the concept of quantum superposition, where a cat in a box can be simultaneously alive and dead until observed."
]

class RAG:
    def __init__(self, model="gpt-4o"):
        self.llm = ChatOpenAI(model=model)
        self.embeddings = OpenAIEmbeddings()
        self.doc_embeddings = None
        self.docs = None

    def load_documents(self, documents):
        """Load documents and compute their embeddings."""
        self.docs = documents
        self.doc_embeddings = self.embeddings.embed_documents(documents)

    def get_most_relevant_docs(self, query):
        """Find the most relevant document for a given query."""
        if not self.docs or not self.doc_embeddings:
            raise ValueError("Documents and their embeddings are not loaded.")

        query_embedding = self.embeddings.embed_query(query)
        similarities = [
            np.dot(query_embedding, doc_emb)
            / (np.linalg.norm(query_embedding) * np.linalg.norm(doc_emb))
            for doc_emb in self.doc_embeddings
        ]
        most_relevant_doc_index = np.argmax(similarities)
        return [self.docs[most_relevant_doc_index]]

    def generate_answer(self, query, relevant_doc):
        """Generate an answer for a given query based on the most relevant document."""
        prompt = f"question: {query}\n\nDocuments: {relevant_doc}"
        messages = [
            ("system", "You are a helpful assistant that answers questions based on given documents only."),
            ("human", prompt),
        ]
        ai_msg = self.llm.invoke(messages)
        return ai_msg.content
# Initialize RAG instance
rag = RAG()

# Load documents
rag.load_documents(sample_docs)

# Query and retrieve the most relevant document
query = "Who was Nikola Tesla?"
relevant_doc = rag.get_most_relevant_docs(query)

# Generate an answer
answer = rag.generate_answer(query, relevant_doc)

print(f"Query: {query}")
print(f"Relevant Document: {relevant_doc}")
print(f"Answer: {answer}")

sample_queries = [
    "Who developed the polio vaccine?",
    "What is the significance of the double helix model in biology?",
    "Who is known as the father of modern chemistry?",
    "What did Galileo Galilei discover with his telescope?",
    "Who formulated the uncertainty principle in quantum mechanics?",
    "What is the significance of the periodic table?",
    "Who invented the World Wide Web?",
    "What was Nikola Tesla known for?",
    "Who discovered penicillin?",
    "What is Schrödinger's cat thought experiment?"
]

expected_responses = [
    "Jonas Salk developed the first successful polio vaccine, which has saved millions of lives worldwide.",
    "The double helix model of DNA, proposed by Watson and Crick, revealed the structure of genetic material and revolutionized molecular biology.",
    "Antoine Lavoisier is considered the father of modern chemistry for establishing the law of conservation of mass and helping to systematize chemical nomenclature.",
    "Galileo Galilei used his telescope to discover moons orbiting Jupiter, phases of Venus, and detailed observations of the Moon, supporting the heliocentric model.",
    "Werner Heisenberg formulated the uncertainty principle, stating that the position and momentum of a particle cannot both be precisely known at the same time.",
    "The periodic table organizes chemical elements by their atomic number and properties, allowing scientists to predict the behavior of elements.",
    "Tim Berners-Lee invented the World Wide Web, enabling information sharing over the internet through hyperlinks and web browsers.",
    "Nikola Tesla was known for his work on alternating current (AC), wireless communication, and numerous other groundbreaking innovations.",
    "Alexander Fleming discovered penicillin, the world’s first true antibiotic, which revolutionized medicine and saved countless lives.",
    "Schrödinger's cat is a thought experiment that illustrates the paradox of quantum superposition, showing how a cat can be both alive and dead until observed."
]


dataset = []

for query,reference in zip(sample_queries,expected_responses):

    relevant_docs = rag.get_most_relevant_docs(query)
    response = rag.generate_answer(query, relevant_docs)
    dataset.append(
        {
            "user_input":query,
            "retrieved_contexts":relevant_docs,
            "response":response,
            "reference":reference
        }
    )

evaluation_dataset = EvaluationDataset.from_list(dataset)
evaluator_llm = LangchainLLMWrapper(llm)

result = evaluate(dataset=evaluation_dataset,metrics=[LLMContextRecall(), Faithfulness(), FactualCorrectness()],llm=evaluator_llm)
print(result)
df = result.to_pandas()
df.to_csv("output/simple_rag_openai_results.csv", index=False)
print("✅ Full evaluation complete. Results saved to ragas_full_evaluation_results.csv")
