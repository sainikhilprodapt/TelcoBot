from config import Config
import numpy as np
import faiss
import google.generativeai as genai
from typing import List, Dict



class VectorStore:
    def __init__(self):
        genai.configure(api_key=Config.GOOGLE_API_KEY)
        self.embedding_model = genai.get_model(Config.EMBEDDING_MODEL)
        self.dimension = 768
        self.index = faiss.IndexFlatL2(self.dimension)
        self.qa_pairs = []

    def add_qa_pairs(self, qa_pairs: List[Dict]):
        print("Initializing knowledge base...")
        for qa in qa_pairs:
            combined_text = f"Question: {qa['question']}\nAnswer: {qa['answer']}"
            embedding = self.get_embedding(combined_text)
            self.index.add(np.array([embedding]))
            self.qa_pairs.append(qa)
        print("Knowledge base loaded")

    def get_embedding(self, text: str) -> np.ndarray:
        embedding = genai.embed_content(
            model=Config.EMBEDDING_MODEL,
            content=text,
            task_type="retrieval_document"
        )
        return np.array(embedding['embedding'])

    def search(self, query: str, k: int = 3) -> tuple[List[Dict], List[float]]:
        query_embedding = self.get_embedding(query)
        distances, indices = self.index.search(
            np.array([query_embedding]).astype('float32'), k
        )
        max_distance = np.max(distances)
        similarities = [1 - (dist / max_distance) for dist in distances[0]]
        return [self.qa_pairs[i] for i in indices[0]], similarities