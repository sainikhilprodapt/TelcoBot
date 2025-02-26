from config import Config
import google.generativeai as genai
from database import Database
from vectorstore import VectorStore
from typing import List,Dict


class Chatbot:
    def __init__(self):
        genai.configure(api_key=Config.GOOGLE_API_KEY)
        self.model = genai.GenerativeModel(Config.GEMINI_MODEL)
        self.db = Database()
        self.vector_store = VectorStore()
        self.conversation_history = []
        self.initialize_knowledge_base()

    def initialize_knowledge_base(self):
        qa_pairs = self.db.get_qa_pairs()
        self.vector_store.add_qa_pairs(qa_pairs)

    def generate_response(self, query: str) -> tuple[str, float, List[Dict]]:
        # Save query to conversation history
        self.conversation_history.append(f"Customer: {query}")

        # Retrieve relevant QA pairs
        relevant_pairs, similarities = self.vector_store.search(query)
        best_similarity = max(similarities)

        # Construct prompt with context
        context = "\n\n".join([
            f"Q: {qa['question']}\nA: {qa['answer']}"
            for qa in relevant_pairs
        ])

        prompt = f"""You are a knowledgeable and professional telecom customer support agent. Your goal is to provide accurate and relevant responses based on the provided similar Q&A pairs.  

                Instructions:
                - If the customer query closely resembles an existing Q&A pair, adapt the provided answer accordingly while ensuring clarity and professionalism.
                - If the query is telecom-related but does not match any existing Q&A pair, respond with:
                  **"This is beyond my scope. Please contact our customer care team for further assistance."**
                - If the query is unrelated to telecom services, respond with:
                  **"I specialize in telecom-related queries. Unfortunately, I cannot assist with this request."**
                - If the customer indicates they wish to end the conversation, acknowledge their request and provide a polite closing response.  

        
        Similar Q&A pairs:
        {context}

        New customer query: {query}"""

        # Generate response
        response = self.model.generate_content(prompt)
        generated_response = response.text

        # Save response to conversation history
        self.conversation_history.append(f"Bot: {generated_response}")

        return generated_response, best_similarity, relevant_pairs

    def get_conversation_history(self) -> str:
        return "\n".join(self.conversation_history)