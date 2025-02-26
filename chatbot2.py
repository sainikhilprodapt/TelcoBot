import os
import numpy as np
from dotenv import load_dotenv
import psycopg2
import google.generativeai as genai
import faiss
from typing import List,Dict
from psycopg2.extras import RealDictCursor

load_dotenv()

class Config:
    GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
    GEMINI_MODEL = "gemini-1.5-pro"
    EMBEDDING_MODEL = "models/embedding-001"
    DB_CONNECTION = os.getenv("DB_CONNECTION")


class Database:
    def __init__(self):
        self.conn = psycopg2.connect(Config.DB_CONNECTION)
        self._create_tables()

    def _create_tables(self):
        with self.conn.cursor() as cur:
            # Create escalations table if it doesn't exist
            cur.execute("""
                CREATE TABLE IF NOT EXISTS escalations (
                    id SERIAL PRIMARY KEY,
                    customer_name VARCHAR(100),
                    phone_number VARCHAR(20),
                    issue TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            self.conn.commit()

    def get_qa_pairs(self):
        with self.conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute("""
                SELECT question, answer
                FROM queries
            """)
            return cur.fetchall()

    def save_escalation(self, name: str, phone: str, issue: str, conversation: str):
        with self.conn.cursor() as cur:
            cur.execute("""
                INSERT INTO escalations (customer_name, phone_number, issue)
                VALUES (%s, %s, %s)
                RETURNING id
            """, (name, phone, issue))
            escalation_id = cur.fetchone()[0]
            self.conn.commit()
            return escalation_id


def is_farewell_message(text: str) -> bool:
    """Check if the message indicates the user wants to end the conversation"""
    farewell_phrases = [
        'thank you', 'thanks', 'bye', 'goodbye', 'that\'s all',
        'that is all', 'i am done', 'i\'m done', 'done', 'quit',
        'exit', 'that helps', 'that helped', 'got it', 'understood',
        'that\'s it', 'that is it', 'all set', 'i\'m good', 'i am good'
    ]

    text_lower = text.lower()
    return any(phrase in text_lower for phrase in farewell_phrases)


def collect_customer_details() -> tuple[str, str]:
    """Collect customer name and phone number"""
    while True:
        name = input("\nPlease enter your name: ").strip()
        if name:
            break
        print("Name cannot be empty.")

    while True:
        phone = input("Please enter your phone number: ").strip()
        if phone and phone.isdigit() and len(phone) >= 10:
            break
        print("Please enter a valid phone number (at least 10 digits).")

    return name, phone


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
            - If the customer query closely matches an existing Q&A pair, adapt the existing answer accordingly.  
            - If the query is telecom-related but does not exactly match any provided Q&A pair, respond with:  
              **"This is out of my context, Please contact our customer care if you want to know more about it."**  
            - If the query is significantly different and unrelated to telecom, respond with:  
              **"I am specialized to answer telecom-related queries. Unfortunately, I cannot assist with this request."**  
            - If the customer expresses a desire to end the conversation, acknowledge it with an appropriate closing response.  
            
            Maintain a professional and helpful tone in all responses.  

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


def main():
    chatbot = Chatbot()
    print("\nHi, this is TelcoBot! I'm here to help you with your telecom-related queries.")
    print("Feel free to ask your questions, and let me know when you're done.\n")

    while True:
        query = input("\nYour question: ").strip()

        if not query:
            print("Please enter a question.")
            continue

        if is_farewell_message(query):
            print("\nI hope I was able to help. Thank you for using TelcoBot. Have a great day!")
            break

        def is_telecom_related(user_query: str) -> bool:
            telecom_keywords = [
                "network", "sim", "recharge", "bill", "data", "4g", "5g", "plan", "coverage",
                "call", "sms", "internet", "customer service", "roaming", "telecom", "mobile",
                "connectivity", "balance", "billing", "top-up", "subscription", "voice", "operator"
            ]
            return any(keyword in user_query.lower() for keyword in telecom_keywords)

        if not is_telecom_related(query):
            print("\nTelcoBot: I specialize in telecom-related queries. Please ask me about mobile networks, SIM cards, data plans, billing, or connectivity issues.")
            continue

        try:
            response, similarity, relevant_pairs = chatbot.generate_response(query)
            print(f"\nTelcoBot: {response}")

            helpful = input("\nWas this response helpful? (yes/no): ").strip().lower()

            if helpful in ['no', 'n']:
                print("\nI apologize that I couldn't fully resolve your issue.")
                print("Let me connect you with our customer care team.")

                name, phone = collect_customer_details()

                # Retrieve the last user query from chat history
                last_query = None
                for msg in reversed(chatbot.conversation_history):
                    if msg.startswith("Customer:"):
                        last_query = msg.replace("Customer: ", "").strip()
                        break

                escalation_id = chatbot.db.save_escalation(
                    name=name,
                    phone=phone,
                    issue=last_query,
                    conversation=chatbot.get_conversation_history()
                )

                print("\nThank you for providing your details.")
                print("One of our customer care executives will contact you shortly at the provided number.")
                print(f"Your reference number is: #{escalation_id}")
                break

        except Exception as e:
            print(f"\nError: {str(e)}")
            print("Please try again with a different question.")


if __name__ == "__main__":
    main()
