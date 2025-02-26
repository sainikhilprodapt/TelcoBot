import psycopg2
from psycopg2.extras import RealDictCursor
from config import Config

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