from bot import Chatbot
from utils import is_farewell_message, collect_customer_details

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

        # if not is_telecom_related(query):
        #     print("\nTelcoBot: I specialize in telecom-related queries. Please ask me about mobile networks, SIM cards, data plans, billing, or connectivity issues.")
        #     continue

        try:
            response, similarity, relevant_pairs = chatbot.generate_response(query)
            print(f"\nTelcoBot: {response}")

            helpful = input("\nWas this response helpful? (yes/no): ").strip().lower()
            if helpful not in ['yes','no','y','n']:
                print("Please enter (yes/no)")

            if helpful in ['no', 'n']:
                print("\nI apologize that I couldn't fully resolve your issue.")
                print("Let me connect you with our customer care team.")

                name, phone = collect_customer_details()

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
