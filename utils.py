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
