from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch
import socket


def is_online(host="8.8.8.8", port=53, timeout=3):
    """
    Checks if the machine is connected to the internet.
    Defaults to Google's DNS (8.8.8.8) on port 53.
    """
    try:
        socket.setdefaulttimeout(timeout)
        socket.socket(socket.AF_INET, socket.SOCK_STREAM).connect((host, port))
        return True
    except socket.error:
        return False


def load_model():
    print("Loading model from Hugging Face...")
    tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-base")
    model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-base")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    
    return tokenizer, model, device


def generate_questions(note, tokenizer, model, device, num_questions):
    prompt = f"Generate {num_questions} exam-style questions based on the following notes:\n\n{note}"
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, padding=True).to(device)

    outputs = model.generate(
        **inputs,
        max_length=256,
        num_return_sequences=num_questions,
        num_beams=4,
        temperature=0.9,
        top_p=0.95,
        do_sample=True,
        early_stopping=True
    )

    decoded = tokenizer.batch_decode(outputs, skip_special_tokens=True)
    return decoded


if __name__ == "__main__":
    print("=== Teach Assistant ===")
    all_questions = []

    if is_online():
        tokenizer, model, device = load_model()
        print("Online")
        while True:
            print("\nEnter your study note (or type 'exit' to quit):")
            note = input("> ").strip()
            if note.lower() == "exit":
                break

            try:
                count = int(input("Enter how many questions you want: "))
            except ValueError:
                print("Enter a valid input")
                continue

            questions = generate_questions(note, tokenizer, model, device, count)
            all_questions.extend(questions)

            print("\nGenerated Questions:")
            for i, q in enumerate(questions, 1):
                print(f"{i}. {q}")

    else:
        print("Offline ")
