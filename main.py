import ollama
from num2words import num2words
import json



while True:
    question = input("Give me your study note : ")
    #note
    note = f"""{question}"""
    mcq = int(input("How many mcqs do you want? : "))
    marks1 = int(input("How many 1 mark question do you need? : ")) 
    marks2 = int(input("How many 2 mark question do you need? : "))
    marks3 = int(input("How many 3 mark question do you need? : "))
    marks5 = int(input("How many 5 mark question do you need? : "))

    print("\n\n\n")
    answer = input("Do you want answer? (Yes/No): ")
    answer = answer.lower()


    prompt = f"""
    You are a teaching assistant. Read the following note and generate:

    """


    #MCQ/2/3/5 mark questions
    if(mcq != 0):
        if(mcq > 1):
            prompt = prompt + f"""
            {num2words(mcq).capitalize()} 1-mark MCQ questions."""
        else:
            prompt = prompt + f"""
            One 1-mark MCQ question."""


    if(marks1 != 0):
        if(marks1 > 1):
            prompt = prompt + f"""
            {num2words(marks1).capitalize()} 1-mark SAQ questions."""
        else:
            prompt = prompt + f"""
            One 1-mark SAQ question."""

    if(marks2 != 0):
        if(marks2 > 1):
            prompt = prompt + f"""
            {num2words(marks2).capitalize()} 2-mark questions."""
        else:
            prompt = prompt + f"""
            One 2-mark question."""

    if(marks3 != 0):
        if(marks3 > 1):
            prompt = prompt + f"""
            {num2words(marks3).capitalize()} 3-mark questions."""
        else:
            prompt = prompt + f"""
            One 3-mark question."""

    if(marks5 != 0):
        if(marks5 > 1):
            prompt = prompt + f"""
            {num2words(marks5).capitalize()} 5-mark questions."""
        else:
            prompt = prompt + f"""
            One 5-mark question."""

    prompt = prompt + f"""
    Note:
    {note}"""

    if(answer == "yes"):
        prompt = prompt + "\n\nAnd also give all answer acording to the note."
    else:
        prompt = prompt + "\n\nDon't give me the answers."


    print(prompt)

    print("\n\n\n\n\n")

    #LLM
    response = ollama.chat(
        model='llama3.1:8b',
        messages=[
            {"role": "system", "content": "You are an AI that generates exam-style questions from educational notes."},
            {"role": "user", "content": prompt}
        ]
    )

    #output
    generated_text = response['message']['content'].strip()
    print("\nGenerated Questions:\n")
    print(generated_text)

    # # Process list
    # lines = [line.strip() for line in generated_text.split('\n') if line.strip()]
    # questions = []

    # for line in lines:
    #     if line[0].isdigit() or line.startswith("- "):
    #         line = line.lstrip("0123456789.- ")
    #     questions.append(line)

    # #JSON 
    # output_json = {
    #     "note": note.strip(),
    #     "questions": questions
    # }

    # # Save question
    # with open("questions.json", "w", encoding="utf-8") as f:
    #     json.dump(output_json, f, indent=4, ensure_ascii=False)

    # print("\nQuestions saved to 'questions.json'")
