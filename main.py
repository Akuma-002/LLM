import ollama
from num2words import num2words
import json
question = input("Give me your study note : ")


#note
note = f"""{question}"""
mcq = int(input("How many mcqs do you want? : "))
marks1 = int(input("How many 1 mark question do you need? : ")) 
marks2 = int(input("How many 2 mark question do you need? : "))
marks3 = int(input("How many 3 mark question do you need? : "))
marks5 = int(input("How many 5 mark question do you need? : "))


prompt = f"""
You are a teaching assistant. Read the following note and generate:

"""


# Prompt for generating 2/3/5 mark questions
if(mcq != 0):
    prompt = prompt + f"""
    . {num2words(mcq)} 1-mark MCQ question"""


if(marks1 != 0):
    prompt = prompt + f"""
    . {num2words(marks1)} 1-mark question (not MCQ)"""

if(marks2 != 0):
    prompt = prompt + f"""
    . {num2words(marks2)} 2-mark question"""

if(marks3 != 0):
    prompt = prompt + f"""
    . {num2words(marks3)} 3-mark question"""

if(marks5 != 0):
    prompt = prompt + f"""
    . {num2words(marks5)} 5-mark question"""
    

prompt = prompt + f"""
Note:
{note}"""

print(prompt)


#llama3.1:8b 
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

# Process into list of questions
lines = [line.strip() for line in generated_text.split('\n') if line.strip()]
questions = []

for line in lines:
    if line[0].isdigit() or line.startswith("- "):
        line = line.lstrip("0123456789.- ")
    questions.append(line)

# Prepare JSON structure
output_json = {
    "note": note.strip(),
    "questions": questions
}

# Save to file
with open("questions.json", "w", encoding="utf-8") as f:
    json.dump(output_json, f, indent=4, ensure_ascii=False)

print("\n✅ Questions saved to 'questions.json'")
