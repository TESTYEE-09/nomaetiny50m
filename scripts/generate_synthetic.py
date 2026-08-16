import json, random
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
random.seed(42)

records = []

# simple arithmetic
for _ in range(12000):
    kind = random.choice(["add", "sub", "mul", "div"])
    if kind == "add":
        a, b = random.randint(0, 20), random.randint(0, 20)
        q, a_str = f"what is {a} + {b}?", str(a + b)
    elif kind == "sub":
        a, b = random.randint(0, 20), random.randint(0, 20)
        q, a_str = f"what is {a} - {b}?", str(a - b)
    elif kind == "mul":
        a, b = random.randint(1, 12), random.randint(1, 12)
        q, a_str = f"what is {a} * {b}?", str(a * b)
    else:
        b = random.randint(1, 10)
        a = b * random.randint(1, 10)
        q, a_str = f"what is {a} / {b}?", str(a // b)
    records.append({"role": "user", "content": q}); records.append({"role": "assistant", "content": a_str})
    if random.random() < 0.5:
        sym = {"add": "+", "sub": "-", "mul": "*", "div": "/"}[kind]
        q2 = random.choice([f"whats {a} {sym} {b}", f"what's {a} {sym} {b}?", f"what is {a}{sym}{b}?", f"{a}{sym}{b}?", f"{a} {sym} {b} =", f"whats {a}{sym}{b}", f"whats {a}{sym}{b}?", f"what's {a}{sym}{b}", f"what is {a} {sym} {b}", f"{a} {sym} {b}?"])
        records.append({"role": "user", "content": q2}); records.append({"role": "assistant", "content": a_str})

# trivia facts
facts = [
    ("What is the capital of France?", "Paris"),
    ("What is the capital of Japan?", "Tokyo"),
    ("What is the capital of Italy?", "Rome"),
    ("What is the capital of Spain?", "Madrid"),
    ("What is the capital of Germany?", "Berlin"),
    ("What is the capital of the United Kingdom?", "London"),
    ("What is the capital of Canada?", "Ottawa"),
    ("What is the capital of Australia?", "Canberra"),
    ("What is the capital of Brazil?", "Brasília"),
    ("What is the capital of Russia?", "Moscow"),
    ("What is the capital of China?", "Beijing"),
    ("What is the capital of India?", "New Delhi"),
    ("What is the capital of Egypt?", "Cairo"),
    ("What is the capital of Mexico?", "Mexico City"),
    ("What is the capital of the United States?", "Washington, D.C."),
    ("What planet is known as the Red Planet?", "Mars"),
    ("What is the largest planet in the solar system?", "Jupiter"),
    ("How many continents are there?", "Seven"),
    ("How many colors are in a rainbow?", "Seven"),
    ("How many days are in a week?", "Seven"),
    ("How many months are in a year?", "Twelve"),
    ("How many hours are in a day?", "Twenty-four"),
    ("How many minutes are in an hour?", "Sixty"),
    ("What is the smallest prime number?", "2"),
    ("What is 2 + 2?", "4"),
    ("What is 1 + 1?", "2"),
    ("What is 10 + 5?", "15"),
    ("What is 3 + 3?", "6"),
    ("What is 4 + 4?", "8"),
    ("What is 5 + 5?", "10"),
    ("What is 6 + 6?", "12"),
    ("What is 7 + 7?", "14"),
    ("What is 8 + 8?", "16"),
    ("What is 9 + 9?", "18"),
    ("What is 100 - 1?", "99"),
    ("What is 50 - 25?", "25"),
    ("What is 12 * 12?", "144"),
    ("What is 3 * 3?", "9"),
    ("What is 6 * 7?", "42"),
    ("What is 9 * 9?", "81"),
    ("What is 10 / 2?", "5"),
    ("What is 100 / 10?", "10"),
    ("What does a dog say?", "Woof"),
    ("What does a cat say?", "Meow"),
    ("What color is the sky on a clear day?", "Blue"),
    ("What color is grass?", "Green"),
    ("What color is the sun?", "Yellow"),
    ("What is the opposite of hot?", "Cold"),
    ("What is the opposite of up?", "Down"),
    ("What is the opposite of big?", "Small"),
    ("What is the opposite of day?", "Night"),
    ("How many legs does a dog have?", "Four"),
    ("How many legs does a spider have?", "Eight"),
    ("How many wheels does a car have?", "Four"),
    ("How many wheels does a bicycle have?", "Two"),
    ("What do you use to write on paper?", "A pencil"),
    ("What do you read books with?", "Your eyes"),
    ("What does the sun do in the morning?", "It rises"),
    ("What do you eat food with?", "Your mouth"),
    ("What is the freezing point of water in Celsius?", "0"),
    ("What is the boiling point of water in Celsius?", "100"),
    ("How many sides does a triangle have?", "Three"),
    ("How many sides does a square have?", "Four"),
    ("How many sides does a hexagon have?", "Six"),
    ("What is the first letter of the alphabet?", "A"),
    ("What is the last letter of the alphabet?", "Z"),
    ("How many letters are in the word cat?", "Three"),
    ("What is a baby dog called?", "A puppy"),
    ("What is a baby cat called?", "A kitten"),
    ("What is a baby cow called?", "A calf"),
    ("What do bees make?", "Honey"),
    ("What animal gives us milk?", "A cow"),
    ("What animal says moo?", "A cow"),
    ("What animal says quack?", "A duck"),
    ("What is the fastest land animal?", "The cheetah"),
    ("What is the largest animal in the world?", "The blue whale"),
    ("How many eyes do humans have?", "Two"),
    ("How many fingers are on one hand?", "Five"),
    ("What organ pumps blood?", "The heart"),
    ("What organ do you breathe with?", "The lungs"),
    ("What color do you get by mixing red and white?", "Pink"),
    ("What color do you get by mixing blue and yellow?", "Green"),
    ("What color do you get by mixing red and yellow?", "Orange"),
    ("What is the fifth day of the week?", "Friday"),
    ("What is the first day of the week?", "Monday"),
    ("How many seasons are there in a year?", "Four"),
    ("What is the season after winter?", "Spring"),
    ("What is the season after summer?", "Autumn"),
    ("What do you wear on your feet?", "Shoes"),
    ("What do you wear on your head?", "A hat"),
    ("What is the currency of the United States?", "The dollar"),
    ("What is the currency of Japan?", "The yen"),
    ("What is the currency of the United Kingdom?", "The pound"),
    ("What is the currency of Europe?", "The euro"),
    ("How many planets are in the solar system?", "Eight"),
    ("What is the closest planet to the sun?", "Mercury"),
    ("What is the hottest planet in the solar system?", "Venus"),
    ("What is the Earth's only natural satellite?", "The Moon"),
    ("What is the tallest animal?", "The giraffe"),
    ("What is the tallest mountain in the world?", "Mount Everest"),
    ("What is the largest ocean?", "The Pacific Ocean"),
    ("What is the largest continent?", "Asia"),
    ("What is the driest desert in the world?", "The Atacama Desert"),
    ("What is the largest desert in the world?", "The Sahara Desert"),
    ("How many days are in a leap year?", "366"),
    ("How many days are in a normal year?", "365"),
    ("What is 0 + 0?", "0"),
    ("What is 1 * 1?", "1"),
    ("What is 11 + 11?", "22"),
    ("What is 20 + 20?", "40"),
    ("What is 15 + 15?", "30"),
    ("What is 25 + 25?", "50"),
    ("What is 7 + 8?", "15"),
    ("What is 9 + 7?", "16"),
    ("What is 12 + 9?", "21"),
    ("What is 14 + 6?", "20"),
    ("What is 16 + 4?", "20"),
    ("What is 18 + 2?", "20"),
    ("What is 19 + 1?", "20"),
    ("What is 5 - 3?", "2"),
    ("What is 8 - 6?", "2"),
    ("What is 10 - 4?", "6"),
    ("What is 13 - 7?", "6"),
    ("What is 15 - 9?", "6"),
    ("What is 20 - 12?", "8"),
    ("What is 30 - 15?", "15"),
    ("What is 40 - 20?", "20"),
    ("What is 2 * 2?", "4"),
    ("What is 2 * 3?", "6"),
    ("What is 4 * 5?", "20"),
    ("What is 5 * 6?", "30"),
    ("What is 7 * 8?", "56"),
    ("What is 8 * 9?", "72"),
    ("What is 9 * 5?", "45"),
    ("What is 10 * 10?", "100"),
    ("What is 12 * 3?", "36"),
    ("What is 6 / 3?", "2"),
    ("What is 9 / 3?", "3"),
    ("What is 12 / 4?", "3"),
    ("What is 16 / 4?", "4"),
    ("What is 20 / 5?", "4"),
    ("What is 25 / 5?", "5"),
    ("What is 30 / 6?", "5"),
    ("What is 36 / 6?", "6"),
    ("What is 42 / 7?", "6"),
    ("What is 49 / 7?", "7"),
    ("What is 56 / 8?", "7"),
    ("What is 64 / 8?", "8"),
    ("What is 72 / 9?", "8"),
    ("What is 81 / 9?", "9"),
    ("What is 90 / 10?", "9"),
    ("What is 100 / 20?", "5"),
]
for q, a in facts:
    records.append({"role": "user", "content": q}); records.append({"role": "assistant", "content": a})

with (ROOT / "data" / "hf_raw" / "synthetic.jsonl").open("w", encoding="utf-8") as f:
    for i in range(0, len(records), 2):
        record = {"id": f"hf-synthetic-{i:06d}", "category": "math" if "what is" in records[i]["content"] and any(c.isdigit() for c in records[i]["content"]) else "trivia", "messages": [records[i], records[i + 1]]}
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
print(f"records: {len(records)//2}  (12k arithmetic + {len(facts)} facts)")