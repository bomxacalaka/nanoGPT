from simplevibe import llama3
import random
import os
import time
import json

words_file_path = os.path.join(os.path.dirname(__file__), 'words.txt')
with open(words_file_path, 'r') as f:
    words = f.read().split(' ')

total_start = time.time()
iters = 5000
errors = 0
for i in range(iters):
    global_start = time.time()
    random_words = random.sample(words, 10)

    system_prompt = """You are an expert in turning random words into data rich texts, they must contain information such as names, ages, locations, professions, hobbies etc.
These are just examples, use the random words as inspiration for the information you provide.
You MUST only return the text without any additional commentary or explanation.
Make it short, no more than 100 words."""
    result = llama3(messages=[
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": " ".join(random_words)}],
        temperature=0.7,
        max_new_tokens=1000,
        )['output']

    system_prompt = """You are an expert in JSON composition parsing, and extracting meaningful information from it.
You MUST only return the JSON without any additional commentary or explanation.
You MUST always return the JSON in a code block.
Do not include any markdown or code block syntax in the JSON response.
No ```json```"""
    result_2 = llama3(messages=[
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": result}],
        temperature=0.5,
        max_new_tokens=1000,
        )['output']
    try:
        result_2 = json.loads(result_2.replace("```", ""))
    except (json.JSONDecodeError, KeyError) as e:
        print(f"JSON parsing failed, retrying... Error: {e}")

        if "Expecting ',' delimiter" in str(e):
            # Add a } to the end of the response and try to parse it again
            result_2 += '}'
            print("Added } to the end of the response and retrying...")
            try:
                result_2 = json.loads(result_2.replace("```", ""))
            except (json.JSONDecodeError, KeyError) as e:
                print(f"JSON parsing failed again... Error: {e}")
                errors += 1
                continue
    output_file_path = os.path.join(os.path.dirname(__file__), '..', 'shakespeare', 'input.txt')
    input_data = ""
    input_data += "<|start|>"
    input_data += "<|system|>"
    input_data += "Convert anything to JSON."
    input_data += "<|user|>"
    input_data += result
    input_data += "<|assistant|>"
    input_data += json.dumps(result_2)
    input_data += "<|end|>"
    input_data += "\n"
    with open(output_file_path, 'a') as f:
        f.write(input_data)
    # Also append to .jsonl
    json_output_file_path = os.path.join(os.path.dirname(__file__), 'output.jsonl')
    with open(json_output_file_path, 'a') as f:
        json.dump({"input": result, "output": result_2}, f)
        f.write("\n")
    global_end = time.time()
    print(f"{i:04d} | [input, output] len: {[len(result),len(json.dumps(result_2))]} | T: {global_end - global_start:05.2f}s | Errors: {errors:03d} | ETA: {time.time() - total_start:08.2f}s")

print(f"Total errors encountered: {errors}")
print(f"Total time for {iters} iterations: {time.time() - total_start:.2f} seconds")
# Total time for 2000 iterations: 15202.76 seconds