import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer

def analyze_common_sense(prompts):
    # Load pre-trained GPT-2 model and tokenizer
    model_name = "gpt2-medium"
    model = GPT2LMHeadModel.from_pretrained(model_name)
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)

    results = []

    for prompt in prompts:
        input_ids = tokenizer.encode(prompt, return_tensors="pt")
        output = model.generate(input_ids, max_length=50, num_return_sequences=1, no_repeat_ngram_size=2)
        generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
        results.append({"prompt": prompt, "generated_text": generated_text})

    return results

# Example common sense prompts
prompts = [
    "The sun rises in the ",
    "To cool down, people often use ",
    "Eating too much candy can lead to ",
    "People use umbrellas to stay dry in ",
    "Birds can often be found in ",
]

# Analyze common sense for prompts
analysis_results = analyze_common_sense(prompts)
for result in analysis_results:
    print("Prompt:", result["prompt"])
    print("Generated Text:", result["generated_text"])
    print()

