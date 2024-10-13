import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import time
import sys
from tqdm import tqdm

# Load the tokenizer and model
model_id = "DiscoResearch/Llama3-German-8B"
print("Loading tokenizer and model...")
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id)
print("Model loaded successfully.")

# Set device to GPU if available, otherwise CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
print(f"Using device: {device}")

# Set the model to evaluation mode
model.eval()

# Function to generate responses with streaming and progress bar
def generate_response_stream(prompt, max_length=100):
    print("Tokenizing input...")
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    print("Generating response...")
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_length=max_length,
            num_return_sequences=1,
            temperature=0.7,  # Lower temperature for more deterministic output
            top_p=0.1,      # Top-p sampling
            repetition_penalty=1.2,  # Penalty for repetition
            do_sample=True   # Ensure sampling is enabled
        )
    response_ids = outputs[0]
    response_text = tokenizer.decode(response_ids, skip_special_tokens=True)
    print("Response generated.")

    # Estimate total time for generation
    total_tokens = len(response_text.split())
    estimated_time = total_tokens * 0.1  # Assuming 0.1 seconds per token

    # Stream the response word by word with progress bar
    sys.stdout.write("AI: ")
    with tqdm(total=total_tokens, desc="Generating response", unit="token") as pbar:
        for word in response_text.split():
            sys.stdout.write(word + ' ')
            sys.stdout.flush()
            time.sleep(0.1)  # Adjust the sleep time for faster or slower streaming
            pbar.update(1)
    print()  # Newline after the response
    print(response_text)  # Print the full response text after the progress bar

# Main interaction loop
print("AI Assistant is ready. Type 'exit' to quit.")
context = []
while True:
    user_input = input("You: ")
    if user_input.lower() == 'exit':
        break

    # Maintain context of the last 10 user inputs
    context.append(f"User: {user_input}")
    if len(context) > 10:
        context.pop(0)

    # Create full prompt with context
    full_prompt = "\n".join(context)

    print("Generating response...")
    generate_response_stream(full_prompt)

print("Thank you for using the AI Assistant!")
