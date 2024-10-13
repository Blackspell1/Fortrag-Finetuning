from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# Load model directly
tokenizer = AutoTokenizer.from_pretrained("distilbert/distilgpt2")
model = AutoModelForCausalLM.from_pretrained("distilbert/distilgpt2")


def generate_response(prompt, max_length=100):
    # Encode the input prompt
    input_ids = tokenizer.encode(prompt, return_tensors="pt")

    # Generate a response
    with torch.no_grad():
        output = model.generate(
            input_ids,
            max_length=max_length,
            num_return_sequences=1,
            no_repeat_ngram_size=2,
            top_k=50,
            top_p=0.95,
            temperature=0.7
        )

    # Decode the generated response
    response = tokenizer.decode(output[0], skip_special_tokens=True)
    return response


def chat_with_ai():
    print("Welcome to the AI chat! Type 'exit' to end the conversation.")

    while True:
        user_input = input("You: ")

        if user_input.lower() == 'exit':
            print("AI: Goodbye! It was nice chatting with you.")
            break

        # Generate AI response
        prompt = f"Human: {user_input}\nAI:"
        ai_response = generate_response(prompt)

        # Extract the AI's response after the "AI:" prefix
        ai_response = ai_response.split("AI:", 1)[-1].strip()

        print(f"AI: {ai_response}")


if __name__ == "__main__":
    chat_with_ai()
