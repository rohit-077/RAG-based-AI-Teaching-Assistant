import requests
from pathlib import Path


def generate_response(prompt_text: str):
    """
    Sends the given prompt to the DeepSeek model hosted locally on Ollama 
    and returns the generated text response.
    """
    response = requests.post(
        "http://localhost:11434/api/generate",
        json={
            "model": "deepseek-r1:1.5b",
            "prompt": prompt_text,
            "stream": False
        }
    )

    # Check for errors
    if response.status_code != 200:
        raise Exception(f"Request failed: {response.status_code} - {response.text}")

    return response.json().get("response", "")


def main():
    # Define paths
    input_file = Path("prompt.txt")
    output_dir = Path("outputs")
    output_dir.mkdir(exist_ok=True)
    output_file = output_dir / "answer.txt"

    # Read the prompt
    if not input_file.exists():
        print("Error: 'prompt.txt' not found.")
        return

    with open(input_file, "r", encoding="utf-8") as f:
        prompt_text = f.read().strip()

    # Generate response
    print("Generating response from DeepSeek model...")
    answer = generate_response(prompt_text)

    # Save the generated response
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(answer)

    print(f"✅ Response saved successfully at: {output_file}")


if __name__ == "__main__":
    main()
