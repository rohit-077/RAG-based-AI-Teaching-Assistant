import requests
import os
import json
import pandas as pd
import joblib

# function to create embeddings using Ollama API
def create_embeddings(text_list):
    try:
        # hitting local ollama api
        response = requests.post("http://localhost:11434/api/embed", json={
            "model": "bge-m3",
            "input": text_list
        })
    except Exception as e:
        print("couldn't connect to ollama api:", e)
        return None

    # basic response check
    if response.status_code == 200:
        try:
            data = response.json()
            return data.get("embeddings", [])
        except Exception as e:
            print("error while parsing api response:", e)
            return None
    else:
        print(f"ollama api returned an error: {response.status_code} - {response.text}")
        return None


# main logic
print("starting embedding generation process...")

# check if converted_texts folder exists
if not os.path.exists("converted_texts"):
    print("converted_texts folder not found. please make sure transcribed json files exist.")
    exit()

texts = os.listdir("converted_texts")

if not texts:
    print("no json files found in converted_texts folder.")
    exit()

chunks = []

# loop through each json file
for text_file in texts:
    file_path = os.path.join("converted_texts", text_file)
    print(f"\nprocessing file: {file_path}")

    try:
        with open(file_path, "r", encoding="utf-8") as f:
            all_chunks = json.load(f)
    except Exception as e:
        print(f"failed to read {text_file}: {e}")
        continue

    # extract text from chunks
    text_list = [ch.get('text', '') for ch in all_chunks.get("chunks", []) if ch.get('text')]
    if not text_list:
        print(f"no valid text found in {text_file}, skipping.")
        continue

    # get embeddings
    embeddings = create_embeddings(text_list)
    if embeddings is None:
        print(f"failed to generate embeddings for {text_file}.")
        continue

    # pair embeddings with text
    for i, chunk in enumerate(all_chunks["chunks"]):
        try:
            chunks.append({
                "chunk_id": i + 1,
                "chunk_vectors": embeddings[i],
                "text": chunk["text"]
            })
        except IndexError:
            print(f"embedding mismatch for {text_file}, chunk {i+1}. skipping this one.")
            continue

# if no chunks created, exit early
if not chunks:
    print("no embeddings were generated. check your input or api connection.")
    exit()

# convert to dataframe
print("\ncreating dataframe...")
try:
    df = pd.DataFrame(chunks)
except Exception as e:
    print("failed to create dataframe:", e)
    exit()

# save dataframe
try:
    joblib.dump(df, "embeddings.joblib")
    print("embeddings saved successfully to embeddings.joblib")
except Exception as e:
    print("error while saving embeddings:", e)

print("\nall done. embeddings file is ready.")
