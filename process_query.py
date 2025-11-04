from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import joblib
import os
from create_embeddings import create_embeddings

# this script takes a user query, creates its embedding, compares with saved ones
# and prepares a final prompt file that can be used by an LLM to generate answers

print("starting query processing...")

# check if embeddings file exists
if not os.path.exists("embeddings.joblib"):
    print("embeddings.joblib not found. please run create_embeddings.py first.")
    exit()

# load the embeddings dataframe
try:
    df = joblib.load("embeddings.joblib")
    print("embeddings loaded successfully.")
except Exception as e:
    print("failed to load embeddings file:", e)
    exit()

# check if the dataframe has required columns
if "chunk_vectors" not in df.columns or "text" not in df.columns:
    print("invalid dataframe format. make sure it contains 'chunk_vectors' and 'text'.")
    exit()

# take query input
query = input("\nKindly ask your question related to the tutorial: ").strip()
if not query:
    print("no query provided. exiting.")
    exit()

# create query embedding
print("generating embedding for your query...")
query_embed = create_embeddings([query])
if query_embed is None or len(query_embed) == 0:
    print("failed to create query embedding. check your ollama service.")
    exit()

# flatten the embedding array for comparison
query_vector = query_embed[0]

# calculate similarity between query and all chunks
try:
    print("calculating similarity scores...")
    similarity = cosine_similarity(
        np.vstack(df["chunk_vectors"].values),
        [query_vector]
    ).flatten()
except Exception as e:
    print("error while calculating cosine similarity:", e)
    exit()

# get top 10 most similar chunks
top_matches = similarity.argsort()[::-1][:10]
new_df = df.loc[top_matches]

if new_df.empty:
    print("no relevant chunks found. try rephrasing your question.")
    exit()

# display top matches for quick check
print("\nTop relevant chunks:")
for idx, text in enumerate(new_df["text"].values, start=1):
    print(f"{idx}. {text[:150]}...")  # just showing first 150 chars

chunk_texts = new_df["text"].values

# create the prompt for final LLM answer generation
prompt = f"""
You are an intelligent RAG-based answer generator. 
The subject is a machine learning tutorial.

Instructions:
- Analyze the user's query and the provided relevant chunks.
- Only consider questions related to the machine learning tutorial.
- Generate a clear, straightforward answer. Do not ask the user any questions.
- If the chunks do not contain enough information to answer, respond exactly with:
  'Your query is not related to the topic. Please ask a different question. Thank you!'

User query: {query}

Relevant chunks:
{chunk_texts}
"""

# write the prompt to a text file
try:
    with open("prompt.txt", "w", encoding="utf-8") as f:
        f.write(prompt)
    print("\nprompt.txt file created successfully.")
except Exception as e:
    print("failed to write prompt file:", e)
    exit()

print("\nprocess complete. you can now feed 'prompt.txt' to your model for generating the answer.")
