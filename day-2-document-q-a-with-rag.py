# pip install -qU "google-genai==1.7.0" "faiss-cpu" "python-dotenv"

import os
import faiss
import numpy as np
from dotenv import load_dotenv
from google import genai
from google.genai import types
from google.api_core import retry

# Load environment variables and set up API key
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
client = genai.Client(api_key=GOOGLE_API_KEY)

# List models supporting embedContent
for m in client.models.list():
    if "embedContent" in m.supported_actions:
        print(m.name)

# Sample documents
DOCUMENT1 = """Operating the Climate Control System ..."""
DOCUMENT2 = """Your Googlecar has a large touchscreen ..."""
DOCUMENT3 = """Shifting Gears Your Googlecar has an automatic transmission ..."""
documents = [DOCUMENT1, DOCUMENT2, DOCUMENT3]

# Retry handler for quota
is_retriable = lambda e: (isinstance(e, genai.errors.APIError) and e.code in {429, 503})


# Embedding function for Gemini
@retry.Retry(predicate=is_retriable)
def get_embeddings(texts, task_type="retrieval_document"):
    response = client.models.embed_content(
        model="models/text-embedding-004",
        contents=texts,
        config=types.EmbedContentConfig(task_type=task_type),
    )
    return np.array([e.values for e in response.embeddings]).astype("float32")


# Embed the documents
doc_embeddings = get_embeddings(documents, task_type="retrieval_document")

# Create FAISS index
dimension = doc_embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(doc_embeddings)

# Store original documents separately for retrieval
doc_id_map = {i: doc for i, doc in enumerate(documents)}

# Query
query = "How do you use the touchscreen to play music?"
query_embedding = get_embeddings([query], task_type="retrieval_query")

# Search
k = 1
distances, indices = index.search(query_embedding, k)
retrieved_docs = [doc_id_map[i] for i in indices[0]]

print("Retrieved passage:")
print(retrieved_docs[0])

# Prompt assembly
query_oneline = query.replace("\n", " ")
prompt = f"""You are a helpful and informative bot that answers questions using text from the reference passage included below. 
Be sure to respond in a complete sentence, being comprehensive, including all relevant background information. 
However, you are talking to a non-technical audience, so be sure to break down complicated concepts and 
strike a friendly and conversational tone. If the passage is irrelevant to the answer, you may ignore it.

QUESTION: {query_oneline}
"""

for passage in retrieved_docs:
    if passage:  # Check if passage is not empty
        clean_passage = passage.replace("\n", " ")
        prompt += f"PASSAGE: {clean_passage}"
    else:
        print("Warning: Retrieved passage is empty.")

print("\nGeneration prompt:")
print(prompt)

# Generate an answer using the Gemini API
answer = client.models.generate_content(model="gemini-2.0-flash", contents=prompt)

print("\nGenerated answer:")
if hasattr(answer, "text"):
    print(answer.text)
else:
    print("Error: No answer generated.")
