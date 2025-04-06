# ##### Copyright 2025 Google LLC.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# # Day 2 - Document Q&A with RAG using Chroma
#
# Welcome back to the Kaggle 5-day Generative AI course!
#
# **NOTE**: The Day 1 notebook contains lots of information for getting set up with Kaggle Notebooks. If you are having any issues, please [check out the troubleshooting steps there](https://www.kaggle.com/code/markishere/day-1-prompting#Get-started-with-Kaggle-notebooks).
#
# Two big limitations of LLMs are 1) that they only "know" the information that they were trained on, and 2) that they have limited input context windows. A way to address both of these limitations is to use a technique called Retrieval Augmented Generation, or RAG. A RAG system has three stages:
#
# 1. Indexing
# 2. Retrieval
# 3. Generation
#
# Indexing happens ahead of time, and allows you to quickly look up relevant information at query-time. When a query comes in, you retrieve relevant documents, combine them with your instructions and the user's query, and have the LLM generate a tailored answer in natural language using the supplied information. This allows you to provide information that the model hasn't seen before, such as product-specific knowledge or live weather updates.
#
# In this notebook you will use the Gemini API to create a vector database, retrieve answers to questions from the database and generate a final answer. You will use [Chroma](https://docs.trychroma.com/), an open-source vector database. With Chroma, you can store embeddings alongside metadata, embed documents and queries, and search your documents.
#
# ## For help
#
# **Common issues are covered in the [FAQ and troubleshooting guide](https://www.kaggle.com/code/markishere/day-0-troubleshooting-and-faqs).**

# ## Setup
#
# First, install ChromaDB and the Gemini API Python SDK.

# Install dependencies
# import subprocess
# subprocess.run(["pip", "install", "-qU", "google-genai==1.7.0", "chromadb==0.6.3", "sentence-transformers"])

# > **Note**: After installing new packages, you need to restart the kernel for them to be available.

# Import libraries
import os
from google import genai
from google.genai import types
from IPython.display import Markdown
from dotenv import load_dotenv

# Load API key from environment
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# Initialize the client
client = genai.Client(api_key=GOOGLE_API_KEY)

# ## Explore available models
# You will be using the `embedContent` API method to calculate embeddings in this guide.

# List models that support embeddings
for m in client.models.list():
    if "embedContent" in m.supported_actions:
        print(m.name)

# ### Data
# Here is a small set of documents you will use to create an embedding database.

DOCUMENT1 = "Operating the Climate Control System  Your Googlecar has a climate control system that allows you to adjust the temperature and airflow in the car. To operate the climate control system, use the buttons and knobs located on the center console.  Temperature: The temperature knob controls the temperature inside the car. Turn the knob clockwise to increase the temperature or counterclockwise to decrease the temperature. Airflow: The airflow knob controls the amount of airflow inside the car. Turn the knob clockwise to increase the airflow or counterclockwise to decrease the airflow. Fan speed: The fan speed knob controls the speed of the fan. Turn the knob clockwise to increase the fan speed or counterclockwise to decrease the fan speed. Mode: The mode button allows you to select the desired mode. The available modes are: Auto: The car will automatically adjust the temperature and airflow to maintain a comfortable level. Cool: The car will blow cool air into the car. Heat: The car will blow warm air into the car. Defrost: The car will blow warm air onto the windshield to defrost it."
DOCUMENT2 = 'Your Googlecar has a large touchscreen display that provides access to a variety of features, including navigation, entertainment, and climate control. To use the touchscreen display, simply touch the desired icon.  For example, you can touch the "Navigation" icon to get directions to your destination or touch the "Music" icon to play your favorite songs.'
DOCUMENT3 = "Shifting Gears Your Googlecar has an automatic transmission. To shift gears, simply move the shift lever to the desired position.  Park: This position is used when you are parked. The wheels are locked and the car cannot move. Reverse: This position is used to back up. Neutral: This position is used when you are stopped at a light or in traffic. The car is not in gear and will not move unless you press the gas pedal. Drive: This position is used to drive forward. Low: This position is used for driving in snow or other slippery conditions."

documents = [DOCUMENT1, DOCUMENT2, DOCUMENT3]

# ## Creating the embedding database with ChromaDB
#
# Create a custom function to generate embeddings with the Gemini API. In this task, you are implementing a retrieval system,
# so the `task_type` for generating the *document* embeddings is `retrieval_document`. Later, you will use `retrieval_query`
# for the *query* embeddings.

from chromadb import Documents, EmbeddingFunction, Embeddings
from google.api_core import retry

# Define a helper to retry when per-minute quota is reached.
is_retriable = lambda e: (isinstance(e, genai.errors.APIError) and e.code in {429, 503})


class GeminiEmbeddingFunction(EmbeddingFunction):
    # Specify whether to generate embeddings for documents, or queries
    document_mode = True

    @retry.Retry(predicate=is_retriable)
    def __call__(self, input: Documents) -> Embeddings:
        if self.document_mode:
            embedding_task = "retrieval_document"
        else:
            embedding_task = "retrieval_query"

        response = client.models.embed_content(
            model="models/text-embedding-004",
            contents=input,
            config=types.EmbedContentConfig(
                task_type=embedding_task,
            ),
        )
        return [e.values for e in response.embeddings]


# If ChromaDB with ONNX is causing issues, use sentence-transformers as an alternative
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

# Use SentenceTransformer embeddings directly
import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

DB_NAME = "googlecardb"

embed_fn = SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")

chroma_client = chromadb.Client()
db = chroma_client.get_or_create_collection(name=DB_NAME, embedding_function=embed_fn)

try:
    # db.add(documents=documents, ids=[str(i) for i in range(len(documents))])

    for i in range(len(documents)):
        result = db.add(documents=[documents[i]], ids=[str(i)])
        print(f"Successfully added document {i}")

except Exception as e:
    print("Error adding documents to ChromaDB:", e)

i = "smitty"
# Check document count
count = db.count()
print(f"Added {count} documents to the database")

# ## Retrieval: Find relevant documents
# To search the Chroma database, call the `query` method.
# Note that you also switch to the `retrieval_query` mode of embedding generation.

# If using GeminiEmbeddingFunction, switch to query mode
if isinstance(embed_fn, GeminiEmbeddingFunction):
    embed_fn.document_mode = False

# Search the Chroma DB using the specified query
query = "How do you use the touchscreen to play music?"

result = db.query(query_texts=[query], n_results=1)
[all_passages] = result["documents"]

print("Retrieved passage:")
print(all_passages[0])

# ## Augmented generation: Answer the question
# Now that you have found a relevant passage from the set of documents (the *retrieval* step),
# you can now assemble a generation prompt to have the Gemini API *generate* a final answer.

query_oneline = query.replace("\n", " ")

# This prompt is where you can specify any guidance on tone, or what topics the model should stick to, or avoid.
prompt = f"""You are a helpful and informative bot that answers questions using text from the reference passage included below. 
Be sure to respond in a complete sentence, being comprehensive, including all relevant background information. 
However, you are talking to a non-technical audience, so be sure to break down complicated concepts and 
strike a friendly and converstional tone. If the passage is irrelevant to the answer, you may ignore it.

QUESTION: {query_oneline}
"""

# Add the retrieved documents to the prompt.
for passage in all_passages:
    passage_oneline = passage.replace("\n", " ")
    prompt += f"PASSAGE: {passage_oneline}\n"

print(prompt)

# Use the `generate_content` method to generate an answer to the question
answer = client.models.generate_content(model="gemini-2.0-flash", contents=prompt)

print("\nGenerated Answer:")
print(answer.text)

# ## Next steps
#
# Congrats on building a Retrieval-Augmented Generation app!
#
# To learn more about using embeddings in the Gemini API, check out the [Intro to embeddings](https://ai.google.dev/gemini-api/docs/embeddings)
# or to learn more fundamentals, study the [embeddings chapter](https://developers.google.com/machine-learning/crash-course/embeddings)
# of the Machine Learning Crash Course.
#
# For a hosted RAG system, check out the [Semantic Retrieval service](https://ai.google.dev/gemini-api/docs/semantic_retrieval)
# in the Gemini API. You can implement question-answering on your own documents in a single request, or host a database for even faster responses.
