#!/usr/bin/env python3
# Copyright 2025 Google LLC.
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

# Day 4 - Fine tuning a custom model
#
# In this script, you will use the Gemini API to fine-tune a custom, task-specific model.
# Fine-tuning can be used for a variety of tasks from classic NLP problems like entity
# extraction or summarisation, to creative tasks like stylised generation. You will fine-tune
# a model to classify the category a piece of text (a newsgroup post) into the category
# it belongs to (the newsgroup name).

# Install required packages
# import sys
# import subprocess

# # Install dependencies
# subprocess.run([sys.executable, "-m", "pip", "uninstall", "-qqy", "jupyterlab"])  # Remove unused conflicting packages
# subprocess.run([sys.executable, "-m", "pip", "install", "-U", "-q", "google-genai==1.7.0"])

# Import Google Generative AI client
from dotenv import load_dotenv
from google import genai
from google.genai import types
import os

print(f"Using Google Generative AI SDK version: {genai.__version__}")

# Set up API key
# Note: In a regular Python script, you would need to provide your API key directly or
# load it from an environment variable or config file
# from kaggle_secrets import UserSecretsClient

# GOOGLE_API_KEY = UserSecretsClient().get_secret("GOOGLE_API_KEY")
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
client = genai.Client(api_key=GOOGLE_API_KEY)

# Explore available models
print("Models that support fine-tuning:")
for model in client.models.list():
    if "createTunedModel" in model.supported_actions:
        print(model.name)

# Download the dataset
from sklearn.datasets import fetch_20newsgroups

newsgroups_train = fetch_20newsgroups(subset="train")
newsgroups_test = fetch_20newsgroups(subset="test")

# View list of class names for dataset
print("Newsgroups categories:")
print(newsgroups_train.target_names)

# Print a sample data point
print("Sample newsgroup post:")
print(newsgroups_train.data[0])

# Prepare the dataset
import email
import re
import pandas as pd


def preprocess_newsgroup_row(data):
    # Extract only the subject and body
    msg = email.message_from_string(data)
    text = f"{msg['Subject']}\n\n{msg.get_payload()}"
    # Strip any remaining email addresses
    text = re.sub(r"[\w\.-]+@[\w\.-]+", "", text)
    # Truncate the text to fit within the input limits
    text = text[:40000]

    return text


def preprocess_newsgroup_data(newsgroup_dataset):
    # Put data points into dataframe
    df = pd.DataFrame(
        {"Text": newsgroup_dataset.data, "Label": newsgroup_dataset.target}
    )
    # Clean up the text
    df["Text"] = df["Text"].apply(preprocess_newsgroup_row)
    # Match label to target name index
    df["Class Name"] = df["Label"].map(lambda l: newsgroup_dataset.target_names[l])

    return df


# Apply preprocessing to training and test datasets
df_train = preprocess_newsgroup_data(newsgroups_train)
df_test = preprocess_newsgroup_data(newsgroups_test)

print("Preprocessed training data sample:")
print(df_train.head())


def sample_data(df, num_samples, classes_to_keep):
    # Sample rows, selecting num_samples of each Label.
    df = (
        df.groupby("Label")[df.columns]
        .apply(lambda x: x.sample(num_samples))
        .reset_index(drop=True)
    )

    df = df[df["Class Name"].str.contains(classes_to_keep)]
    df["Class Name"] = df["Class Name"].astype("category")

    return df


TRAIN_NUM_SAMPLES = 50
TEST_NUM_SAMPLES = 10
# Keep rec.* and sci.*
CLASSES_TO_KEEP = "^rec|^sci"

df_train = sample_data(df_train, TRAIN_NUM_SAMPLES, CLASSES_TO_KEEP)
df_test = sample_data(df_test, TEST_NUM_SAMPLES, CLASSES_TO_KEEP)

# Evaluate baseline performance
sample_idx = 0
sample_row = preprocess_newsgroup_row(newsgroups_test.data[sample_idx])
sample_label = newsgroups_test.target_names[newsgroups_test.target[sample_idx]]

print("Sample to evaluate:")
print(sample_row)
print("---")
print("Label:", sample_label)

# Try direct prompt approach
response = client.models.generate_content(
    model="gemini-1.5-flash-001", contents=sample_row
)
print("Direct prompt response:")
print(response.text)

# Ask the model directly in a zero-shot prompt
prompt = "From what newsgroup does the following message originate?"
baseline_response = client.models.generate_content(
    model="gemini-1.5-flash-001", contents=[prompt, sample_row]
)
print("Zero-shot prompt response:")
print(baseline_response.text)

# Use a system instruction for more direct prompting
from google.api_core import retry

system_instruct = """
You are a classification service. You will be passed input that represents
a newsgroup post and you must respond with the newsgroup from which the post
originates.
"""

# Define a helper to retry when per-minute quota is reached.
is_retriable = lambda e: (isinstance(e, genai.errors.APIError) and e.code in {429, 503})


# If you want to evaluate your own technique, replace this body of this function
# with your model, prompt and other code and return the predicted answer.
@retry.Retry(predicate=is_retriable)
def predict_label(post: str) -> str:
    response = client.models.generate_content(
        model="gemini-1.5-flash-001",
        config=types.GenerateContentConfig(system_instruction=system_instruct),
        contents=post,
    )

    rc = response.candidates[0]

    # Any errors, filters, recitation, etc we can mark as a general error
    if rc.finish_reason.name != "STOP":
        return "(error)"
    else:
        # Clean up the response.
        return response.text.strip()


prediction = predict_label(sample_row)

print("System instruction prediction:")
print(prediction)
print()
print("Correct!" if prediction == sample_label else "Incorrect.")

# Run a short evaluation
import tqdm
from tqdm.rich import tqdm as tqdmr
import warnings

# Enable tqdm features on Pandas.
tqdmr.pandas()

# But suppress the experimental warning
warnings.filterwarnings("ignore", category=tqdm.TqdmExperimentalWarning)


# Further sample the test data to be mindful of the free-tier quota.
df_baseline_eval = sample_data(df_test, 2, ".*")

print("Running baseline evaluation...")

# Make predictions using the sampled data.
df_baseline_eval["Prediction"] = df_baseline_eval["Text"].progress_apply(predict_label)

# And calculate the accuracy.
accuracy = (
    df_baseline_eval["Class Name"] == df_baseline_eval["Prediction"]
).sum() / len(df_baseline_eval)
print(f"Baseline accuracy: {accuracy:.2%}")

print("Baseline evaluation results:")
print(df_baseline_eval[["Class Name", "Prediction"]])

# Tune a custom model
from collections.abc import Iterable
import random

# Convert the data frame into a dataset suitable for tuning.
input_data = {
    "examples": df_train[["Text", "Class Name"]]
    .rename(columns={"Text": "textInput", "Class Name": "output"})
    .to_dict(orient="records")
}

# If you are re-running this script, add your model_id here.
model_id = None

# Or try and find a recent tuning job.
if not model_id:
    queued_model = None
    # Newest models first.
    for m in reversed(client.tunings.list()):
        # Only look at newsgroup classification models.
        if m.name.startswith("tunedModels/newsgroup-classification-model"):
            # If there is a completed model, use the first (newest) one.
            if m.state.name == "JOB_STATE_SUCCEEDED":
                model_id = m.name
                print("Found existing tuned model to reuse.")
                break

            elif m.state.name == "JOB_STATE_RUNNING" and not queued_model:
                # If there's a model still queued, remember the most recent one.
                queued_model = m.name
    else:
        if queued_model:
            model_id = queued_model
            print("Found queued model, still waiting.")


# Upload the training data and queue the tuning job.
if not model_id:
    print("Starting new tuning job...")
    tuning_op = client.tunings.tune(
        base_model="models/gemini-1.5-flash-001-tuning",
        training_dataset=input_data,
        config=types.CreateTuningJobConfig(
            tuned_model_display_name="Newsgroup classification model",
            batch_size=16,
            epoch_count=2,
        ),
    )

    print(f"Tuning job state: {tuning_op.state}")
    model_id = tuning_op.name

print(f"Model ID: {model_id}")

# Monitor tuning progress
import datetime
import time

MAX_WAIT = datetime.timedelta(minutes=10)

print("Monitoring tuning job progress...")
while not (tuned_model := client.tunings.get(name=model_id)).has_ended:
    print(f"Current state: {tuned_model.state}")
    time.sleep(60)

    # Don't wait too long. Use a public model if this is going to take a while.
    if (
        datetime.datetime.now(datetime.timezone.utc) - tuned_model.create_time
        > MAX_WAIT
    ):
        print("Taking a shortcut, using a previously prepared model.")
        model_id = "tunedModels/newsgroup-classification-model-ltenbi1b"
        tuned_model = client.tunings.get(name=model_id)
        break


print(f"Done! The model state is: {tuned_model.state.name}")

if not tuned_model.has_succeeded and tuned_model.error:
    print("Error:", tuned_model.error)

# Use the new model
print("Testing the tuned model with new text:")
new_text = """
First-timer looking to get out of here.

Hi, I'm writing about my interest in travelling to the outer limits!

What kind of craft can I buy? What is easiest to access from this 3rd rock?

Let me know how to do that please.
"""

response = client.models.generate_content(model=model_id, contents=new_text)

print("Tuned model response:")
print(response.text)


# Evaluate the tuned model
@retry.Retry(predicate=is_retriable)
def classify_text(text: str) -> str:
    """Classify the provided text into a known newsgroup."""
    response = client.models.generate_content(model=model_id, contents=text)
    rc = response.candidates[0]

    # Any errors, filters, recitation, etc we can mark as a general error
    if rc.finish_reason.name != "STOP":
        return "(error)"
    else:
        return rc.content.parts[0].text


print("Evaluating tuned model...")
# The sampling here is just to minimise your quota usage. If you can, you should
# evaluate the whole test set with `df_model_eval = df_test.copy()`.
df_model_eval = sample_data(df_test, 4, ".*")

df_model_eval["Prediction"] = df_model_eval["Text"].progress_apply(classify_text)

accuracy = (df_model_eval["Class Name"] == df_model_eval["Prediction"]).sum() / len(
    df_model_eval
)
print(f"Tuned model accuracy: {accuracy:.2%}")

# Compare token usage
# Calculate the *input* cost of the baseline model with system instructions.
sysint_tokens = client.models.count_tokens(
    model="gemini-1.5-flash-001", contents=[system_instruct, sample_row]
).total_tokens
print(f"System instructed baseline model: {sysint_tokens} (input)")

# Calculate the input cost of the tuned model.
tuned_tokens = client.models.count_tokens(
    model=tuned_model.base_model, contents=sample_row
).total_tokens
print(f"Tuned model: {tuned_tokens} (input)")

savings = (sysint_tokens - tuned_tokens) / tuned_tokens
print(f"Token savings: {savings:.2%}")  # Note that this is only n=1.

# Compare output tokens
baseline_token_output = baseline_response.usage_metadata.candidates_token_count
print("Baseline (verbose) output tokens:", baseline_token_output)

tuned_model_output = client.models.generate_content(model=model_id, contents=sample_row)
tuned_tokens_output = tuned_model_output.usage_metadata.candidates_token_count
print("Tuned output tokens:", tuned_tokens_output)

print(
    """
Next steps:
- Try tuning a model for other tasks or with different datasets
- Learn about when supervised fine-tuning is most effective: 
  https://cloud.google.com/blog/products/ai-machine-learning/supervised-fine-tuning-for-gemini-llm
- Check out the fine-tuning tutorial for more examples:
  https://ai.google.dev/gemini-api/docs/model-tuning/tutorial?hl=en&lang=python
"""
)
