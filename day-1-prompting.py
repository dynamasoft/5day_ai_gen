#!/usr/bin/env python
# -*- coding: utf-8 -*-

# @title Licensed under the Apache License, Version 2.0 (the "License");
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

# Install necessary packages
# !pip uninstall -qqy jupyterlab
# !pip install -U -q "google-genai==1.7.0"

from google import genai
from google.genai import types
from google.api_core import retry
from IPython.display import HTML, Markdown, display
import enum
import io
import typing_extensions as typing
from pprint import pprint
from dotenv import load_dotenv
import os

# Set up API key - in a Python script you should use environment variables
# or a secure configuration method instead of the Kaggle secrets method
# shown in the notebook
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# Set up a retry helper
is_retriable = lambda e: (isinstance(e, genai.errors.APIError) and e.code in {429, 503})
genai.models.Models.generate_content = retry.Retry(predicate=is_retriable)(
    genai.models.Models.generate_content
)

# Initialize the client
client = genai.Client(api_key=GOOGLE_API_KEY)


# Run your first prompt
def run_first_prompt():
    response = client.models.generate_content(
        model="gemini-2.0-flash", contents="Explain AI to me like I'm a kid."
    )
    print(response.text)


# Start a chat
def start_chat():
    chat = client.chats.create(model="gemini-2.0-flash", history=[])
    response = chat.send_message("Hello! My name is Zlork.")
    print(response.text)

    response = chat.send_message(
        "Can you tell me something interesting about dinosaurs?"
    )
    print(response.text)

    response = chat.send_message("Do you remember what my name is?")
    print(response.text)


# List available models
def list_models():
    for model in client.models.list():
        print(model.name)

    # Get detailed information about a specific model
    for model in client.models.list():
        if model.name == "models/gemini-2.0-flash":
            pprint(model.to_json_dict())
            break


# Explore generation parameters - Output length
def explore_output_length():
    short_config = types.GenerateContentConfig(max_output_tokens=200)

    response = client.models.generate_content(
        model="gemini-2.0-flash",
        config=short_config,
        contents="Write a 1000 word essay on the importance of olives in modern society.",
    )
    print(response.text)

    response = client.models.generate_content(
        model="gemini-2.0-flash",
        config=short_config,
        contents="Write a short poem on the importance of olives in modern society.",
    )
    print(response.text)


# Explore generation parameters - Temperature
def explore_temperature():
    high_temp_config = types.GenerateContentConfig(temperature=2.0)

    print("High temperature (2.0) results:")
    for _ in range(5):
        response = client.models.generate_content(
            model="gemini-2.0-flash",
            config=high_temp_config,
            contents="Pick a random colour... (respond in a single word)",
        )
        if response.text:
            print(response.text, "-" * 25)

    low_temp_config = types.GenerateContentConfig(temperature=0.0)

    print("\nLow temperature (0.0) results:")
    for _ in range(5):
        response = client.models.generate_content(
            model="gemini-2.0-flash",
            config=low_temp_config,
            contents="Pick a random colour... (respond in a single word)",
        )
        if response.text:
            print(response.text, "-" * 25)


# Explore generation parameters - Top-P
def explore_top_p():
    model_config = types.GenerateContentConfig(
        # These are the default values for gemini-2.0-flash.
        temperature=1.0,
        top_p=0.95,
    )

    story_prompt = "You are a creative writer. Write a short story about a cat who goes on an adventure."
    response = client.models.generate_content(
        model="gemini-2.0-flash", config=model_config, contents=story_prompt
    )
    print(response.text)


# Zero-shot prompting
def zero_shot_prompting():
    model_config = types.GenerateContentConfig(
        temperature=0.1,
        top_p=1,
        max_output_tokens=5,
    )

    zero_shot_prompt = """Classify movie reviews as POSITIVE, NEUTRAL or NEGATIVE.
    Review: "Her" is a disturbing study revealing the direction
    humanity is headed if AI is allowed to keep evolving,
    unchecked. I wish there were more movies like this masterpiece.
    Sentiment: """

    response = client.models.generate_content(
        model="gemini-2.0-flash", config=model_config, contents=zero_shot_prompt
    )
    print(response.text)

    # Enum mode
    class Sentiment(enum.Enum):
        POSITIVE = "positive"
        NEUTRAL = "neutral"
        NEGATIVE = "negative"

    response = client.models.generate_content(
        model="gemini-2.0-flash",
        config=types.GenerateContentConfig(
            response_mime_type="text/x.enum", response_schema=Sentiment
        ),
        contents=zero_shot_prompt,
    )
    print(response.text)

    enum_response = response.parsed
    print(enum_response)
    print(type(enum_response))


# One-shot and few-shot prompting
def few_shot_prompting():
    few_shot_prompt = """Parse a customer's pizza order into valid JSON:

    EXAMPLE:
    I want a small pizza with cheese, tomato sauce, and pepperoni.
    JSON Response:
    ```
    {
    "size": "small",
    "type": "normal",
    "ingredients": ["cheese", "tomato sauce", "pepperoni"]
    }
    ```

    EXAMPLE:
    Can I get a large pizza with tomato sauce, basil and mozzarella
    JSON Response:
    ```
    {
    "size": "large",
    "type": "normal",
    "ingredients": ["tomato sauce", "basil", "mozzarella"]
    }
    ```

    ORDER:
    """

    customer_order = "Give me a large with cheese & pineapple"

    response = client.models.generate_content(
        model="gemini-2.0-flash",
        config=types.GenerateContentConfig(
            temperature=0.1,
            top_p=1,
            max_output_tokens=250,
        ),
        contents=[few_shot_prompt, customer_order],
    )
    print(response.text)

    # JSON mode
    class PizzaOrder(typing.TypedDict):
        size: str
        ingredients: list[str]
        type: str

    response = client.models.generate_content(
        model="gemini-2.0-flash",
        config=types.GenerateContentConfig(
            temperature=0.1,
            response_mime_type="application/json",
            response_schema=PizzaOrder,
        ),
        contents="Can I have a large dessert pizza with apple and chocolate",
    )
    print(response.text)


# Chain of Thought (CoT)
def chain_of_thought():
    prompt = """When I was 4 years old, my partner was 3 times my age. Now, I
    am 20 years old. How old is my partner? Return the answer directly."""

    response = client.models.generate_content(model="gemini-2.0-flash", contents=prompt)
    print("Direct answer:")
    print(response.text)

    prompt = """When I was 4 years old, my partner was 3 times my age. Now,
    I am 20 years old. How old is my partner? Let's think step by step."""

    response = client.models.generate_content(model="gemini-2.0-flash", contents=prompt)
    print("\nStep-by-step reasoning:")
    print(response.text)


# ReAct: Reason and act
def react_method():
    model_instructions = """
    Solve a question answering task with interleaving Thought, Action, Observation steps. Thought can reason about the current situation,
    Observation is understanding relevant information from an Action's output and Action can be one of three types:
     (1) <search>entity</search>, which searches the exact entity on Wikipedia and returns the first paragraph if it exists. If not, it
         will return some similar entities to search and you can try to search the information from those topics.
     (2) <lookup>keyword</lookup>, which returns the next sentence containing keyword in the current context. This only does exact matches,
         so keep your searches short.
     (3) <finish>answer</finish>, which returns the answer and finishes the task.
    """

    example1 = """Question
    Musician and satirist Allie Goertz wrote a song about the "The Simpsons" character Milhouse, who Matt Groening named after who?
    
    Thought 1
    The question simplifies to "The Simpsons" character Milhouse is named after who. I only need to search Milhouse and find who it is named after.
    
    Action 1
    <search>Milhouse</search>
    
    Observation 1
    Milhouse Mussolini Van Houten is a recurring character in the Fox animated television series The Simpsons voiced by Pamela Hayden and created by Matt Groening.
    
    Thought 2
    The paragraph does not tell who Milhouse is named after, maybe I can look up "named after".
    
    Action 2
    <lookup>named after</lookup>
    
    Observation 2
    Milhouse was named after U.S. president Richard Nixon, whose middle name was Milhous.
    
    Thought 3
    Milhouse was named after U.S. president Richard Nixon, so the answer is Richard Nixon.
    
    Action 3
    <finish>Richard Nixon</finish>
    """

    example2 = """Question
    What is the elevation range for the area that the eastern sector of the Colorado orogeny extends into?
    
    Thought 1
    I need to search Colorado orogeny, find the area that the eastern sector of the Colorado orogeny extends into, then find the elevation range of the area.
    
    Action 1
    <search>Colorado orogeny</search>
    
    Observation 1
    The Colorado orogeny was an episode of mountain building (an orogeny) in Colorado and surrounding areas.
    
    Thought 2
    It does not mention the eastern sector. So I need to look up eastern sector.
    
    Action 2
    <lookup>eastern sector</lookup>
    
    Observation 2
    The eastern sector extends into the High Plains and is called the Central Plains orogeny.
    
    Thought 3
    The eastern sector of Colorado orogeny extends into the High Plains. So I need to search High Plains and find its elevation range.
    
    Action 3
    <search>High Plains</search>
    
    Observation 3
    High Plains refers to one of two distinct land regions
    
    Thought 4
    I need to instead search High Plains (United States).
    
    Action 4
    <search>High Plains (United States)</search>
    
    Observation 4
    The High Plains are a subregion of the Great Plains. From east to west, the High Plains rise in elevation from around 1,800 to 7,000 ft (550 to 2,130m).
    
    Thought 5
    High Plains rise in elevation from around 1,800 to 7,000 ft, so the answer is 1,800 to 7,000 ft.
    
    Action 5
    <finish>1,800 to 7,000 ft</finish>
    """

    question = """Question
    Who was the youngest author listed on the transformers NLP paper?
    """

    # You will perform the Action; so generate up to, but not including, the Observation.
    react_config = types.GenerateContentConfig(
        # stop_sequences=["\nObservation"],
        system_instruction=model_instructions
        + example1
        + example2,
    )

    # Create a chat that has the model instructions and examples pre-seeded.
    react_chat = client.chats.create(
        model="gemini-2.0-flash",
        config=react_config,
    )

    resp = react_chat.send_message(question)
    print("Initial response:")
    print(resp.text)

    observation = """Observation 1
    [1706.03762] Attention Is All You Need
    Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Lukasz Kaiser, Illia Polosukhin
    We propose a new simple network architecture, the Transformer, based solely on attention mechanisms, dispensing with recurrence and convolutions entirely.
    """
    resp = react_chat.send_message(observation)
    print("\nResponse after providing observation:")
    print(resp.text)


# Thinking mode
def thinking_mode():
    print("Using thinking mode model:")
    response = client.models.generate_content_stream(
        model="gemini-2.0-flash-thinking-exp",
        contents="Who was the youngest author listed on the transformers NLP paper?",
    )

    buf = io.StringIO()
    for chunk in response:
        buf.write(chunk.text)
        # Display the response as it is streamed
        print(chunk.text, end="")

    print("\n\nFull response:")
    print(buf.getvalue())


# Generating code
def generate_code():
    # The Gemini models love to talk, so it helps to specify they stick to the code if that
    # is all that you want.
    code_prompt = """
    Write a Python function to calculate the factorial of a number. No explanation, provide only the code.
    """

    response = client.models.generate_content(
        model="gemini-2.0-flash",
        config=types.GenerateContentConfig(
            temperature=1,
            top_p=1,
            max_output_tokens=1024,
        ),
        contents=code_prompt,
    )
    print(response.text)


# Code execution
def code_execution():
    config = types.GenerateContentConfig(
        tools=[types.Tool(code_execution=types.ToolCodeExecution())],
    )

    code_exec_prompt = """
    Generate the first 14 odd prime numbers, then calculate their sum.
    """

    response = client.models.generate_content(
        model="gemini-2.0-flash", config=config, contents=code_exec_prompt
    )

    for part in response.candidates[0].content.parts:
        pprint(part.to_json_dict())
        print("-----")

    print("\nFormatted response:")
    for part in response.candidates[0].content.parts:
        if part.text:
            print(f"TEXT: {part.text}")
        elif part.executable_code:
            print(f"CODE:\n{part.executable_code.code}")
        elif part.code_execution_result:
            if part.code_execution_result.outcome != "OUTCOME_OK":
                print(f"STATUS: {part.code_execution_result.outcome}")
            print(f"OUTPUT:\n{part.code_execution_result.output}")


# Explaining code
def explain_code():
    # In a script you would probably read from a file instead of using curl
    # file_contents = !curl https://raw.githubusercontent.com/magicmonty/bash-git-prompt/refs/heads/master/gitprompt.sh
    # For the script, we'll use a simple example instead
    file_contents = """
    #!/bin/bash
    # A simple bash script to demonstrate
    echo "Hello, world!"
    """

    explain_prompt = f"""
    Please explain what this file does at a very high level. What is it, and why would I use it?
    
    ```
    {file_contents}
    ```
    """

    response = client.models.generate_content(
        model="gemini-2.0-flash", contents=explain_prompt
    )
    print(response.text)


def main():
    print("Welcome to the Gemini API examples!")
    print("Uncomment the functions you want to run in the main function.")

    # Uncomment the functions you want to run
    # run_first_prompt()
    # start_chat()
    # list_models()
    # explore_output_length()
    # explore_temperature()
    # explore_top_p()
    # zero_shot_prompting()
    # few_shot_prompting()
    # chain_of_thought()
    # react_method()
    # thinking_mode()
    # generate_code()
    # code_execution()
    explain_code()


if __name__ == "__main__":
    main()
