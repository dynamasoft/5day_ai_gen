# 5-Day Generative AI Intensive Course with Google

This repository contains resources, notebooks, and scripts for the 5-Day Generative AI Intensive Course with Google. The course explores cutting-edge techniques and tools for working with Large Language Models (LLMs), including evaluation, document Q&A, and building agents. Each day focuses on a specific aspect of generative AI, providing hands-on experience and practical insights.

## Directory Structure

```
.env
.gitignore
day-1-evaluation-and-structured-output.ipynb
day-1-prompting.ipynb
day-2-document-q-a-with-rag.ipynb
day-2-python.py
day-3-building-an-agent-with-langgraph.ipynb
day-3-function-calling-with-the-gemini-api.ipynb
day-3-python-langgraph.py
day-4-fine-tuning-a-custom-model.ipynb
day-4-google-search-grounding.ipynb
notebook001fbddaac.ipynb
.vscode/
    launch.json
data/
    gemini.pdf
new_env/
    pyvenv.cfg
    Include/
    Lib/
        site-packages/
    Scripts/
WhitePaper/
    Agents_v8.pdf
    Foundational Large Language models & text generation_v2.pdf
    Solving Domain-Specific problems using LLMs_v7.pdf
    vectorstores_v2.pdf
```

### Key Files and Directories

- **`.env`**: Environment configuration file for managing sensitive variables like API keys.
- **`.gitignore`**: Specifies files and directories to be ignored by Git, including `.env` and `new_env/`.
- **Notebooks**:
  - `day-1-evaluation-and-structured-output.ipynb`: Techniques for evaluating LLM outputs and generating structured data.
  - `day-1-prompting.ipynb`: Introduction to prompting techniques for LLMs.
  - `day-2-document-q-a-with-rag.ipynb`: Implementation of Retrieval-Augmented Generation (RAG) for document-based Q&A.
  - `day-3-building-an-agent-with-langgraph.ipynb`: Building an agent using LangGraph.
  - `day-3-function-calling-with-the-gemini-api.ipynb`: Exploring function calling with the Gemini API.
  - `day-4-fine-tuning-a-custom-model.ipynb`: Fine-tuning a custom model for specific tasks.
  - `day-4-google-search-grounding.ipynb`: Using Google Search for grounding LLM responses.
  - `notebook001fbddaac.ipynb`: Additional notebook with various examples and exercises.
- **Python Scripts**:
  - `day-2-python.py`: Script for Day 2 activities, including RAG implementation using Chroma.
  - `day-3-python-langgraph.py`: Script for Day 3 activities involving LangGraph.
- **`.vscode/`**: Contains Visual Studio Code configuration files.
  - `launch.json`: Debugging configuration for Python scripts.
- **`data/`**: Contains input files, such as `gemini.pdf`, used in the notebooks.
- **`new_env/`**: Virtual environment directory for managing Python dependencies.
- **`WhitePaper/`**: Contains whitepapers on LLMs and related topics.

## Course Overview

### Day 1: Evaluation and Structured Output
- **Introduction**: Learn techniques for evaluating LLM outputs and generating structured data. This day focuses on understanding the quality of model outputs and how to structure them effectively.
- **Key Files**:
  - Notebook: [day-1-evaluation-and-structured-output.ipynb](day-1-evaluation-and-structured-output.ipynb)
  - Notebook: [day-1-prompting.ipynb](day-1-prompting.ipynb)

### Day 2: Document Q&A with RAG
- **Introduction**: Implement Retrieval-Augmented Generation (RAG) to overcome LLM limitations. This day emphasizes using external knowledge sources for document-based Q&A.
- **Key Files**:
  - Notebook: [day-2-document-q-a-with-rag.ipynb](day-2-document-q-a-with-rag.ipynb)
  - Script: [day-2-python.py](day-2-python.py)

### Day 3: Building an Agent with LangGraph
- **Introduction**: Build an agent using LangGraph to handle complex workflows. This day explores creating agents capable of multi-step reasoning and decision-making.
- **Key Files**:
  - Notebook: [day-3-building-an-agent-with-langgraph.ipynb](day-3-building-an-agent-with-langgraph.ipynb)
  - Notebook: [day-3-function-calling-with-the-gemini-api.ipynb](day-3-function-calling-with-the-gemini-api.ipynb)
  - Script: [day-3-python-langgraph.py](day-3-python-langgraph.py)

### Day 4: Fine-Tuning and Grounding
- **Introduction**: Fine-tune a custom model for specific tasks and use Google Search for grounding LLM responses. This day focuses on enhancing model performance and reliability.
- **Key Files**:
  - Notebook: [day-4-fine-tuning-a-custom-model.ipynb](day-4-fine-tuning-a-custom-model.ipynb)
  - Notebook: [day-4-google-search-grounding.ipynb](day-4-google-search-grounding.ipynb)

### Day 5: LLM Ops with Google Vertex AI
- **Introduction**: Explore operationalizing LLMs using Google Vertex AI. This day focuses on deploying, monitoring, and managing LLMs in production environments.
- **Key Files**:
  - Notebook: [day-5-llm-ops-with-google-vertex-ai.ipynb](day-5-llm-ops-with-google-vertex-ai.ipynb)
  - Script: [day-5-llm-ops-with-google-vertex-ai.py](day-5-llm-ops-with-google-vertex-ai.py)

## Getting Started

1. **Set up the virtual environment**:
   - Navigate to the `new_env/` directory.
   - Activate the virtual environment:
     - On Windows:
       ```sh
       new_env\Scripts\activate.bat
       ```
     - On macOS/Linux:
       ```sh
       source new_env/Scripts/activate
       ```

2. **Install dependencies**:
   - Ensure the virtual environment is activated, then run:
     ```sh
     pip install -r requirements.txt
     ```
   *(Note: Add a `requirements.txt` file if not already present.)*

3. **Run notebooks**:
   - Open the `.ipynb` files in Jupyter Notebook or Visual Studio Code.

4. **Explore whitepapers**:
   - Refer to the documents in the `WhitePaper/` directory for insights into LLMs and agent-building.

## Notes

- Ensure the virtual environment is activated before running any Python scripts or notebooks.
- The `.env` file should contain your API keys and other sensitive information.
- The `.gitignore` file ensures that sensitive files and the virtual environment are not tracked by Git.

## Resources

- **Kaggle Notebooks**:
  - [Day 1 - Prompting](https://www.kaggle.com/code/markishere/day-1-prompting)
  - [Day 1 - Evaluation and Structured Output](https://www.kaggle.com/code/markishere/day-1-evaluation-and-structured-output)
  - [Day 0 - Troubleshooting and FAQs](https://www.kaggle.com/code/markishere/day-0-troubleshooting-and-faqs)
- **Whitepapers**:
  - [Evaluating Large Language Models](https://services.google.com/fh/files/blogs/neurips_evaluation.pdf)
  - [Gemini API Documentation](https://ai.google.dev/gemini-api/docs)

## License

This project is licensed under the Apache License 2.0. See the [LICENSE](https://www.apache.org/licenses/LICENSE-2.0) file for details.

## Acknowledgments

- The whitepapers in the `WhitePaper/` directory provide valuable insights into solving domain-specific problems using LLMs.
- This workspace is designed for educational and experimental purposes.