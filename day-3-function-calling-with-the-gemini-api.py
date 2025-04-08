# Import necessary libraries
import sqlite3
from dotenv import load_dotenv
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import textwrap
import asyncio
from pprint import pformat
from IPython.display import display, Image, Markdown
from google import genai
from google.genai import types
import os

# Setup API key
# In a real script, you'd use environment variables or a config file
# GOOGLE_API_KEY = "your_api_key_here"  # Replace with your API key

# Automated retry
from google.api_core import retry

is_retriable = lambda e: (isinstance(e, genai.errors.APIError) and e.code in {429, 503})

if not hasattr(genai.models.Models.generate_content, "__wrapped__"):
    genai.models.Models.generate_content = retry.Retry(predicate=is_retriable)(
        genai.models.Models.generate_content
    )


# Create a local database
def setup_database():
    # Create a new SQLite database
    db_file = "sample.db"
    db_conn = sqlite3.connect(db_file)
    cursor = db_conn.cursor()

    # Create tables
    cursor.execute(
        """
    CREATE TABLE IF NOT EXISTS products (
        product_id INTEGER PRIMARY KEY AUTOINCREMENT,
        product_name VARCHAR(255) NOT NULL,
        price DECIMAL(10, 2) NOT NULL
    );
    """
    )

    cursor.execute(
        """
    CREATE TABLE IF NOT EXISTS staff (
        staff_id INTEGER PRIMARY KEY AUTOINCREMENT,
        first_name VARCHAR(255) NOT NULL,
        last_name VARCHAR(255) NOT NULL
    );
    """
    )

    cursor.execute(
        """
    CREATE TABLE IF NOT EXISTS orders (
        order_id INTEGER PRIMARY KEY AUTOINCREMENT,
        customer_name VARCHAR(255) NOT NULL,
        staff_id INTEGER NOT NULL,
        product_id INTEGER NOT NULL,
        FOREIGN KEY (staff_id) REFERENCES staff (staff_id),
        FOREIGN KEY (product_id) REFERENCES products (product_id)
    );
    """
    )

    # Insert sample data
    cursor.execute(
        """
    INSERT INTO products (product_name, price) VALUES
        ('Laptop', 799.99),
        ('Keyboard', 129.99),
        ('Mouse', 29.99);
    """
    )

    cursor.execute(
        """
    INSERT INTO staff (first_name, last_name) VALUES
        ('Alice', 'Smith'),
        ('Bob', 'Johnson'),
        ('Charlie', 'Williams');
    """
    )

    cursor.execute(
        """
    INSERT INTO orders (customer_name, staff_id, product_id) VALUES
        ('David Lee', 1, 1),
        ('Emily Chen', 2, 2),
        ('Frank Brown', 1, 3);
    """
    )

    # Commit changes and return connection
    db_conn.commit()
    return db_conn


# Define database functions
def list_tables():
    """Retrieve the names of all tables in the database."""
    print(" - DB CALL: list_tables()")

    cursor = db_conn.cursor()

    # Fetch the table names
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")

    tables = cursor.fetchall()
    return [t[0] for t in tables]


def describe_table(table_name: str) -> list[tuple[str, str]]:
    """Look up the table schema.

    Returns:
      List of columns, where each entry is a tuple of (column, type).
    """
    print(f" - DB CALL: describe_table({table_name})")

    cursor = db_conn.cursor()

    cursor.execute(f"PRAGMA table_info({table_name});")

    schema = cursor.fetchall()
    # [column index, column name, column type, ...]
    return [(col[1], col[2]) for col in schema]


def execute_query(sql: str) -> list[list[str]]:
    """Execute an SQL statement, returning the results."""
    print(f" - DB CALL: execute_query({sql})")

    cursor = db_conn.cursor()

    cursor.execute(sql)
    return cursor.fetchall()


async def handle_response(stream, tool_impl=None):
    """Stream output and handle any tool calls during the session."""
    all_responses = []

    async for msg in stream.receive():
        all_responses.append(msg)

        if text := msg.text:
            # Output any text chunks that are streamed back
            if len(all_responses) < 2 or not all_responses[-2].text:
                # Display a header if this is the first text chunk
                print("### Text")

            print(text, end="")

        elif tool_call := msg.tool_call:
            # Handle tool-call requests
            for fc in tool_call.function_calls:
                print("### Tool call")

                # Execute the tool and collect the result to return to the model
                if callable(tool_impl):
                    try:
                        result = tool_impl(**fc.args)
                    except Exception as e:
                        result = str(e)
                else:
                    result = "ok"

                tool_response = types.LiveClientToolResponse(
                    function_responses=[
                        types.FunctionResponse(
                            name=fc.name,
                            id=fc.id,
                            response={"result": result},
                        )
                    ]
                )
                await stream.send(input=tool_response)

        elif msg.server_content and msg.server_content.model_turn:
            # Print any messages showing code the model generated and ran
            for part in msg.server_content.model_turn.parts:
                if code := part.executable_code:
                    print(f"### Code\n```\n{code.code}\n```")

                elif result := part.code_execution_result:
                    print(
                        f"### Result: {result.outcome}\n```\n{pformat(result.output)}\n```"
                    )

                elif img := part.inline_data:
                    display(Image(img.data))

    print()
    return all_responses


# Function for printing chat turns
def print_chat_turns(chat):
    """Prints out each turn in the chat history, including function calls and responses."""
    for event in chat.get_history():
        print(f"{event.role.capitalize()}:")

        for part in event.parts:
            if txt := part.text:
                print(f'  "{txt}"')
            elif fn := part.function_call:
                args = ", ".join(f"{key}={val}" for key, val in fn.args.items())
                print(f"  Function call: {fn.name}({args})")
            elif resp := part.function_response:
                print("  Function response:")
                print(textwrap.indent(str(resp.response["result"]), "    "))

        print()


# Main function to run the example
async def main():
    # Initialize the Gemini API client with your API key
    load_dotenv()
    GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
    client = genai.Client(api_key=GOOGLE_API_KEY)

    # Create/connect to database
    global db_conn
    db_conn = setup_database()

    # Testing the database functions
    print("Testing database functions:")
    print(list_tables())
    print(describe_table("products"))
    print(execute_query("select * from products"))

    # Define the database tools list
    db_tools = [list_tables, describe_table, execute_query]

    # System instruction for the chatbot
    instruction = """You are a helpful chatbot that can interact with an SQL database
    for a computer store. You will take the users questions and turn them into SQL
    queries using the tools available. Once you have the information you need, you will
    answer the user's question using the data returned.
    
    Use list_tables to see what tables are present, describe_table to understand the
    schema, and execute_query to issue an SQL SELECT query."""

    # Create a chat with automatic function calling enabled
    print("\nAsking about the cheapest product:")
    chat = client.chats.create(
        model="gemini-2.0-flash",
        config=types.GenerateContentConfig(
            system_instruction=instruction,
            tools=db_tools,
        ),
    )

    resp = chat.send_message("What is the cheapest product?")
    print(f"\n{resp.text}")

    # Create another chat for a different question
    print("\nAsking about salesperson focus:")
    chat = client.chats.create(
        model="gemini-2.0-flash",
        config=types.GenerateContentConfig(
            system_instruction=instruction,
            tools=db_tools,
        ),
    )

    response = chat.send_message(
        "What products should salesperson Alice focus on to round out her portfolio? Explain why."
    )
    print(f"\n{response.text}")

    # Print the detailed chat history
    print("\nDetailed chat history:")
    print_chat_turns(chat)

    # Compositional function calling with the Live API
    print("\nUsing compositional function calling with Live API:")
    model = "gemini-2.0-flash-exp"
    live_client = genai.Client(
        api_key=GOOGLE_API_KEY, http_options=types.HttpOptions(api_version="v1alpha")
    )

    # Wrap the existing execute_query tool
    execute_query_tool_def = types.FunctionDeclaration.from_callable(
        client=live_client, callable=execute_query
    )

    # System instruction for database interface
    sys_int = """You are a database interface. Use the `execute_query` function
    to answer the users questions by looking up information in the database,
    running any necessary queries and responding to the user.
    
    You need to look up table schema using sqlite3 syntax SQL, then once an
    answer is found be sure to tell the user. If the user is requesting an
    action, you must also execute the actions.
    """

    config = {
        "response_modalities": ["TEXT"],
        "system_instruction": {"parts": [{"text": sys_int}]},
        "tools": [
            {"code_execution": {}},
            {"function_declarations": [execute_query_tool_def.to_json_dict()]},
        ],
    }

    # Example 1: Insert new orders
    print("\nInserting new orders:")
    async with live_client.aio.live.connect(model=model, config=config) as session:
        message = "Please generate and insert 5 new rows in the orders table."
        print(f"> {message}\n")

        await session.send(input=message, end_of_turn=True)
        await handle_response(session, tool_impl=execute_query)

    # Example 2: Analyze and plot orders by staff
    print("\nAnalyzing orders by staff:")
    async with live_client.aio.live.connect(model=model, config=config) as session:
        message = "Can you figure out the number of orders that were made by each of the staff?"
        print(f"> {message}\n")

        await session.send(input=message, end_of_turn=True)
        await handle_response(session, tool_impl=execute_query)

        message = "Generate and run some code to plot this as a python seaborn chart"
        print(f"> {message}\n")

        await session.send(input=message, end_of_turn=True)
        await handle_response(session, tool_impl=execute_query)


# Run the main function
if __name__ == "__main__":
    # Note: In a real script, you would need to setup an asyncio event loop
    asyncio.run(main())
    print("To run this script, you need to:")
    print("1. Set your Google API key")
    print("2. Uncomment the asyncio.run(main()) line")
    print("3. Run the script with Python 3.7+")
