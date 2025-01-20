from langchain_ollama import ChatOllama
import requests
import json
from typing import Annotated
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.tools import tool
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from datetime import datetime
from langgraph.prebuilt import create_react_agent
from langchain_community.document_loaders import YoutubeLoader
class State(TypedDict):
    messages: Annotated[list, add_messages]

current_datetime = datetime.now()
formatted_datetime = current_datetime.strftime("%Y-%m-%d %H:%M:%S")


graph_builder = StateGraph(State)

llmOllama = ChatOllama(
    model="qwen2.5:32b",
    base_url="https://ollama.batnano.fr",
    temperature=6,


)

def invoke_n8n_webhook(method, url, function_name, payload):
    """
    Helper function to make a GET or POST request.

    Args:
        method (str): HTTP method ('GET' or 'POST')
        url (str): The API endpoint
        function_name (str): The name of the tool the AI agent invoked
        payload (dict): The payload for POST requests

    Returns:
        str: The API response in JSON format or an error message
    """
    headers = {
        "Content-Type": "application/json"
    }

    try:
        if method == "GET":
            response = requests.get(url, headers=headers)
        elif method == "POST":
            response = requests.post(url, headers=headers, json=payload)
        else:
            return f"Unsupported method: {method}"

        response.raise_for_status()
        return json.dumps(response.json(), indent=2)
    except Exception as e:
        return f"Exception when calling {function_name}: {e}"

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# ~~~~~~~~~~~~~~~~~~~ n8n AI Agent Tool Functions ~~~~~~~~~~~~~~~~~~~~~~
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~




@tool
def get_youtube_transcript(url:str):
    """
        gets the transcript of a youtube video.

        Args:
            url (str): A Youtube url

        Returns:
            Transcript of the youtube video.
        """
    loader = YoutubeLoader.from_youtube_url(
        url, add_video_info=False,language=["fr","en"]
    )
    return loader.load()

@tool
def check_table_availability(payload:dict):

    """
    Checks the availability of a table for a specified number of guests at a specific time and date.

    Args:
        payload (dict): A dictionary containing the following keys:
            - guests (int): The number of guests for the reservation.
            - time (str): The time of the reservation in the format 'HH:MM'.
            - date (str): The date of the reservation in the format 'YYYY-MM-DD'.

    Returns:
        Response from the webhook call indicating table availability.
    """
    return invoke_n8n_webhook(
        "POST",
        "https://batnano-n8n-4292efb2cdf2.herokuapp.com/webhook/0119b32c-b257-4bcb-a78c-a4016640b844",
        "check_table_availability",
        payload=payload
    )

@tool
def reserve_table(payload:dict):
    """
        Reserves a table for a specified number of guests at a specific time and date for a given name an email and notes.

        Args:
            payload (dict): A dictionary containing the following keys:
                - guests (int): The number of guests for the reservation.
                - time (str): The time of the reservation in the format 'HH:MM'.
                - date (str): The date of the reservation in the format 'YYYY-MM-DD'.
                - notes (str): The special requirements of the customer
                - email (str): The email of the customer
                - name (str): The name to which the reservation is
                - phone (str): The phone number of the customer

        Returns:
            Response from the webhook call indicating if the table was successfully reserved.
        """
    return invoke_n8n_webhook(
        "POST",
        "https://batnano-n8n-a0b56247e135.herokuapp.com/webhook/0119b32c-b257-4bcb-a78c-a4016640b844",
        "reserve_table",
        payload=payload
    )

tool = TavilySearchResults(max_results=2)
tools = [tool, check_table_availability,reserve_table,get_youtube_transcript]
llm = llmOllama
llm_with_tools = llm.bind_tools(tools)


def chatbot(state: State):
    message = llm_with_tools.invoke(state["messages"])
    # Because we will be interrupting during tool execution,
    # we disable parallel tool calling to avoid repeating any
    # tool invocations when we resume.
    assert len(message.tool_calls) <= 1
    return {"messages": [message]}


graph_builder.add_node("chatbot", chatbot)

tool_node = ToolNode(tools=tools)
graph_builder.add_node("tools", tool_node)

graph_builder.add_conditional_edges(
    "chatbot",
    tools_condition,
)
prompt = f"""
current datetime {formatted_datetime}
You are a helpful agent tat makes reservation for the restaurant 'l'imprévu'.
You first check for table availability with the check_table_availability function then you reserve a table with reserve_table function.
You always ask for confirmation before reserving the table
If there is no table available you propose the alternatives but you do not reserve without customer consent
You also summarize youtube videos if a youtube video url is submitted
"""
graph_builder.add_edge("tools", "chatbot")
graph_builder.add_edge(START, "chatbot")
graph = create_react_agent(llm, tools=tools, state_modifier=prompt)