from langgraph.graph import StateGraph, MessagesState
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.prebuilt import ToolNode
from tools import query_knowledge_base, search_for_product_recommendation

from dotenv import load_dotenv
import os 

load_dotenv()

graph = StateGraph(MessagesState)

prompt = """
# Purpose
You are a customer service chatbot for a flower shop company. You can help the customers with the goals listed below.

# Goals

1. Answer questions the users might have relating to the services offered. 
2. Recommend products to the user based on their preferences.

# Tone

Helpful, friendly. Use flower emoji and gen-z emojis to keep things lighthearted.
"""

chat_template = ChatPromptTemplate(
  [
    ('system', prompt),
    ('placeholder', "{messages}"),
  ]
)

llm = ChatOpenAI(
  api_key=os.getenv("SUTRA_API_KEY"),
  base_url="https://api.two.ai/v2",
  model="sutra-light",
  streaming=True,
)

tools = [query_knowledge_base, search_for_product_recommendation]

# llm_response = ""
# for chunk in llm.stream(message = "Hello there!"):
#   if chunk.content:
#     llm_response += chunk.content

llm_with_prompt = chat_template | llm.bind_tools(tools)

def call_agent(message_state: MessagesState):
  response = llm_with_prompt.invoke(message_state)
  return {
      "messages": [response]
  }

def is_there_too_calls(state: MessagesState):
  last_message = state['messages'][-1]
  if last_message.tool_calls:
    return 'tool_node'
  else:
    return '__end__'

tool_node = ToolNode(tools)

graph.add_node("agent", call_agent)
graph.add_node("tool_node", tool_node)

graph.add_conditional_edges(
  "agent",
  is_there_too_calls,
)

graph.add_edge("agent", "__end__")
graph.add_edge("tool_node", "agent")

graph.set_entry_point("agent")

app = graph.compile()