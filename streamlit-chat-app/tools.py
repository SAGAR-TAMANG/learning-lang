from langchain_core.tools import tool
from typing import List, Dict
from vector_space import FlowerShopVectorStore

vector_space = FlowerShopVectorStore()

@tool
def query_knowledge_base(query: str) -> List[Dict[str, str]]:
  """
  Looks up information in a knowledge base to help with answering customer questions and getting information on the basis of business processes.
  
  Args:
    query(str): Question to ask the knowledge base

  Return:
    List[Dict[str, str]]: Potentially relevant question and answer pairs from the knowledge base
  """
  return vector_space.query_faqs(query=query)

@tool
def search_for_product_recommendation(description: str):
  """
  Looks up information in a knowledge base to suggest customers products.

  Args:
    query(str): User's Description of Product

  Return:
    List[Dict[str, str]]: Potentially relevant products.
  """
  return vector_space.query_inventories(query=description)
