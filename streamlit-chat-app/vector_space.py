from chromadb import PersistentClient, EmbeddingFunction, Embeddings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from typing import List
import json

MODEL_NAME = "dunzhang/stella_en_1.5B_v5"
DB_PATH = "./.chromadb"
FAQ_PATH = "./FAQ.json"
INVENTORY_PATH = "./inventory.json"

class Product:
  def __init__(self, name:str, id:str, description:str, type:str, price:float, quantity:int):
    self.name = name
    self.description = description
    self.type = type
    self.price = float
    self.quantity = quantity

class QuestionAnswerPairs:
  def __init__(self, question:str, answer:str):
    self.question = question
    self.ansewr = answer

class CustomEmbeddinClass(EmbeddingFunction):
  def __init__(self, model_name=MODEL_NAME) -> None:
    self.embedding_mode = HuggingFaceEmbedding(MODEL_NAME)
  
  def __call__(self, input_texts: List[str]) -> Embeddings:
    return [self.embedding_mode.get_text_embedding(text) for text in input_texts]

class FlowerShopVectorStore:
  def __init__(self) -> None:
    custom_embedding_function = CustomEmbeddinClass(MODEL_NAME)
    db = PersistentClient(path=DB_PATH)
    self.faq_collections = db.get_or_create_collection(name='FAQ', embedding_function=custom_embedding_function)
    self.inventory_collections = db.get_or_create_collection(name='Inventory', embedding_function=custom_embedding_function)

    if self.faq_collections.count() == 0:
      self._load_faq_collections()
      
    if self.inventory_collections.count() == 0:
      self._load_inventory_collections()

  def _load_faq_collections(self):
    with open(FAQ_PATH, 'r') as f:
      faqs = json.load(f)

    self.faq_collections.add(
      documents=[faq['question'] for faq in faqs] + [faq['answer'] for faq in faqs],
      ids=[str(i) for i in range(0, 2 * len(faqs))],
      metadatas= faqs + faqs,
    )
  
  def _load_inventory_collections(self):
    with open(INVENTORY_PATH, 'r') as f:
      inventories = json.load(f)
    
    self.inventory_collections.add(
      documents=[inventory['description'] for inventory in inventories],
      ids=[str(i) for i in range(0, len(inventories))],
      metadatas = inventories
    )
    
  def query_fqs(self, query:str):
    return self.faq_collections.query(query_texts=[query], n_results=5)
    
  def query_inventories(self, query:str):
    return self.inventory_collections.query(query_texts=[query], n_results=5)