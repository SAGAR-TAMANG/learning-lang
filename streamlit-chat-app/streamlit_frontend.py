import streamlit as st
from vector_space import FlowerShopVectorStore
from chatbot import app
from langchain_core.messages import AIMessage, HumanMessage

st.set_page_config(layout='wide', page_title="Flow Shop Chat Bot", page_icon='💐')

if 'message_history' not in st.session_state:
  st.session_state.message_history = [AIMessage(content="Hey, I am the Flower Shop Chatbot, How can I help?")]

left_col, main_col, right_col = st.columns([1, 2, 1])
# 1. Button for Clear

with left_col:
  if st.button('Clear Chat'):
    st.session_state.message_history = []

  # collection_choice = st.radio('Which collection?', ['faq', 'inventory'])

# 2. chat history & input

with main_col:
  user_input = st.chat_input("Type here...")

  if user_input:
    # if collection_choice == 'faq':
    #   related_questions = vector_store.query_fqs(user_input)
    # else:
    #   related_questions = vector_store.query_inventories(user_input)
    st.session_state.message_history.append(HumanMessage(content=user_input))
    
    response = app.invoke({
      "messages": st.session_state.message_history
    })

    st.session_state.message_history = response["messages"]

  for i in range(1, len(st.session_state.message_history) + 1):
    this_message = st.session_state.message_history[-i]
    if isinstance(this_message, AIMessage):
      message_box = st.chat_message("assistant")
    else:
      message_box = st.chat_message("user")
    message_box.markdown(this_message.content)

# 3. State variables
with right_col:
  st.text(st.session_state.message_history)