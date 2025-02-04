import logging
import pandas as pd
import streamlit as st
from pathlib import Path
from dotenv import load_dotenv
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.callbacks import CallbackManager, StreamingStdOutCallbackHandler
from langchain_core.messages import AIMessage
from langchain_core.prompts import PromptTemplate
from langchain_ollama import OllamaLLM, OllamaEmbeddings

def main():
    load_dotenv()
    ttl = "Chat India"
    st.set_page_config(page_title=ttl, page_icon="flag-in")

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None

    st.title(ttl)

    if "vector" not in st.session_state:
        with st.spinner("wait thora... model is getting finetuned"):
            
            # Load tabular data from CSV
            csv_text = read_csvs()
            # print('CSV::::: ', csv_text)
            
            # Generate text chunks
            text_chunks = get_text_chunks(csv_text)

            # Create vector store
            vectorstore = get_vectorstore(text_chunks)

            # Create conversation chain
            st.session_state.messages = get_conversation(vectorstore)

    user_question = st.chat_input("Now I am ready ask me question")
    if user_question:
        with st.spinner("I am processing it wait " + user_question):
            handle_userinput(user_question)

def read_csvs():
    csv_search = Path("data/").glob("*.csv")
    csv_files = [str(file.absolute()) for file in csv_search]
    csv_text = ""
    for csv in csv_files:
        try:
            df = pd.read_csv(csv)
            csv_text += "\n".join([", ".join([f"{col}: {val}" for col, val in row.items()]) for index, row in df.iterrows()]) + "\n"
        except Exception as e:
            logging.error(f"Error reading CSV {csv}: {e}")
    return csv_text

def get_text_chunks(text):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks

def get_vectorstore(text_chunks):
    embeddings = OllamaEmbeddings(model="nomic-embed-text")
    vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    st.session_state.vector = True
    return vectorstore

def get_conversation(vectorstore):
    callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])
    llm = OllamaLLM(model="llama3.2", temperature=0.4, callbacks=callback_manager)

    prompt = PromptTemplate.from_template(
        """ I am trained on youtube comments having {context}.
        Question: {question}
        Answer
        """
    )

    memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory=memory,
        combine_docs_chain_kwargs={"prompt": prompt}
    )
    return conversation_chain

def handle_userinput(user_question):
    response = st.session_state.messages({'question': user_question})
    st.session_state.chat_history = response['chat_history']
    for i, message in enumerate(st.session_state.chat_history):
        if isinstance(message, AIMessage):
            with st.chat_message("Bot"):
                st.write(message.content)
        else:
            with st.chat_message("myself"):
                st.write(message.content)

if __name__ == '__main__':
    main()
