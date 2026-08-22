import os
import json
import numpy as np
import streamlit as st
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# 1. Load your secret API key from the .env file
load_dotenv()

# 2. Configure the Streamlit webpage
st.set_page_config(page_title="CloudTasker Support", page_icon="☁️")
st.title("CloudTasker Support ☁️")

# 3. Database Setup (Runs once and saves to disk)
@st.cache_resource
def get_embeddings():
    return HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

embeddings = get_embeddings()

@st.cache_resource
def setup_database():
    
    # Initialize the local ChromaDB database
    vectorstore = Chroma(
        collection_name="cloudtasker_faqs", 
        embedding_function=embeddings,
        persist_directory="./chroma_db" # Saves the database to your folder
    )
    
    # If the database is empty, load the JSON file
    if len(vectorstore.get()['ids']) == 0:
        with open("faq_data.json", "r") as f:
            faq_data = json.load(f)
            
        docs = []
        for item in faq_data:
            # We store the question and answer as the searchable text, 
            # and attach the category as a hidden filter (metadata)
            doc_text = f"Question: {item['question']}\nAnswer: {item['answer']}"
            doc = Document(page_content=doc_text, metadata={"category": item['category']})
            docs.append(doc)
            
        vectorstore.add_documents(docs)
        
    return vectorstore

vectorstore = setup_database()

# 4. Set up the "Brain" (Using a free Google Gemma model on OpenRouter)
llm = ChatOpenAI(
    model="openai/gpt-oss-20b", 
    temperature=0,
    api_key=os.environ.get("OPENROUTER_API_KEY"),
    base_url="https://openrouter.ai/api/v1"
)

# 5. Step 1: The Triage Logic (Semantic Routing)
@st.cache_data
def get_category_vectors():
    data = [
        ("hi", "greeting"), ("hello", "greeting"), ("good morning", "greeting"), ("hey buddy", "greeting"), ("what's up", "greeting"),
        ("how do I upgrade to pro?", "billing"), ("credit card declined", "billing"), ("download invoice", "billing"), ("payment issue", "billing"), ("billing issue", "billing"), ("manage payments", "billing"), ("charges", "billing"),
        ("generate API key", "technical"), ("webhook 500 error", "technical"), ("rate limit", "technical"), ("developer portal", "technical"), ("tech issue", "technical"), ("technical error", "technical"), ("server crash", "technical"),
        ("reset password", "account_access"), ("change password", "account_access"), ("update password", "account_access"), ("two factor authentication", "account_access"), ("invite team members", "account_access"), ("login issue", "account_access")
    ]
    phrases = [d[0] for d in data]
    cats = [d[1] for d in data]
    vecs = embeddings.embed_documents(phrases)
    return np.array(vecs), cats

def classify_message(user_message):
    ref_vecs, cats = get_category_vectors()
    query_vec = np.array(embeddings.embed_query(user_message))
    
    # Calculate cosine similarity
    norms = np.linalg.norm(ref_vecs, axis=1) * np.linalg.norm(query_vec)
    similarities = np.dot(ref_vecs, query_vec) / norms
    
    best_idx = np.argmax(similarities)
    best_score = similarities[best_idx]
    
    if best_score < 0.2:
        return "unknown"
        
    return cats[best_idx]

# 6. Step 2: The RAG and Escalation Logic
def generate_answer(user_message, category):
    # Filter the database to ONLY search within the detected category
    retriever = vectorstore.as_retriever(
        search_kwargs={"k": 2, "filter": {"category": category}}
    )
    retrieved_docs = retriever.invoke(user_message)
    
    # Combine the retrieved text into a single string
    context = "\n\n".join([doc.page_content for doc in retrieved_docs])
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a Customer Support Assistant. 
        Rule 1: Answer the user's question using ONLY the provided Retrieved Documents. You may use your reasoning to understand synonyms and user intent (e.g., recognizing that 'change' and 'reset' mean the same thing). However, you must still strictly use ONLY the facts and instructions found in the Retrieved Documents.
        Rule 2: If the documents DO NOT contain the answer, or if the user is asking for something outside this scope, abort and output exactly: ESCALATE: [Reason]
        
        Retrieved Documents:
        {context}"""),
        ("human", "{message}")
    ])
    chain = prompt | llm | StrOutputParser()
    return chain.invoke({"message": user_message, "context": context})

# 7. The Chat UI loop
# Create an empty list to store chat history if it doesn't exist yet
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Hi! I'm the CloudTasker support bot. How can I help you today?"}
    ]

# Display all previous messages on the screen
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Wait for the user to type something and hit Enter
if user_input := st.chat_input("Type your question here..."):
    
    # Show the user's message on screen
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)
        
    # Process the AI's response
    with st.chat_message("assistant"):
        
        # Trigger our semantic routing architecture
        category = classify_message(user_input)
        
        if category == "greeting":
            # Dynamic Greeting Generation
            with st.spinner("Writing reply..."):
                greeting_prompt = ChatPromptTemplate.from_messages([
                    ("system", "You are a friendly customer support bot for CloudTasker. Politely respond to the user's greeting and ask how you can help them today. Be concise."),
                    ("human", "{message}")
                ])
                greeting_chain = greeting_prompt | llm | StrOutputParser()
                final_reply = greeting_chain.invoke({"message": user_input})
            st.markdown(final_reply)
            st.session_state.messages.append({"role": "assistant", "content": final_reply})
        else:
            with st.spinner("Checking our knowledge base..."):
                
                # Fallback if the semantic router gives a weird category
                if category not in ["billing", "technical", "account_access"]:
                    raw_response = "ESCALATE: Could not determine the department."
                else:
                    raw_response = generate_answer(user_input, category)
                
                # Check if the AI decided to escalate
                if "ESCALATE:" in raw_response.upper():
                    upper_response = raw_response.upper()
                    idx = upper_response.find("ESCALATE:")
                    reason = raw_response[idx + len("ESCALATE:"):].strip(" *-\n:")
                    final_reply = f"🚨 **Transferring to Human Agent**\n\n*Reason: {reason}*"
                else:
                    final_reply = f"**[Category: {category.replace('_', ' ').title()}]**\n\n{raw_response}"
                    
                # Display and save the final reply
                st.markdown(final_reply)
                st.session_state.messages.append({"role": "assistant", "content": final_reply})