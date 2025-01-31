import streamlit as st
from langchain.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
import os
import google.generativeai as genai

load_dotenv()
os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

def get_conversational_chain():

    prompt_template = """
    Answer the question as detailed as possible from the provided context, make sure to provide all the details, if the answer is not in
    provided context just say, "answer is not available in the context", don't provide the wrong answer\n\n
    
    **Instructions:**
    
    1. For all the queries being asked from the "Short Description" display all the steps from "Resolution".
    2. If the "Resolution" has multiple steps give the exact answer present in the "Resolution".
    3. If the answer cannot be found in either "Short Description" or "Resolution", state "Answer not found in the provided PDF."
    4. If the input is a common greeting, respond with an appropriate greeting message.
    5. If the question requires information from the "Resolution" of any ticket to be answered correctly:
        - If the "Resolution" section contains steps, provide the steps as part of the answer without summarizing them. 
              **Example:**
              * Step 1: ...
              * Step 2: ...
        - Otherwise, provide the "Resolution" information as is.
    6. Remove any common names (e.g., John, David, Mary) after Hi from the "resolution".
    
    Context:\n {context}?\n
    Question: \n{question}\n

    Answer:
    """
    model = ChatGoogleGenerativeAI(model="gemini-pro",
                                     temperature=0.3)

    prompt = PromptTemplate(template = prompt_template, input_variables = ["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)

    return chain

def user_input(user_question):
    embeddings = GoogleGenerativeAIEmbeddings(model = "models/embedding-001")
    
    new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization = True) 
    docs = new_db.similarity_search(user_question)

    chain = get_conversational_chain()

    
    response = chain(
        {"input_documents":docs, "question": user_question}
        , return_only_outputs=True)

    print(response)
    st.write("Reply: ", response["output_text"])

def main():
    st.set_page_config("Chat PDF")
    st.header("MES Chatbot- Internal")

    user_question = st.text_input("Ask your questions related to MES")

    if user_question:
        user_input(user_question)

if __name__ == "__main__":
    main()