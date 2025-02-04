import os
import streamlit as st
from jira import JIRA
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
#from langchain.vectorstores import FAISS
from langchain_ollama.llms import OllamaLLM
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
import warnings

# Suppress the SSL verification warning
from requests.packages.urllib3.exceptions import InsecureRequestWarning
warnings.simplefilter('ignore', InsecureRequestWarning)

from docx import Document
import pandas as pd
import re
#import exceptions 

from dotenv import load_dotenv
load_dotenv()

#api_key = st.secrets["api_key"]
jira_url = st.secrets["jira_url"]
jira_email = st.secrets["jira_email"]
jira_api_token = st.secrets["jira_api_token"]

# Jira setup
#jira_url = os.getenv('jira_url')
#jira_email = os.getenv('jira_email')
#jira_api_token = os.getenv('jira_api_token')
import certifi
os.environ['REQUESTS_CA_BUNDLE'] = certifi.where()


# Initialize Ollama LLM
ollama_model = OllamaLLM(model="mistral")  # Replace "mistral" with your desired Ollama model

# Load files from the directory
def load_pdf_text(directory):
    text = ""
    for file in os.listdir(directory):
        if file.endswith(".pdf"):
            pdf_reader = PdfReader(os.path.join(directory, file))
            for page in pdf_reader.pages:
                text += page.extract_text() or ""
    return text

def load_docx_text(directory):
    text = ""
    for file in os.listdir(directory):
        if file.endswith(".docx"):
            doc = Document(os.path.join(directory, file))
            for para in doc.paragraphs:
                text += para.text
    return text

def load_excel_text(directory):
    text = ""
    for file in os.listdir(directory):
        if file.endswith(".xlsx"):
            df = pd.read_excel(os.path.join(directory, file))
            text += df.to_string(index=False)
    return text

# Jira Functions
def fetch_jira_story_details(jira, story_key):
    try:
        story = jira.issue(story_key)
        return story.fields.summary, story.fields.description
    except Exception as e:
        st.error(f"Error fetching Jira story details: {e}")
        return None, None

def attach_file_to_jira(jira, story_key, file_path):
    try:
        with open(file_path, "rb") as file:
            jira.add_attachment(issue=story_key, attachment=file)
        st.success(f"File '{file_path}' attached successfully to story {story_key}.")
    except Exception as e:
        st.error(f"Error attaching file to story {story_key}: {e}")

# Text Processing Functions
def get_text_chunks(text):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=900,  # Reduce chunk size slightly to avoid overlap issues
        chunk_overlap=100,
        length_function=len
    )
    return text_splitter.split_text(text)


#from langchain.embeddings import HuggingFaceEmbeddings

# Initialize HuggingFace embeddings
embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")  # Change to any preferred model

def get_vectorstore(text_chunks):
    # Create FAISS vectorstore with HuggingFace embeddings
    vectorstore = FAISS.from_texts(text_chunks, embedding=embedding_model)
    return vectorstore


def create_conversational_chain(vectorstore):
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=ollama_model,
        retriever=vectorstore.as_retriever(),
        memory=memory
    )
    return conversation_chain

def generate_test_cases_with_ollama(conversation_chain, story_description):
    prompt = f"As a QA engineer, please create a few test cases for the following story: {story_description}"
    response = conversation_chain({'question': prompt})
    return response['chat_history'][-1].content

def save_test_cases_to_excel(test_cases, file_name="test_cases.xlsx"):
    columns = ["Name", "Objective", "PreCondition", "Test Steps", "Expected Results", "Status"]
    test_cases_data = []
    
    for i, case in enumerate(test_cases.strip().split("\n\n")):
        case_lines = case.strip().split("\n")
        name = f"Test Case {i+1}"
        objective = ""
        precondition = ""
        test_steps = ""
        expected_results = ""
        status = "Pending"

        for line in case_lines:
            line = line.strip()
            if re.match(r"(?i)^Test steps", line):
                test_steps = line.split(":", 1)[-1].strip()
            elif re.match(r"(?i)^Expected Result", line):
                expected_results = line.split(":", 1)[-1].strip()
            else:
                objective += f"{line} "

        test_cases_data.append([name, objective.strip(), precondition, test_steps, expected_results, status])

    df = pd.DataFrame(test_cases_data, columns=columns)
    df.to_excel(file_name, index=False)
    return file_name

# Streamlit App
def main():
    st.set_page_config(page_title="DART AI Test-case Generator", page_icon=":bulb:")
    directory = os.getcwd()

    st.title("DART AI Test-case Generator")
    
    # Load documents
    with st.spinner("Processing documents on DART..."):
        raw_text = load_pdf_text(directory) + load_docx_text(directory) + load_excel_text(directory)
        text_chunks = get_text_chunks(raw_text)
        vectorstore = get_vectorstore(text_chunks)
        st.session_state.conversation_chain = create_conversational_chain(vectorstore)
        st.success("DART documents processed successfully!")

    # Initialize Jira
    try:
        jira = JIRA(
            server="https://equinixjira.atlassian.net/",
            basic_auth=(jira_email, jira_api_token), options={"verify": False}
        )
        
        st.sidebar.success("Connected to DART NETWORK Jira Board successfully!")
    except Exception as e:
        st.sidebar.error(f"Failed to connect to Jira: {e}")
        return

    # Generate Test Cases
    story_key = st.text_input("Enter the Jira story key:")

    if story_key:
        with st.spinner("Fetching Jira story details..."):
            story_summary, story_description = fetch_jira_story_details(jira, story_key)

        if story_summary and story_description:
            st.write(f"**Summary:** {story_summary}")
            st.write(f"**Description:** {story_description}")

            with st.spinner("Generating test cases..."):
                test_cases = generate_test_cases_with_ollama(st.session_state.conversation_chain, story_description)
                st.subheader("Generated Test Cases")
                st.write(test_cases)

                # Save and attach to Jira
                excel_file_name = save_test_cases_to_excel(test_cases)
                with open(excel_file_name, "rb") as file:
                    st.download_button("Download Test Cases", file, file_name="test_cases.xlsx")

                if st.button("Attach Test Cases to Jira"):
                    attach_file_to_jira(jira, story_key, excel_file_name)

if __name__ == "__main__":
    main()
