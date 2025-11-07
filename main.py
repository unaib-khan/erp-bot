import os
import io
import hashlib
import logging
import json
from concurrent.futures import ThreadPoolExecutor
from PyPDF2 import PdfReader
from PIL import Image
import pdfplumber
import fitz  # PyMuPDF
import streamlit as st
from dotenv import load_dotenv
from langchain.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
from langchain.chains.question_answering import load_qa_chain
from langchain_ollama import OllamaLLM
import google.generativeai as genai
from langchain_experimental.text_splitter import SemanticChunker
from langchain.memory import ConversationBufferMemory
from langchain.chains import LLMChain
from langchain_community.embeddings import HuggingFaceEmbeddings


# ==========================
# Setup & Configuration
# ==========================
load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")
if api_key:
    genai.configure(api_key=api_key)
else:
    st.error("GOOGLE_API_KEY is missing! Add it to your .env file.")
    st.stop()

logging.basicConfig(level=logging.INFO)

UPLOAD_FOLDER = "uploaded_files"
VECTOR_STORE_FOLDER = "vector_stores"
METADATA_FILE = "metadata.json"

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(VECTOR_STORE_FOLDER, exist_ok=True)

# Load metadata from file if available
if os.path.exists(METADATA_FILE):
    with open(METADATA_FILE, "r") as f:
        processed_files = json.load(f)
else:
    processed_files = {}

# ==========================
# Utility Functions
# ==========================
def compute_hash(file_path):
    with open(file_path, "rb") as f:
        file_hash = hashlib.sha256()
        while chunk := f.read(8192):
            file_hash.update(chunk)
        return file_hash.hexdigest()


def extract_text_with_pdfplumber(pdf_path):
    extracted_text = ""
    with pdfplumber.open(pdf_path) as pdf:
        for page_number, page in enumerate(pdf.pages, start=1):
            text = page.extract_text() or ""
            extracted_text += f"\n--- Page {page_number} ---\n{text}"
    return extracted_text.strip()


def extract_text_from_pdf(pdf_path, enable_ocr=False, ocr_tool="fitz"):
    combined_text = ""
    if ocr_tool == "pdfplumber":
        return extract_text_with_pdfplumber(pdf_path)
    elif ocr_tool == "fitz":
        doc = fitz.open(pdf_path)
        for page_number in range(len(doc)):
            page = doc[page_number]
            text = page.get_text()
            combined_text += f"\n--- Page {page_number + 1} ---\n{text}"
        return combined_text.strip()
    else:
        reader = PdfReader(pdf_path)
        for page_number, page in enumerate(reader.pages, start=1):
            text = page.extract_text() or ""
            combined_text += f"\n--- Page {page_number} ---\n{text}"
        return combined_text.strip()


def save_metadata(metadata):
    with open(METADATA_FILE, "w") as f:
        json.dump(metadata, f, indent=4)


def process_pdf(uploaded_file, tag, enable_ocr, ocr_tool):
    tag_folder = os.path.join(UPLOAD_FOLDER, tag)
    os.makedirs(tag_folder, exist_ok=True)

    file_path = os.path.join(tag_folder, uploaded_file.name)
    with open(file_path, "wb") as f:
        f.write(uploaded_file.read())

    file_hash = compute_hash(file_path)
    if file_hash in processed_files:
        st.warning(f"Duplicate file detected: {uploaded_file.name}")
        return None

    text = extract_text_from_pdf(file_path, enable_ocr=enable_ocr, ocr_tool=ocr_tool)
    CHUNKING_STRATEGY = "semantic"
    if CHUNKING_STRATEGY == "semantic":
        # embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        text_splitter = SemanticChunker(
        embeddings,
        breakpoint_threshold_type="percentile",   # could also use "standard_deviation"
        breakpoint_threshold_amount=95            # higher = fewer, larger chunks
    )
    else:
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)



    # text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    chunks = text_splitter.split_text(text)

    # embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.from_texts(chunks, embedding=embeddings)
    vector_store_path = os.path.join(VECTOR_STORE_FOLDER, f"{file_hash}.faiss")
    vector_store.save_local(vector_store_path)

    metadata_entry = {
        "name": uploaded_file.name,
        "path": file_path,
        "vector_store_path": vector_store_path,
        "text": text,
        "tag": tag,  # Include tag in metadata
        "chunks": chunks
    }
    processed_files[file_hash] = metadata_entry
    save_metadata(processed_files)
    return metadata_entry


def delete_tag(tag_to_delete):
    global processed_files
    processed_files = {k: v for k, v in processed_files.items() if v.get("tag") != tag_to_delete}
    save_metadata(processed_files)
    st.rerun()


def delete_file(file_to_delete):
    global processed_files
    processed_files = {k: v for k, v in processed_files.items() if v.get("name") != file_to_delete}
    save_metadata(processed_files)
    st.rerun()


def display_tags_with_delete():
    st.subheader("Tags")
    tag_list = list(set(metadata.get("tag", "Untitled") for metadata in processed_files.values()))
    for tag in tag_list:
        col1, col2 = st.columns([4, 1])
        col1.write(tag)
        if col2.button("❌", key=f"delete_tag_{tag}"):
            delete_tag(tag)


def display_files_with_delete(selected_tag):
    st.subheader(f"Files in Tag: {selected_tag}")
    files_in_tag = [f for f in processed_files.values() if f.get("tag") == selected_tag]
    for file_metadata in files_in_tag:
        col1, col2 = st.columns([4, 1])
        col1.write(file_metadata["name"])
        if col2.button("❌", key=f"delete_file_{file_metadata['name']}"):
            delete_file(file_metadata["name"])


def ask_question_with_model(question, context, model_choice, vector_store_path, memory, capabilities="Leave Management, Attendance, Payroll, Performance,Recruitment Workflow System (RWS) Functional & Technical Specification"):
    vector_store = FAISS.load_local(vector_store_path, HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2"), allow_dangerous_deserialization=True)
    docs = vector_store.similarity_search(question,k=8)
    combined_context = "\n\n".join([d.page_content for d in docs])
    show_greeting = len(memory.chat_memory.messages) == 0
    greeting_text = "Start with a warm greeting." if show_greeting else "Continue naturally without greeting."
    prompt = """
    You are **VOCTRAM**, a friendly, intelligent, and conversational **ERP assistant** created by the **Voctrum IT Team**.

    {greeting_instruction}

    Your job is to help users understand and use the ERP system based strictly on the provided official internal documentation.  
    You also maintain a warm, natural tone during casual conversation.

    ---

    ## 🎯 Core Behavioral Logic

    1. **Dual-mode awareness**  
    You can detect whether a user’s message is:
    - **ERP-related** → e.g., questions about Leave, Attendance, Payroll, Performance, or RWS.  
        → Use the official documentation context to answer.  
    - **General or personal** (like “How are you?”, “Who made you?”, “What’s your name?”)  
        → Respond naturally as a friendly assistant, *without referencing ERP documentation.*

    Examples of casual replies:
    - “I’m doing great and ready to help you 😊”
    - “I was developed by the Voctrum IT Team to assist you with our ERP system. How can I help today?”
    - “I don’t have feelings, but I’m always here and happy to help!”

    ⚠️ Never say “I couldn’t find that in the official ERP documentation” for casual or personal questions.

    ---

    ## 🧭 ERP Behavior Rules

    2. **When the question is ERP-related**:
    - Use **only** the content in `{context}` to answer.
    - Never use external or internet information.
    - Never hallucinate.
    - If the requested info is not in the documentation, reply exactly:
        > **"I couldn’t find that in the official ERP documentation."**

    3. **When explaining content**:
    - Adjust your depth:
        - If the user asks *“Explain”*, *“How”*, or *“What does this mean”*, → **elaborate clearly**.
        - If the user asks *“Give me an overview”* or *“Summarize”*, → **give a concise, high-level answer**.
    - Stay factual and easy to follow.
    - Your goal is to make the content clear and readable — not shorter or longer than necessary.

    4. **Preserve and format structure**:
    - If the document uses numbering, letters, or hierarchy (A., B., C., 1., 2., 3.), **maintain that structure**.
    - Use indentation, bullets, and headings for clarity.
    - Example:
        ```
        A. Main Module
            - Description
        B. Submodule
            - Key Fields
        ```
    - Never flatten or remove headings from the source.

    ---

    ## 💬 Personality & Tone

    5. **Be friendly and human-like**, not robotic.  
    Use short natural expressions:
    - “Let’s walk through it step by step 😊”
    - “No worries — I’ll explain it clearly.”

    6. **Greeting logic:**
    - On the first user message, give a warm short greeting.
    - After that, continue naturally without repeating a greeting.

    7. **Scope reminder**  
    If the user asks something outside the ERP domain, say politely:
    > “I’m sorry, I can only assist with:
    > - Leave Management  
    > - Attendance  
    > - Payroll  
    > - Performance  
    > - Recruitment Workflow System (RWS) Functional & Technical Specification.”

    ---

    ## 🧩 Example Situations

    - **User:** “Who created you?”  
    **You:** “I was developed by the Voctrum IT Team to help you understand and use the ERP system. How can I assist you today?”

    - **User:** “How are you?”  
    **You:** “I’m doing great and ready to assist you 😊 What can I help you with today?”

    - **User:** “What is the attendance module?”  
    **You:** (Use `{context}`) “The Attendance Module tracks employee working hours...”

    - **User:** “Tell me about Payroll.”  
    **You:** (Use `{context}`) “Payroll processes salary computation, deductions, and disbursement...”

    ---

    ## ⚖️ Rules You Must Always Follow

    - Never use external or internet data.  
    - Never hallucinate.  
    - Only use the `{context}` for ERP-related information.  
    - If relevant info is missing, reply:  
    > **"I couldn’t find that in the official ERP documentation."**  
    - Preserve structure, order, and clarity.  
    - Be friendly, concise, and professional.

    ---

    ## 📂 Context (Official ERP Documentation)
    {context}

    ## 💬 Chat History
    {chat_history}

    ## 👤 User Question
    {question}

    ---

    ### 💡 Friendly, Conversational Answer
    """

    prompt = PromptTemplate(template=prompt, input_variables=["context", "question", "chat_history", "greeting_instruction"])
    model = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.3)
        # === Memory (Persistent per session) ===
    # if "chat_memory" not in st.session_state:
    #     st.session_state.chat_memory = ConversationBufferMemory(
    #         memory_key="chat_history",
    #         input_key="question",
    #         return_messages=True
    #     )

    memory = st.session_state.chat_memory

    # === LLM Chain ===
    chain = LLMChain(
        llm=model,
        prompt=prompt,
        memory=memory,
        verbose=False
    )

    # === Run and Return Answer ===
    response = chain.invoke({
        "context": combined_context,
        "question": question,
        "chat_history": memory.chat_memory.messages,
        "greeting_instruction": greeting_text
    })
    return response["text"]

    # chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
    # response = chain({"input_documents": docs, "question": question}, return_only_outputs=True)
    # # Collect the context text (joined from docs)
    # # context_text = "\n\n".join([d.page_content for d in docs])

    


# ==========================
# Streamlit Application
# ==========================
# def main():
#     st.set_page_config(page_title="RAG Application")
#     st.header("📘 ERP Chatbot")

#     # ✅ 1️⃣ Initialize chat memory ONCE
#     if "chat_memory" not in st.session_state:
#         st.session_state.chat_memory = ConversationBufferMemory(
#             memory_key="chat_history",
#             input_key="question",
#             return_messages=True
#         )

#     # --- Sidebar UI ---
#     uploaded_files = st.sidebar.file_uploader("Upload PDFs", type=["pdf"], accept_multiple_files=True)
#     tag = st.sidebar.text_input("Enter a tag:")
#     enable_ocr = st.sidebar.checkbox("Enable OCR")
#     ocr_tool = st.sidebar.radio("Choose OCR Tool", ["fitz", "pdfplumber"])
#     model_choice = st.sidebar.selectbox("Choose LLM Model", ["Gemini", "Mistral"])
#     process_button = st.sidebar.button("Process Files")

#     # --- File processing ---
#     if process_button and uploaded_files and tag:
#         with st.spinner("Processing files..."):
#             with ThreadPoolExecutor() as executor:
#                 for uploaded_file in uploaded_files:
#                     result = executor.submit(process_pdf, uploaded_file, tag, enable_ocr, ocr_tool).result()
#                     if result:
#                         st.success(f"✅ Processed: {uploaded_file.name}")
#                     else:
#                         st.error(f"❌ Failed to process: {uploaded_file.name}")

#     # --- Show processed files ---
#     if processed_files:
#         display_tags_with_delete()
        
#         tag_list = list(set(metadata.get("tag", "Untitled") for metadata in processed_files.values()))
#         tag_choice = st.selectbox("Select a Tag to View Training Data", tag_list)

#         if tag_choice:
#             display_files_with_delete(tag_choice)
#             file_choice = st.selectbox(
#                 "Select a Training Data File to Ask a Question",
#                 [f["name"] for f in processed_files.values() if f.get("tag") == tag_choice]
#             )

#             # --- File selected ---
#             if file_choice:
#                 selected_file = next(f for f in processed_files.values() if f["name"] == file_choice)

#                 # --- Chat input ---
#                 user_input = st.chat_input("💬 Ask about the ERP system...")

#                 if user_input:
#                     # --- Show past messages ---
#                     for msg in st.session_state.chat_memory.chat_memory.messages:
#                         with st.chat_message("user" if msg.type == "human" else "assistant"):
#                             st.markdown(msg.content)
#                     # --- Handle user input ---
#                     if prompt := user_input.strip():
#                         # Show user message instantly
#                         with st.chat_message("user"):
#                             st.markdown(prompt)

#                     # Generate response
#                     response = ask_question_with_model(
#                         user_input,
#                         selected_file["text"],
#                         model_choice,
#                         selected_file["vector_store_path"],
#                         st.session_state.chat_memory
#                     )
#                         # Show assistant message
#                     with st.chat_message("assistant"):
#                         st.markdown(response)

#                     # Append to memory ONLY ONCE
#                     # st.session_state.chat_memory.chat_memory.add_user_message(user_input)
#                     # st.session_state.chat_memory.chat_memory.add_ai_message(response)
#                     with open("debug_chat_memory.json", "w") as f:
#                         json.dump([msg.dict() for msg in st.session_state.chat_memory.chat_memory.messages], f, indent=4)

#                 # # ✅ Always display chat history (outside the if block)
#                 # st.markdown("### 💬 Chat History")
#                 # for msg in st.session_state.chat_memory.chat_memory.messages:
#                 #     if msg.type == "human":
#                 #         st.markdown(f"**🧑 You:** {msg.content}")
#                 #     else:
#                 #         st.markdown(f"**🤖 ERP Assistant:** {msg.content}")

# if __name__ == "__main__":
#     main()
import streamlit as st
import json
from concurrent.futures import ThreadPoolExecutor
from langchain.memory import ConversationBufferMemory

def main():
    st.set_page_config(page_title="RAG Application")
    st.header("📘 ERP Chatbot")

    # ✅ Initialize chat memory once
    if "chat_memory" not in st.session_state:
        st.session_state.chat_memory = ConversationBufferMemory(
            memory_key="chat_history",
            input_key="question",
            return_messages=True
        )

    # --- Sidebar UI (renamed to sound advanced) ---
    st.sidebar.title("🧠 GenAI Core Configuration")

    uploaded_files = st.sidebar.file_uploader(
        "📂 Inject Cognitive Data Modules",  # instead of “Upload PDFs”
        type=["pdf"],
        accept_multiple_files=True
    )

    tag = st.sidebar.text_input(
        "🔖 Define Contextual Memory ID",  # instead of “Enter a tag”
        placeholder="e.g., ERP_Core_Set_01"
    )

    enable_ocr = st.sidebar.checkbox(
        "🧩 Activate Synthetic Vision (OCR Mode)"  # instead of “Enable OCR”
    )

    ocr_tool = st.sidebar.radio(
        "🔬 Select Vision Engine Protocol",  # instead of “Choose OCR Tool”
        ["fitz", "pdfplumber"]
    )

    model_choice = st.sidebar.selectbox(
        "⚙️ Neural Reasoning Model Selector",  # instead of “Choose LLM Model”
        ["Gemini", "Mistral"]
    )

    process_button = st.sidebar.button(
        "🚀 Initialize Knowledge Embedding Sequence"  # instead of “Process Files”
    )

    # --- File processing ---
    if process_button and uploaded_files and tag:
        with st.spinner("🧠 Engaging cognitive embedding pipeline..."):
            with ThreadPoolExecutor() as executor:
                for uploaded_file in uploaded_files:
                    result = executor.submit(
                        process_pdf, uploaded_file, tag, enable_ocr, ocr_tool
                    ).result()
                    if result:
                        st.success(f"✅ Indexed Knowledge Module: {uploaded_file.name}")
                    else:
                        st.error(f"❌ Failed to index: {uploaded_file.name}")

    # --- Show processed files (unchanged logic) ---
    if processed_files:
        display_tags_with_delete()

        tag_list = list(set(metadata.get("tag", "Untitled") for metadata in processed_files.values()))
        tag_choice = st.selectbox("📂 Select Context Memory Node", tag_list)

        if tag_choice:
            display_files_with_delete(tag_choice)
            file_choice = st.selectbox(
                "🧾 Select Embedded Knowledge Unit",
                [f["name"] for f in processed_files.values() if f.get("tag") == tag_choice]
            )

            # --- File selected ---
            if file_choice:
                selected_file = next(f for f in processed_files.values() if f["name"] == file_choice)

                # --- Chat input ---
                user_input = st.chat_input("💬 Query the ERP Intelligence System...")

                if user_input:
                    # --- Show past messages ---
                    for msg in st.session_state.chat_memory.chat_memory.messages:
                        with st.chat_message("user" if msg.type == "human" else "assistant"):
                            st.markdown(msg.content)

                    # --- Handle user input ---
                    if prompt := user_input.strip():
                        with st.chat_message("user"):
                            st.markdown(prompt)

                    # --- Generate response ---
                    response = ask_question_with_model(
                        user_input,
                        selected_file["text"],
                        model_choice,
                        selected_file["vector_store_path"],
                        st.session_state.chat_memory
                    )

                    with st.chat_message("assistant"):
                        st.markdown(response)

                    # --- Save chat logs ---
                    with open("debug_chat_memory.json", "w") as f:
                        json.dump(
                            [msg.dict() for msg in st.session_state.chat_memory.chat_memory.messages],
                            f,
                            indent=4
                        )

if __name__ == "__main__":
    main()




