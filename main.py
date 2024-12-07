from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import WebBaseLoader
from langchain.chains import RetrievalQA
from langchain.text_splitter import RecursiveCharacterTextSplitter
import streamlit as st

@st.cache_resource
def load_retriever(api_key):
    loader = WebBaseLoader(["https://uz.wikipedia.org/wiki/Audi"])
    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    docs = text_splitter.split_documents(documents)

    embeddings = OpenAIEmbeddings(api_key=api_key)
    vector_store = FAISS.from_documents(docs, embeddings)
    return vector_store.as_retriever()

st.markdown(
    """
    <style>
    .centered-title {
        text-align: center;
    }
    </style>
    <h1 class="centered-title">Audi Company</h1>
    """,
    unsafe_allow_html=True
)
st.caption("🤖 Men Audi kompaniyasi haqida ma'lumot beruvchi shaxsiy yordamchiman.")

if "openai_api_key" not in st.session_state:
    st.session_state["openai_api_key"] = None

if st.session_state["openai_api_key"] is None:
    openai_api_key = st.text_input("OpenAI API Key", type="password", placeholder="API kalitini kiriting:")
    if openai_api_key:
        if openai_api_key.startswith("sk-") and len(openai_api_key) >= 51:
            st.session_state["openai_api_key"] = openai_api_key
            st.experimental_rerun()
        else:
            st.error("API kaliti noto'g'ri.")
else:
    openai_api_key = st.session_state["openai_api_key"]
    retriever = load_retriever(api_key=openai_api_key)

    chat_model = ChatOpenAI(api_key=openai_api_key, model="gpt-4")
    qa_chain = RetrievalQA.from_chain_type(
        llm=chat_model,
        retriever=retriever,
        return_source_documents=True
    )

    if "messages" not in st.session_state:
        st.session_state["messages"] = [{"role": "assistant", "content": "Assalomu Alaykum! Audi haqida savollar bering, men yordam beraman."}]

    for msg in st.session_state.messages:
        st.chat_message(msg["role"]).write(msg["content"])

    if prompt := st.chat_input():
        st.session_state.messages.append({"role": "user", "content": prompt})
        st.chat_message("user").write(prompt)

        result = qa_chain({"query": prompt})
        response = result["result"]
        source_docs = result.get("source_documents", [])

        if not source_docs or not response.strip():
            response = "Bu savolga oid ma'lumot topilmadi. Audi haqidagi boshqa savol bering."

        st.session_state.messages.append({"role": "assistant", "content": response})
        st.chat_message("assistant").write(response)