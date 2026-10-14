
import streamlit as st
from pathlib import Path
from dotenv import load_dotenv

from llama_index.core import StorageContext, load_index_from_storage
from llama_index.embeddings.google_genai import GoogleGenAIEmbedding

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()

# Configuration de la page
st.set_page_config(
    page_title="Chatbot RAG PDF",
    page_icon="📄",
    layout="centered"
)

st.title("📄 Chatbot RAG sur PDF")
st.caption("LlamaIndex + LangChain + Gemini")
st.write("Posez vos questions sur le document Médiathèque Aurore.")


# Charger le pipeline une seule fois
@st.cache_resource
def init_rag():
    embed_model = GoogleGenAIEmbedding(
        model_name="gemini-embedding-001"
    )

    storage = StorageContext.from_defaults(
        persist_dir="storage_256"
    )

    index = load_index_from_storage(
        storage,
        embed_model=embed_model
    )

    retriever = index.as_retriever(similarity_top_k=2)

    llm = ChatGoogleGenerativeAI(
        model="gemini-3.5-flash-lite"
    )

    prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            """Tu es un assistant documentaire.
            Réponds en français uniquement à partir du contexte.
            Si la réponse est absente, dis :
            "Je ne trouve pas cette information dans le document."
            Cite les sources avec [1], [2], etc.
            N'invente aucune information."""
        ),
        (
            "human",
            "Contexte :\n{context}\n\nQuestion : {question}"
        )
    ])

    return retriever, prompt | llm


# Fonction principale du RAG
def ask_rag(question):
    retriever, chain = init_rag()

    results = retriever.retrieve(question)

    context_parts = []
    sources = []

    for i, result in enumerate(results, start=1):
        metadata = result.node.metadata
        page = metadata.get("page_label", "?")
        filename = metadata.get("file_name", "document.pdf")
        content = result.node.get_content()

        context_parts.append(
            f"[{i}] {filename}, page {page}\n{content}"
        )

        sources.append({
            "reference": i,
            "page": page,
            "score": result.score,
            "extrait": content[:300]
        })

    if not results:
        return "Aucun passage trouvé.", []

    response = chain.invoke({
        "context": "\n\n".join(context_parts),
        "question": question
    })

    return response.text, sources


# Afficher les sources
def display_sources(sources):
    if sources:
        with st.expander("📚 Sources consultées"):
            for source in sources:
                score = source["score"]
                score_text = f"{score:.3f}" if score is not None else "N/A"

                st.markdown(
                    f"**[{source['reference']}] Page {source['page']}** "
                    f"— Similarité : {score_text}"
                )
                st.caption(source["extrait"])


# Historique de conversation
if "messages" not in st.session_state:
    st.session_state.messages = []

with st.sidebar:
    st.header("Projet RAG")
    st.write("Document : document.pdf")

    if st.button("Effacer la conversation"):
        st.session_state.messages = []
        st.rerun()


# Réafficher les messages existants
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message.get("sources"):
            display_sources(message["sources"])


# Nouvelle question
if question := st.chat_input("Pose ta question sur le PDF..."):

    st.session_state.messages.append({
        "role": "user",
        "content": question
    })

    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Recherche dans le document..."):
            try:
                answer, sources = ask_rag(question)

                st.markdown(answer)
                display_sources(sources)

                st.session_state.messages.append({
                    "role": "assistant",
                    "content": answer,
                    "sources": sources
                })

            except Exception as e:
                st.error(f"Erreur : {e}")
