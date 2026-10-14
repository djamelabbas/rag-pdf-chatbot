from dotenv import load_dotenv

from llama_index.core import StorageContext, load_index_from_storage
from llama_index.embeddings.google_genai import GoogleGenAIEmbedding

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate

# 1. Charger les variables d'environnement
load_dotenv()

# 2. Recharger l'index existant avec le meme modele d'embeddings
embed_model = GoogleGenAIEmbedding(
    model_name="gemini-embedding-001"
)

storage_context = StorageContext.from_defaults(
    persist_dir="storage_256"
)

index = load_index_from_storage(
    storage_context,
    embed_model=embed_model
)

# 3. Configurer la recherche semantique
retriever = index.as_retriever(similarity_top_k=2)

# 4. Configurer le LLM via LangChain
llm = ChatGoogleGenerativeAI(
    model="gemini-3.5-flash-lite"
)

# 5. Creer le prompt
prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        """Tu es un assistant documentaire.
        Reponds en francais uniquement avec les informations
        disponibles dans le contexte fourni.

        Si l'information est absente, reponds :
        "Je ne trouve pas cette information dans le document."

        Cite les sources utilisees avec [1], [2], etc.
        N'invente aucune information."""
    ),
    (
        "human",
        "Contexte :\n{context}\n\nQuestion : {question}"
    )
])

# 6. Construire la chaine LangChain
chain = prompt | llm

print("Chatbot RAG pret ! (tape 'quit' pour quitter)")

# 7. Boucle de questions
while True:
    question = input("\nTa question : ").strip()

    if question.lower() in ("quit", "exit"):
        break

    if not question:
        continue

    # Recherche des chunks pertinents
    results = retriever.retrieve(question)

    # Construction du contexte et des references
    passages = []

    print("\n--- Sources retrouvees ---")

    for i, result in enumerate(results, start=1):
        metadata = result.node.metadata
        filename = metadata.get("file_name", "document.pdf")
        page = metadata.get("page_label", "inconnue")

        passages.append(
            f"[{i}] Source : {filename}, page {page}\n"
            f"{result.node.get_content()}"
        )

        print(f"[{i}] {filename} - page {page}")
        print(f"Score : {result.score}")
        print(f"Extrait : {result.node.get_content()[:300]}")
        print("-" * 40)

    if not passages:
        print("Aucun passage retrouve.")
        continue

    context = "\n\n".join(passages)

    # Generation de la reponse
    response = chain.invoke({
        "context": context,
        "question": question
    })

    print("\n--- Reponse Gemini ---")
    print(response.text)
