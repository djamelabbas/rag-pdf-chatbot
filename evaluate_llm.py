
import re
from dotenv import load_dotenv
from llama_index.core import StorageContext, load_index_from_storage
from llama_index.embeddings.google_genai import GoogleGenAIEmbedding
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()

# Charger notre index optimise
index = load_index_from_storage(
    StorageContext.from_defaults(persist_dir="storage_256"),
    embed_model=GoogleGenAIEmbedding(
        model_name="gemini-embedding-001"
    )
)

retriever = index.as_retriever(similarity_top_k=2)
llm = ChatGoogleGenerativeAI(model="gemini-3.5-flash-lite")

prompt = ChatPromptTemplate.from_messages([
    ("system", """Tu es un assistant documentaire.
    Reponds en francais uniquement a partir du contexte.
    Si la reponse est absente, dis :
    "Je ne trouve pas cette information dans le document."
    Cite les sources avec [1], [2], etc.
    N'invente aucune information."""),
    ("human", "Contexte :\n{context}\n\nQuestion : {question}")
])

chain = prompt | llm

# Questions + expressions attendues + abstention
tests = [
    ("Tarif etudiant",
     "Combien coûte l'inscription pour un étudiant majeur ?",
     [r"\b8\s*€"], False),

    ("Emprunt livres",
     "Combien de livres peut-on emprunter ?",
     [r"\b5\s+livres?\b"], False),

    ("Atelier Python",
     "Quand a lieu l'atelier Python ?",
     [r"mercredi", r"17\s*h\s*30", r"19\s*h"], False),

    ("Impression",
     "Quel est le prix d'une impression en noir et blanc ?",
     [r"0[,.]10\s*€"], False),

    ("Retard",
     "Après combien de jours de retard suspend-on les emprunts ?",
     [r"\b14\s+jours\b"], False),

    ("Information absente",
     "Qui dirige la médiathèque ?",
     [r"je ne trouve pas cette information"], True)
]

passed = 0

for label, question, patterns, abstention in tests:
    results = retriever.retrieve(question)

    context = "\n\n".join(
        f"[{i}] {r.node.get_content()}"
        for i, r in enumerate(results, start=1)
    )

    answer = chain.invoke({
        "context": context,
        "question": question
    }).text

    correct = all(
        re.search(pattern, answer, re.IGNORECASE)
        for pattern in patterns
    )

    cited = abstention or bool(
        re.search(r"\[[12]\]", answer)
    )

    success = correct and cited
    passed += success

    print(f"\n{'PASS' if success else 'FAIL'} - {label}")
    print(f"Reponse : {answer}")
    print(f"Exactitude indicative : {correct}")
    print(f"Citation presente : {cited}")

print(f"\nRESULTAT : {passed}/{len(tests)} tests reussis")
print(f"Taux de reussite : {passed / len(tests) * 100:.1f}%")
