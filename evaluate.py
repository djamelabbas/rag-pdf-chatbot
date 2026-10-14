
import unicodedata
from dotenv import load_dotenv
from llama_index.core import StorageContext, load_index_from_storage
from llama_index.embeddings.google_genai import GoogleGenAIEmbedding
import sys

storage_dir = sys.argv[1] if len(sys.argv) > 1 else "storage"

load_dotenv()

# 1. Charger l'index existant
embed_model = GoogleGenAIEmbedding(
    model_name="gemini-embedding-001"
)

storage = StorageContext.from_defaults(
    persist_dir=storage_dir
)

index = load_index_from_storage(
    storage,
    embed_model=embed_model
)

# 2. Jeu de tests : question + passage attendu
tests = [
    (
        "Combien coûte l'inscription pour un étudiant majeur ?",
        "Étudiants majeurs 8 €"
    ),
    (
        "Combien de livres peut-on emprunter ?",
        "Livres 5 21 jours"
    ),
    (
        "Quand a lieu l'atelier Python ?",
        "Découvrir Python Chaque mercredi"
    ),
    (
        "Quel est le prix d'une impression en noir et blanc ?",
        "Impression : 0,10"
    ),
    (
        "Après combien de jours de retard suspend-on les emprunts ?",
        "Après 14 jours de retard"
    )
]

# Normaliser accents et espaces pour comparer les textes
def normalize(text):
    text = unicodedata.normalize("NFKD", text.casefold())
    text = "".join(
        char for char in text
        if not unicodedata.combining(char)
    )
    return " ".join(text.split())

# 3. Rechercher les 3 premiers passages
retriever = index.as_retriever(similarity_top_k=3)
ranks = []

for question, expected in tests:
    results = retriever.retrieve(question)

    rank = next(
        (
            i for i, result in enumerate(results, start=1)
            if normalize(expected) in normalize(
                result.node.get_content()
            )
        ),
        None
    )

    ranks.append(rank)

    print(f"\nQuestion : {question}")
    print(f"Bon passage : rang {rank if rank else 'non trouvé'}")

# 4. Calculer les métriques
print("\n===== RESULTATS =====")

for k in [1, 2, 3]:
    hits = sum(
        1 for rank in ranks
        if rank is not None and rank <= k
    )

    score = hits / len(tests) * 100
    print(f"Hit@{k} : {score:.1f}% ({hits}/{len(tests)})")
