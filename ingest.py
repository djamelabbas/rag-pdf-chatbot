
from pathlib import Path
from dotenv import load_dotenv

from llama_index.core import SimpleDirectoryReader, VectorStoreIndex
from llama_index.core.node_parser import SentenceSplitter
from llama_index.embeddings.google_genai import GoogleGenAIEmbedding

# 1. Charger la clé API depuis .env
load_dotenv()

# 2. Charger le PDF
pdf_path = Path("data/document.pdf")
documents = SimpleDirectoryReader(
    input_files=[str(pdf_path)]
).load_data()

print(f"Documents extraits : {len(documents)}")

# 3. Découper le texte en morceaux
splitter = SentenceSplitter(
    chunk_size=256,
    chunk_overlap=40
)

nodes = splitter.get_nodes_from_documents(documents)
print(f"Chunks créés : {len(nodes)}")

# 4. Générer les embeddings et construire l'index
embed_model = GoogleGenAIEmbedding(
    model_name="gemini-embedding-001"
)

index = VectorStoreIndex(
    nodes,
    embed_model=embed_model
)

# 5. Sauvegarder l'index localement
# index.storage_context.persist(persist_dir="storage")
index.storage_context.persist(persist_dir="storage_256")

print("Indexation terminée !")
