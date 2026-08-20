from src.pipeline import build_retrieval_stack
from src.config import AppConfig

cfg = AppConfig.from_yaml('config/config.yaml')

# مش هنعيد chunking -- هنحمّل الـ chunks الموجودة فعلاً وهنبني/نفحص الـ Chroma بس
retriever = build_retrieval_stack(cfg, chunks=None)

collection = retriever.collection

print("=" * 50)
print("عدد الـ vectors المخزّنة في Chroma:", collection.count())
print("=" * 50)

# هات أول 3 عناصر واتأكد إن كل حاجة موجودة صح
sample = collection.peek(limit=3)
print("\nأول 3 IDs:", sample["ids"])
print("\nأول document (نص):", sample["documents"][0][:200])
print("\nأول metadata:", sample["metadatas"][0])
print("\nطول أول embedding vector:", len(sample["embeddings"][0]))