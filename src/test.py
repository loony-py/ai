from sentence_transformers import SentenceTransformer, util, CrossEncoder
import torch
import numpy as np

# 1. Load a strong embedding model
model = SentenceTransformer("multi-qa-mpnet-base-dot-v1")

# Example corpus (could be documents, knowledge base, etc.)
corpus = [
    "Sentence Transformers can create dense embeddings.",
    "Transformers are widely used in Natural Language Processing.",
    "You can use embeddings for semantic search and clustering.",
    "Large Language Models can generate text but embeddings retrieve knowledge.",
    "Vector databases like FAISS or Pinecone store embeddings efficiently."
]

# Precompute embeddings (useful for large-scale retrieval)
corpus_embeddings = model.encode(corpus, convert_to_tensor=True, show_progress_bar=True)

# 2. Semantic Search
query = "How do I store embeddings for fast similarity search?"
query_emb = model.encode(query, convert_to_tensor=True)

hits = util.semantic_search(query_emb, corpus_embeddings, top_k=3)
print("\nSemantic Search Results:")
for hit in hits[0]:
    print(f"{corpus[hit['corpus_id']]} (score: {hit['score']:.4f})")

# 3. Clustering (discover hidden structure in embeddings)
from sklearn.cluster import KMeans
num_clusters = 2
kmeans = KMeans(n_clusters=num_clusters)
kmeans.fit(corpus_embeddings.cpu().numpy())

print("\nCluster Assignments:")
for i, sentence in enumerate(corpus):
    print(f"Cluster {kmeans.labels_[i]}: {sentence}")

# 4. Re-ranking with a CrossEncoder (fine-grained scoring)
cross_encoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')

# Take top-3 semantic search hits, then rerank
rerank_candidates = [corpus[hit['corpus_id']] for hit in hits[0]]
rerank_scores = cross_encoder.predict([(query, doc) for doc in rerank_candidates])

print("\nRe-ranked Results:")
for doc, score in sorted(zip(rerank_candidates, rerank_scores), key=lambda x: x[1], reverse=True):
    print(f"{doc} (cross-encoder score: {score:.4f})")

# 5. Hybrid approach with LLM + embeddings
# Example: Use embeddings to narrow down candidates, then pass top results to an LLM for final answer
from transformers import pipeline
qa_pipeline = pipeline("text2text-generation", model="google/flan-t5-base")

context = "\n".join(rerank_candidates)
question = "What's the best way to store embeddings?"
prompt = f"Context: {context}\n\nQuestion: {question}\nAnswer:"

response = qa_pipeline(prompt, max_length=100, do_sample=False)[0]['generated_text']

print("\nLLM Answer:", response)
