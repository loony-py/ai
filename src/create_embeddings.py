from sentence_transformers import SentenceTransformer
import psycopg2

# Load free embedding model
model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

# Example docs
docs = [
    "You can update your billing info from the Billing section in Account Settings.",
    "To reset your password, go to Account Settings → Security → Reset Password.",
    "For two-factor authentication, enable it in Security Settings under Account.",
]

# Generate embeddings
embeddings = model.encode(docs).tolist()

# Connect to PostgreSQL
conn = psycopg2.connect("dbname=ai user=sankar password=san@pos#25 host=localhost")
cur = conn.cursor()

# Insert docs + embeddings
for content, emb in zip(docs, embeddings):
    cur.execute("INSERT INTO documents (content, embedding) VALUES (%s, %s)", (content, emb))

conn.commit()
cur.close()
conn.close()
