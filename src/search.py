from sentence_transformers import SentenceTransformer
import psycopg2

# Load open-source embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")  # ~384 dim

# Connect to Postgres
conn = psycopg2.connect("dbname=ai user=sankar password=san@pos#25 host=localhost")
cur = conn.cursor()

def to_pgvector(embedding):
    return "[" + ",".join(str(x) for x in embedding) + "]"

query = "How can I change my password?"
query_vec = model.encode(query).tolist()
query_vec_str = to_pgvector(query_vec)

cur.execute("""
SELECT content, embedding <-> %s::vector AS distance
FROM documents
ORDER BY embedding <-> %s::vector
LIMIT 1;
""", (query_vec_str, query_vec_str))

results = cur.fetchall()
for r in results:
    print(r)
