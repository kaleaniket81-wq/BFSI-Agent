"""ingestion/src/storage/vector_store.py — Store and retrieve chunks from pgvector."""
import os
import uuid
import psycopg2
import psycopg2.extras
from typing import List, Dict
from dotenv import load_dotenv

load_dotenv()


def get_connection():
    return psycopg2.connect(
        host=os.getenv("DB_HOST", "localhost"),
        port=int(os.getenv("DB_PORT", 5432)),
        dbname=os.getenv("DB_NAME", "bfsi_intelligence"),
        user=os.getenv("DB_USER", "bfsi_user"),
        password=os.getenv("DB_PASSWORD"),
    )


class VectorStore:
    """Persist document chunks with embeddings to PostgreSQL + pgvector."""

    def save_document(self, filename: str, doc_type: str,
                      page_count: int, file_size: int = 0) -> str:
        """Insert a document record and return its UUID."""
        doc_id = str(uuid.uuid4())
        with get_connection() as conn, conn.cursor() as cur:
            cur.execute(
                """INSERT INTO documents (id, filename, doc_type, file_size, page_count, status)
                   VALUES (%s, %s, %s, %s, %s, 'processing')""",
                (doc_id, filename, doc_type, file_size, page_count),
            )
        return doc_id

    def save_chunks(self, document_id: str, chunks: List[Dict],
                    embeddings: List[List[float]]) -> int:
        """Bulk-insert chunks with their embeddings."""
        records = [
            (
                str(uuid.uuid4()),
                document_id,
                c["chunk_index"],
                c["text"],
                c.get("page_number"),
                c.get("token_count"),
                embeddings[i],
            )
            for i, c in enumerate(chunks)
        ]
        with get_connection() as conn, conn.cursor() as cur:
            psycopg2.extras.execute_values(
                cur,
                """INSERT INTO document_chunks
                   (id, document_id, chunk_index, chunk_text, page_number, token_count, embedding)
                   VALUES %s""",
                records,
                template="(%s,%s,%s,%s,%s,%s,%s::vector)",
            )
            cur.execute(
                "UPDATE documents SET status='indexed' WHERE id=%s",
                (document_id,),
            )
        return len(records)

    def similarity_search(self, query_embedding: List[float],
                          top_k: int = 5, doc_type: str = None) -> List[Dict]:
        """Find the top-k most similar chunks using cosine similarity."""
        filter_clause = "AND d.doc_type = %s" if doc_type else ""
        params = [query_embedding, top_k]
        if doc_type:
            params.insert(1, doc_type)

        sql = f"""
            SELECT
                dc.chunk_text,
                dc.page_number,
                d.filename,
                d.doc_type,
                1 - (dc.embedding <=> %s::vector) AS similarity
            FROM document_chunks dc
            JOIN documents d ON dc.document_id = d.id
            WHERE d.status = 'indexed'
            {filter_clause}
            ORDER BY dc.embedding <=> %s::vector
            LIMIT %s
        """
        # Rebuild params for the two embedding placeholders
        if doc_type:
            params = [query_embedding, doc_type, query_embedding, top_k]
        else:
            params = [query_embedding, query_embedding, top_k]

        with get_connection() as conn, conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(sql, params)
            return [dict(r) for r in cur.fetchall()]
