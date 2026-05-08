"""analytics/src/analytics_engine.py — Natural language to SQL for BFSI loan analytics."""
import os
import time
import psycopg2
import psycopg2.extras
from typing import Dict, List
from dotenv import load_dotenv

from llm.src.provider.base import LLMProviderFactory
from ingestion.src.storage.vector_store import get_connection

load_dotenv()

SCHEMA_CONTEXT = """
PostgreSQL schema for BFSI loan analytics:

TABLE loan_records:
  loan_id VARCHAR, customer_name VARCHAR, loan_amount DECIMAL,
  interest_rate DECIMAL, tenure_months INTEGER, emi_amount DECIMAL,
  disbursement_dt DATE, status VARCHAR ('active'|'closed'|'npa'|'overdue')

TABLE emi_schedule:
  loan_id VARCHAR, installment_no INTEGER, due_date DATE,
  emi_amount DECIMAL, principal DECIMAL, interest DECIMAL,
  paid_date DATE, paid_amount DECIMAL,
  status VARCHAR ('pending'|'paid'|'overdue')

TABLE query_log:
  question TEXT, llm_provider VARCHAR, latency_ms INTEGER, created_at TIMESTAMP

Rules:
- Always use parameterized queries (no user input in SQL)
- Never use DROP, DELETE, UPDATE, INSERT, ALTER, TRUNCATE
- Use SELECT only
- Return ONLY the SQL query, nothing else
"""


class AnalyticsEngine:
    """Generate and execute SQL from natural-language BFSI queries."""

    def __init__(self):
        self.llm = LLMProviderFactory.get()

    def query(self, question: str) -> Dict:
        start = time.time()

        # Step 1: Generate SQL
        sql = self._generate_sql(question)
        print(f"[Analytics] Generated SQL:\n{sql}")

        # Step 2: Execute
        try:
            results, columns = self._execute(sql)
        except Exception as e:
            return {
                "success": False,
                "error":   str(e),
                "sql":     sql,
                "results": [],
            }

        latency_ms = int((time.time() - start) * 1000)
        return {
            "success":    True,
            "question":   question,
            "sql":        sql,
            "columns":    columns,
            "results":    results,
            "row_count":  len(results),
            "provider":   self.llm.name,
            "latency_ms": latency_ms,
        }

    def _generate_sql(self, question: str) -> str:
        prompt = f"Convert this business question to a PostgreSQL SELECT query:\n\n{question}"
        sql = self.llm.complete(
            prompt,
            system=SCHEMA_CONTEXT,
            temperature=0.0,
        ).strip()
        # Strip markdown code fences if model adds them
        if sql.startswith("```"):
            lines = sql.split("\n")
            sql = "\n".join(l for l in lines if not l.startswith("```")).strip()
        return sql

    def _execute(self, sql: str) -> tuple:
        with get_connection() as conn, conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(sql)
            rows = [dict(r) for r in cur.fetchall()]
            cols = list(rows[0].keys()) if rows else []
            return rows, cols

    def portfolio_summary(self) -> Dict:
        """Pre-built dashboard query — overall loan portfolio health."""
        sql = """
            SELECT
                status,
                COUNT(*)                    AS loan_count,
                SUM(loan_amount)            AS total_disbursed,
                AVG(interest_rate)          AS avg_rate,
                SUM(emi_amount)             AS monthly_emi_inflow
            FROM loan_records
            GROUP BY status
            ORDER BY total_disbursed DESC
        """
        rows, cols = self._execute(sql)
        return {"columns": cols, "results": rows}

    def overdue_emi_report(self, days: int = 30) -> Dict:
        """EMIs overdue by more than `days` days."""
        sql = """
            SELECT
                e.loan_id,
                lr.customer_name,
                e.due_date,
                e.emi_amount,
                CURRENT_DATE - e.due_date AS days_overdue
            FROM emi_schedule e
            JOIN loan_records lr ON e.loan_id = lr.loan_id
            WHERE e.status = 'overdue'
              AND CURRENT_DATE - e.due_date > %s
            ORDER BY days_overdue DESC
        """
        with get_connection() as conn, conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(sql, (days,))
            rows = [dict(r) for r in cur.fetchall()]
        return {"columns": list(rows[0].keys()) if rows else [], "results": rows}
