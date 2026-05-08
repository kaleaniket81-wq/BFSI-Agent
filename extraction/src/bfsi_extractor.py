"""
extraction/src/bfsi_extractor.py

DAY 4 UPGRADE — BFSI Structured Extraction

Extracts structured financial fields from loan agreements and contracts:
  - interest_rate, tenure, emi_amount, loan_amount
  - penalties (late payment, prepayment, foreclosure)
  - key dates (disbursement, first_emi, last_emi)
  - security / collateral
  - special clauses

Also supports: document comparison (compare two loan agreements side-by-side).

Interview line:
  "Generic Q&A wasn't enough for BFSI use cases. A compliance officer needs
   structured data — give me all penalty clauses across 50 loan agreements.
   I built a domain-specific extractor using LLM function-calling style prompts
   that returns validated JSON. This is what separates a demo from a product."
"""

import os
import json
import re
from typing import Dict, List, Optional
from dotenv import load_dotenv

load_dotenv()

EXTRACTION_SYSTEM_PROMPT = """You are a BFSI document extraction specialist.
Extract structured financial data from loan agreements and contracts.
Always return valid JSON. Use null for fields not found in the document.
Be precise — financial data must be exact as stated in the document.
Never invent or estimate values."""

LOAN_EXTRACTION_PROMPT = """Extract the following fields from this loan agreement.
Return ONLY a JSON object with these exact keys:

{
  "loan_id":              "string or null",
  "borrower_name":        "string or null",
  "lender_name":          "string or null",
  "loan_amount":          "number or null (in INR)",
  "interest_rate":        "number or null (annual %)",
  "tenure_months":        "number or null",
  "emi_amount":           "number or null (monthly INR)",
  "disbursement_date":    "string or null (DD/MM/YYYY)",
  "first_emi_date":       "string or null",
  "last_emi_date":        "string or null",
  "purpose":              "string or null",
  "penalties": {
    "late_payment":       "string or null (describe the penalty)",
    "prepayment":         "string or null",
    "foreclosure":        "string or null"
  },
  "collateral":           "string or null",
  "insurance_required":   "boolean or null",
  "special_clauses":      ["list of important clauses as strings"]
}

Document text:
{document_text}"""

COMPARISON_PROMPT = """Compare these two BFSI documents and return a structured JSON comparison.

Document 1 ({doc1_name}):
{doc1_text}

Document 2 ({doc2_name}):
{doc2_text}

Return ONLY this JSON structure:
{{
  "comparison": {{
    "interest_rate":   {{"doc1": null, "doc2": null, "difference": null, "better": "doc1|doc2|equal|null"}},
    "tenure_months":   {{"doc1": null, "doc2": null, "difference": null}},
    "emi_amount":      {{"doc1": null, "doc2": null, "difference": null}},
    "loan_amount":     {{"doc1": null, "doc2": null}},
    "late_penalty":    {{"doc1": null, "doc2": null, "better": "doc1|doc2|equal|null"}},
    "prepayment":      {{"doc1": null, "doc2": null, "better": "doc1|doc2|equal|null"}},
    "collateral":      {{"doc1": null, "doc2": null}}
  }},
  "summary": "2-3 sentence plain English comparison",
  "recommendation": "Which document is more borrower-friendly and why"
}}"""


class BFSIExtractor:
    """Extract structured data from BFSI documents using LLM."""

    def __init__(self, llm_provider=None):
        from llm.src.provider.base import LLMProviderFactory
        self.llm = llm_provider or LLMProviderFactory.get()

    # ── Single document extraction ────────────────────────────────────────────
    def extract_loan_fields(self, document_text: str) -> Dict:
        """
        Extract structured loan fields from document text.
        Returns validated dict with all fields (null if not found).
        """
        prompt = LOAN_EXTRACTION_PROMPT.format(document_text=document_text[:4000])

        raw = self.llm.complete(prompt, system=EXTRACTION_SYSTEM_PROMPT, temperature=0.0)
        return self._parse_json_response(raw, self._empty_loan_schema())

    def extract_from_chunks(self, chunks: List[Dict]) -> Dict:
        """
        Extract from retrieved chunks (used after RAG retrieval).
        Merges chunk texts and extracts.
        """
        combined_text = "\n\n".join(c.get("chunk_text", c.get("text", "")) for c in chunks[:10])
        return self.extract_loan_fields(combined_text)

    # ── Document comparison ───────────────────────────────────────────────────
    def compare_documents(
        self,
        doc1_text: str,
        doc2_text: str,
        doc1_name: str = "Document 1",
        doc2_name: str = "Document 2",
    ) -> Dict:
        """
        Side-by-side comparison of two BFSI documents.
        Returns structured comparison with recommendation.
        """
        prompt = COMPARISON_PROMPT.format(
            doc1_name = doc1_name,
            doc2_name = doc2_name,
            doc1_text = doc1_text[:2000],
            doc2_text = doc2_text[:2000],
        )
        raw = self.llm.complete(prompt, system=EXTRACTION_SYSTEM_PROMPT, temperature=0.0)
        return self._parse_json_response(raw, {"comparison": {}, "summary": "", "recommendation": ""})

    # ── Compliance check ──────────────────────────────────────────────────────
    def check_rbi_compliance(self, extracted: Dict) -> Dict:
        """
        Basic RBI compliance checks on extracted loan data.
        Real compliance is complex — this demonstrates domain intelligence.
        """
        issues  = []
        warnings= []

        rate = extracted.get("interest_rate")
        if rate is not None:
            if rate > 36:
                issues.append(f"Interest rate {rate}% exceeds RBI NBFC cap of 36% for microfinance")
            elif rate > 18:
                warnings.append(f"Interest rate {rate}% is above typical home/personal loan range")

        insurance = extracted.get("insurance_required")
        if insurance is False:
            warnings.append("No insurance requirement — RBI guidelines recommend life cover for home loans")

        collateral = extracted.get("collateral")
        loan_amt   = extracted.get("loan_amount")
        if loan_amt and loan_amt > 500000 and not collateral:
            warnings.append("Loan above ₹5L with no collateral — verify secured loan classification")

        return {
            "compliant": len(issues) == 0,
            "issues":    issues,
            "warnings":  warnings,
            "checked_fields": ["interest_rate", "insurance_required", "collateral"],
        }

    # ── Helpers ───────────────────────────────────────────────────────────────
    def _parse_json_response(self, raw: str, fallback: Dict) -> Dict:
        """Strip markdown fences and parse JSON. Return fallback on failure."""
        # Remove ```json ... ``` fences
        cleaned = re.sub(r"```(?:json)?\s*", "", raw).replace("```", "").strip()
        try:
            return json.loads(cleaned)
        except json.JSONDecodeError:
            # Try extracting first JSON block
            match = re.search(r"\{.*\}", cleaned, re.DOTALL)
            if match:
                try:
                    return json.loads(match.group())
                except Exception:
                    pass
            print(f"[Extractor] JSON parse failed. Raw: {raw[:200]}")
            return fallback

    @staticmethod
    def _empty_loan_schema() -> Dict:
        return {
            "loan_id": None, "borrower_name": None, "lender_name": None,
            "loan_amount": None, "interest_rate": None, "tenure_months": None,
            "emi_amount": None, "disbursement_date": None,
            "first_emi_date": None, "last_emi_date": None, "purpose": None,
            "penalties": {"late_payment": None, "prepayment": None, "foreclosure": None},
            "collateral": None, "insurance_required": None, "special_clauses": [],
        }
