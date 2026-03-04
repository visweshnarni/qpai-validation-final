import os
import json
import joblib
import requests
from datetime import datetime, timedelta

from fastapi import FastAPI
from pydantic import BaseModel
from dotenv import load_dotenv

from google import genai

# ----------------------------
# LOAD ENV + MODEL
# ----------------------------

load_dotenv()

client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

model = joblib.load("outpass_model.pkl")

app = FastAPI(title="Outpass AI Validation Service")

# ----------------------------
# REQUEST MODEL
# ----------------------------

class OutpassValidationRequest(BaseModel):
    attendance_pct: int
    attendance_attainable: int
    past_outpasses_gt3: int
    is_emergency: int
    religious_exception: int
    reason: str
    reason_category: str
    document_url: str | None = None


# ----------------------------
# OCR USING GEMINI
# ----------------------------

def extract_text_with_gemini(url):

    if not url:
        return ""

    response = requests.get(url, timeout=10)

    content_type = response.headers.get("Content-Type", "application/pdf")
    content_type = content_type.split(";")[0].strip()

    gemini_response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=[
            {
                "role": "user",
                "parts": [
                    {"text": "Extract all readable text from this document."},
                    {
                        "inline_data": {
                            "mime_type": content_type,
                            "data": response.content
                        }
                    }
                ]
            }
        ]
    )

    return gemini_response.text


# ----------------------------
# LLM ANALYSIS
# ----------------------------

def analyze_with_llm(reason, category, is_emergency, religious_exception, document_text):

    today = datetime.now().strftime("%d %B %Y")
    tomorrow = (datetime.now() + timedelta(days=1)).strftime("%d %B %Y")

    prompt = f"""
You are analyzing a college outpass request.
Return STRICT JSON.

Context:
Today's Date: {today}
Tomorrow's Date: {tomorrow}

Reason Category: {category}
Reason: "{reason}"
Emergency Flag: {is_emergency}
Religious Exception: {religious_exception}

Document Text:
\"\"\"{document_text}\"\"\"

Rules:

VAGUE RULE:
- A reason is vague ONLY if it is generic and unclear.
- "Medical checkup", "Hospital visit", "Exam", "Family function" are NOT vague.
- Reasons like "I need to go", "Personal work", "Urgent", "Important work" ARE vague.

DOCUMENT SUPPORT RULE:
- Document supports reason only if it clearly matches the reason.

DOCUMENT DATE RULE:
- Check if the document contains a date.It is to ensure that the requested outpass date and document date align, preventing misuse of old documents or future-dated documents.
- doc_has_date = 1 if a date exists in document text
- doc_has_date = 0 if no date found

DATE VALIDATION:
- If a date exists:
    - doc_date_valid = 1 ONLY if date equals today or tomorrow
    - otherwise 0
- If no date exists:
    - doc_date_valid = 0

If no document uploaded:
doc_supports_reason = 0
doc_has_date = 0
doc_date_valid = 0

Return JSON:

{{
"is_vague": 0 or 1,
"doc_supports_reason": 0 or 1,
"doc_has_date": 0 or 1,
"doc_date_valid": 0 or 1,
"comment": "clear explanation"
}}
"""

    try:

        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt,
            config={"response_mime_type": "application/json"}
        )

        return json.loads(response.text)

    except Exception as e:

        return {
            "is_vague": 1,
            "doc_supports_reason": 0,
            "doc_has_date": 0,
            "doc_date_valid": 0,
            "comment": f"LLM Error: {str(e)}"
        }


# ----------------------------
# MAIN API
# ----------------------------

@app.post("/validate")
def validate_outpass(data: OutpassValidationRequest):

    # OCR
    document_text = extract_text_with_gemini(data.document_url)

    # LLM reasoning
    llm_result = analyze_with_llm(
        data.reason,
        data.reason_category,
        data.is_emergency,
        data.religious_exception,
        document_text
    )

    doc_uploaded = 1 if data.document_url else 0

    # ----------------------------
    # BUILD FEATURE VECTOR
    # ----------------------------

    features = [[
        data.attendance_pct,
        data.attendance_attainable,
        data.past_outpasses_gt3,
        llm_result["is_vague"],
        data.is_emergency,
        doc_uploaded,
        llm_result["doc_supports_reason"],
        llm_result["doc_has_date"],
        llm_result["doc_date_valid"],
        data.religious_exception
    ]]

    prediction = model.predict(features)[0]

    decision_map = {
        0: "AUTO_APPROVE",
        1: "MANUAL_VERIFY",
        2: "REJECT"
    }

    return {
        "decision": decision_map[prediction],
        "explanation": llm_result["comment"],
        "features_used": {
            "attendance_pct": data.attendance_pct,
            "attendance_attainable": data.attendance_attainable,
            "past_outpasses_gt3": data.past_outpasses_gt3,
            "is_vague": llm_result["is_vague"],
            "is_emergency": data.is_emergency,
            "doc_uploaded": doc_uploaded,
            "doc_supports_reason": llm_result["doc_supports_reason"],
            "doc_has_date": llm_result["doc_has_date"],
            "doc_date_valid": llm_result["doc_date_valid"],
            "religious_exception": data.religious_exception
        }
    }