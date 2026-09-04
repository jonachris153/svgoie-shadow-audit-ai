# OIE Shadow Audit AI Service v2

AI/ML REST service for the OIE Shadow Audit & Compliance Intelligence workflow.

Endpoint:
`POST /shadow-audit/analyze`

The request contains closed exception cases and returns:
- driftScore
- driftStatus
- primarySignals
- aiInsight
- recommendedAction
- affected case IDs
- officer/branch concentration
- exception breakdown
- repeated-justification signal when justification text is supplied

The model is a prototype Isolation Forest baseline trained on representative synthetic normal patterns. Replace the synthetic baseline with sufficient representative historical data when available.

Run:
```powershell
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
uvicorn main:app --host 0.0.0.0 --port 8000
```

Open:
`http://127.0.0.1:8000/docs`

Newgen target once deployed:
- Operation: POST
- Resource Path: `/shadow-audit/analyze`
- Request: Application/JSON
- Response: Application/JSON
- Base URI: actual deployed HTTPS service URL
