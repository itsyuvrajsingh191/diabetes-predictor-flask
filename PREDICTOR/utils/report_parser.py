import re
import json

FIELD_ALIASES = {
    "glucose":               "Glucose",
    "blood glucose":         "Glucose",
    "blood sugar":           "Glucose",
    "bloodpressure":         "BloodPressure",
    "blood pressure":        "BloodPressure",
    "bp":                    "BloodPressure",
    "bmi":                   "BMI",
    "body mass index":       "BMI",
    "insulin":               "Insulin",
    "age":                   "Age",
    "pregnancies":           "Pregnancies",
    "skin thickness":        "SkinThickness",
    "skinthickness":         "SkinThickness",
    "diabetespedigreefunction": "DiabetesPedigreeFunction",
    "diabetes pedigree":     "DiabetesPedigreeFunction",
    "pedigree":              "DiabetesPedigreeFunction",
    "dpf":                   "DiabetesPedigreeFunction",
}

DEFAULTS = {
    "Pregnancies": 1,
    "Glucose": 100,
    "BloodPressure": 72,
    "SkinThickness": 20,
    "Insulin": 80,
    "BMI": 25.0,
    "DiabetesPedigreeFunction": 0.35,
    "Age": 30,
}

def _normalise_key(raw: str) -> str:
    return FIELD_ALIASES.get(raw.strip().lower(), raw.strip())

def parse_uploaded_report(file_bytes: bytes, file_type: str) -> dict:
    if file_type == "pdf":
        return _parse_pdf(file_bytes)
    if file_type == "json":
        return _parse_json(file_bytes)
    raise ValueError(f"Unsupported file type: {file_type}")

def _parse_pdf(file_bytes: bytes) -> dict:
    """
    Extract biomarker values from a PDF report.
    Looks for patterns like:  Glucose: 120  or  BMI 27.3
    """
    try:
        import pdfplumber
        import io
        with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
            text = "\n".join(page.extract_text() or "" for page in pdf.pages)
    except ImportError:
        # Fallback: try PyPDF2
        try:
            import PyPDF2, io
            reader = PyPDF2.PdfReader(io.BytesIO(file_bytes))
            text = "\n".join(p.extract_text() or "" for p in reader.pages)
        except Exception:
            return DEFAULTS.copy()

    result = DEFAULTS.copy()

    # Try to find "Label: value" or "Label value" patterns
    pattern = re.compile(
        r'([A-Za-z ]+?)\s*[:\-–]\s*([0-9]+(?:\.[0-9]+)?)',
        re.IGNORECASE
    )
    for match in pattern.finditer(text):
        raw_key = match.group(1).strip()
        value   = float(match.group(2))
        std_key = _normalise_key(raw_key)
        if std_key in result:
            result[std_key] = value

    return result

def _parse_json(file_bytes: bytes) -> dict:
    raw = json.loads(file_bytes)
    result = DEFAULTS.copy()
    # Accept nested {"features": {...}} or flat {"Glucose": 120}
    if "features" in raw:
        raw = raw["features"]
    for k, v in raw.items():
        std = _normalise_key(k)
        if std in result:
            try:
                result[std] = float(v)
            except (TypeError, ValueError):
                pass
    return result
