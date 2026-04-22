from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
    HRFlowable,
    KeepTogether,
)
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT
from reportlab.graphics.shapes import Drawing, Rect, Circle, String
from reportlab.graphics import renderPDF
from io import BytesIO
from datetime import datetime
import math

# ── Colour palette ────────────────────────────────────────────────────────────
TEAL = colors.HexColor("#00C896")
AMBER = colors.HexColor("#FFB830")
RED = colors.HexColor("#FF6B6B")
DARK = colors.HexColor("#0A0F1E")
DARK2 = colors.HexColor("#111827")
MUTED = colors.HexColor("#8B95B0")
LIGHT = colors.HexColor("#F0F4FF")
WHITE = colors.white
BORDER = colors.HexColor("#E2E8F0")
TEAL_BG = colors.HexColor("#E6FBF5")
AMB_BG = colors.HexColor("#FFF8E6")
RED_BG = colors.HexColor("#FFF0F0")


def _risk_color(score):
    if score < 30:
        return TEAL, TEAL_BG, "Low Risk"
    if score < 60:
        return AMBER, AMB_BG, "Moderate Risk"
    return RED, RED_BG, "High Risk"


def _gauge_drawing(score: float) -> Drawing:
    W, H = 220, 130
    d = Drawing(W, H)
    cx, cy, r_outer, r_inner = W / 2, 30, 90, 60

    # Background arc segments (green / amber / red)
    import math

    def arc_seg(start_deg, end_deg, clr):
        steps = 30
        pts = []
        for i in range(steps + 1):
            a = math.radians(start_deg + (end_deg - start_deg) * i / steps)
            pts.append((cx + r_outer * math.cos(a), cy + r_outer * math.sin(a)))
        for i in range(steps, -1, -1):
            a = math.radians(start_deg + (end_deg - start_deg) * i / steps)
            pts.append((cx + r_inner * math.cos(a), cy + r_inner * math.sin(a)))
        from reportlab.graphics.shapes import Polygon

        p = Polygon([v for pt in pts for v in pt], fillColor=clr, strokeColor=None)
        d.add(p)

    arc_seg(180, 240, colors.HexColor("#E6FBF5"))
    arc_seg(240, 300, colors.HexColor("#FFF8E6"))
    arc_seg(300, 360, colors.HexColor("#FFF0F0"))

    # Needle
    angle = math.radians(180 + score * 1.8)
    nx = cx + (r_inner - 5) * math.cos(angle)
    ny = cy + (r_inner - 5) * math.sin(angle)
    from reportlab.graphics.shapes import Line

    needle_color, _, _ = _risk_color(score)
    d.add(Line(cx, cy, nx, ny, strokeColor=needle_color, strokeWidth=3))
    d.add(Circle(cx, cy, 6, fillColor=needle_color, strokeColor=WHITE, strokeWidth=2))

    # Score text
    s = String(
        cx,
        cy - 28,
        f"{score:.0f}%",
        fontName="Helvetica-Bold",
        fontSize=22,
        fillColor=needle_color,
        textAnchor="middle",
    )
    d.add(s)
    lbl_color, _, risk_label = _risk_color(score)
    d.add(
        String(
            cx,
            cy - 46,
            risk_label,
            fontName="Helvetica",
            fontSize=10,
            fillColor=MUTED,
            textAnchor="middle",
        )
    )
    return d


def generate_pdf_report(data: dict) -> bytes:
    buf = BytesIO()
    doc = SimpleDocTemplate(
        buf,
        pagesize=A4,
        leftMargin=2 * cm,
        rightMargin=2 * cm,
        topMargin=2 * cm,
        bottomMargin=2 * cm,
    )

    styles = getSampleStyleSheet()
    story = []

    # ── Helpers ──────────────────────────────────────────────────────────────
    def H(
        text,
        size=14,
        color=DARK,
        bold=True,
        align=TA_LEFT,
        space_before=12,
        space_after=6,
    ):
        style = ParagraphStyle(
            "h",
            fontName="Helvetica-Bold" if bold else "Helvetica",
            fontSize=size,
            textColor=color,
            alignment=align,
            spaceAfter=space_after,
            spaceBefore=space_before,
        )
        return Paragraph(text, style)

    def P(text, size=10, color=DARK, align=TA_LEFT, space_after=4):
        style = ParagraphStyle(
            "p",
            fontName="Helvetica",
            fontSize=size,
            textColor=color,
            alignment=align,
            spaceAfter=space_after,
            leading=15,
        )
        return Paragraph(text, style)

    def divider():
        return HRFlowable(
            width="100%", thickness=0.5, color=BORDER, spaceAfter=8, spaceBefore=8
        )

    # ── Header banner ────────────────────────────────────────────────────────
    header_data = [
        [
            Paragraph(
                '<font size="18"><b>GlucoSense AI</b></font><br/>'
                '<font size="9" color="#8B95B0">Diabetes Risk Prediction Report</font>',
                ParagraphStyle(
                    "hdr",
                    fontName="Helvetica-Bold",
                    fontSize=18,
                    textColor=DARK,
                    leading=24,
                ),
            ),
            Paragraph(
                f'<font size="9" color="#8B95B0">Generated: {datetime.now().strftime("%d %B %Y, %H:%M")}<br/>'
                f"Powered by Random Forest ML Model</font>",
                ParagraphStyle(
                    "hdr2",
                    fontName="Helvetica",
                    fontSize=9,
                    textColor=MUTED,
                    alignment=TA_RIGHT,
                    leading=14,
                ),
            ),
        ]
    ]
    header_tbl = Table(header_data, colWidths=["60%", "40%"])
    header_tbl.setStyle(
        TableStyle(
            [
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("BACKGROUND", (0, 0), (-1, -1), DARK),
                ("ROUNDEDCORNERS", [8]),
                ("TOPPADDING", (0, 0), (-1, -1), 16),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 16),
                ("LEFTPADDING", (0, 0), (0, -1), 18),
                ("RIGHTPADDING", (-1, 0), (-1, -1), 18),
            ]
        )
    )
    story.append(header_tbl)
    story.append(Spacer(1, 16))

    # ── Risk score section ───────────────────────────────────────────────────
    result = data.get("result", {})
    risk_score = float(result.get("risk_score", data.get("risk_score", 0)))
    features = result.get("features", data.get("features", {}))
    contribs = result.get("contributions", {})
    whatif = result.get("whatif", [])
    statuses = result.get("statuses", {})
    raw_model_accuracy = result.get("model_accuracy")
    if isinstance(raw_model_accuracy, (int, float)):
        model_accuracy_text = f"{float(raw_model_accuracy):.2f}%"
    else:
        model_accuracy_text = "N/A"
    r_color, r_bg, r_label = _risk_color(risk_score)

    gauge = _gauge_drawing(risk_score)

    risk_info = Paragraph(
        f'<b><font size="14">{r_label}</font></b><br/><br/>'
        f'<font size="10" color="#555">Risk Score: <b>{risk_score:.0f} / 100</b><br/>'
        f"Probability: <b>{float(result.get('probability', risk_score / 100)):.2%}</b><br/>"
        f"Model Accuracy: <b>{model_accuracy_text}</b></font>",
        ParagraphStyle(
            "ri", fontName="Helvetica", fontSize=12, textColor=DARK, leading=20
        ),
    )

    risk_tbl = Table([[gauge, risk_info]], colWidths=["45%", "55%"])
    risk_tbl.setStyle(
        TableStyle(
            [
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("BACKGROUND", (0, 0), (-1, -1), r_bg),
                ("ROUNDEDCORNERS", [8]),
                ("TOPPADDING", (0, 0), (-1, -1), 14),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 14),
                ("LEFTPADDING", (0, 0), (-1, -1), 12),
                ("RIGHTPADDING", (0, 0), (-1, -1), 12),
            ]
        )
    )
    story.append(KeepTogether([H("Risk Assessment", size=12), risk_tbl]))
    story.append(Spacer(1, 14))

    # ── Biomarkers table ─────────────────────────────────────────────────────
    story.append(divider())
    story.append(H("Patient Biomarkers", size=12))

    feat_map = {
        "Glucose": "Glucose (mg/dL)",
        "BloodPressure": "Blood Pressure (mmHg)",
        "BMI": "BMI (kg/m²)",
        "Insulin": "Insulin (μU/mL)",
        "Age": "Age (years)",
        "Pregnancies": "Pregnancies",
        "DiabetesPedigreeFunction": "Diabetes Pedigree",
        "SkinThickness": "Skin Thickness (mm)",
    }
    bio_rows = [["Biomarker", "Value", "Status"]]
    for key, label in feat_map.items():
        val = features.get(key, "—")
        st = statuses.get(key, {})
        status_text = st.get("status", "—")
        status_clr = {"Normal": TEAL, "Borderline": AMBER, "High": RED}.get(
            status_text, MUTED
        )
        bio_rows.append(
            [
                Paragraph(
                    label,
                    ParagraphStyle(
                        "bl", fontName="Helvetica", fontSize=9, textColor=DARK
                    ),
                ),
                Paragraph(
                    f"<b>{val:.1f}</b>" if isinstance(val, float) else f"<b>{val}</b>",
                    ParagraphStyle(
                        "bv", fontName="Helvetica-Bold", fontSize=9, textColor=DARK
                    ),
                ),
                Paragraph(
                    f"<b>{status_text}</b>",
                    ParagraphStyle(
                        "bs",
                        fontName="Helvetica-Bold",
                        fontSize=9,
                        textColor=status_clr,
                    ),
                ),
            ]
        )

    bio_tbl = Table(bio_rows, colWidths=["50%", "25%", "25%"])
    bio_tbl.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), DARK),
                ("TEXTCOLOR", (0, 0), (-1, 0), WHITE),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTSIZE", (0, 0), (-1, 0), 9),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [LIGHT, WHITE]),
                ("GRID", (0, 0), (-1, -1), 0.3, BORDER),
                ("TOPPADDING", (0, 0), (-1, -1), 7),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 7),
                ("LEFTPADDING", (0, 0), (-1, -1), 10),
            ]
        )
    )
    story.append(bio_tbl)
    story.append(Spacer(1, 14))

    # ── Feature contributions ────────────────────────────────────────────────
    if contribs:
        story.append(divider())
        story.append(H("Feature Contributions (Why this prediction?)", size=12))
        sorted_c = sorted(contribs.items(), key=lambda x: abs(x[1]), reverse=True)
        max_c = max(abs(v) for _, v in sorted_c) or 1

        contrib_rows = [["Feature", "Contribution to Risk", "Impact"]]
        for feat, val in sorted_c[:6]:
            bar_pct = abs(val) / max_c
            bar_filled = int(bar_pct * 20)
            bar = "█" * bar_filled + "░" * (20 - bar_filled)
            impact = "↑ Increases" if val > 0 else "↓ Decreases"
            clr = RED if val > 0 else TEAL
            bar_color = "#FF6B6B" if val > 0 else "#00C896"
            contrib_rows.append(
                [
                    feat_map.get(feat, feat),
                    Paragraph(
                        f'<font color="{bar_color}">{bar}</font> {val:+.1f}%',
                        ParagraphStyle("cb", fontName="Courier", fontSize=8),
                    ),
                    Paragraph(
                        f"<b>{impact}</b>",
                        ParagraphStyle(
                            "ci", fontName="Helvetica-Bold", fontSize=9, textColor=clr
                        ),
                    ),
                ]
            )

        c_tbl = Table(contrib_rows, colWidths=["30%", "45%", "25%"])
        c_tbl.setStyle(
            TableStyle(
                [
                    ("BACKGROUND", (0, 0), (-1, 0), DARK2),
                    ("TEXTCOLOR", (0, 0), (-1, 0), WHITE),
                    ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                    ("FONTSIZE", (0, 0), (-1, 0), 9),
                    ("ROWBACKGROUNDS", (0, 1), (-1, -1), [LIGHT, WHITE]),
                    ("GRID", (0, 0), (-1, -1), 0.3, BORDER),
                    ("TOPPADDING", (0, 0), (-1, -1), 6),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
                    ("LEFTPADDING", (0, 0), (-1, -1), 10),
                ]
            )
        )
        story.append(c_tbl)
        story.append(Spacer(1, 14))

    # ── What-If scenarios ────────────────────────────────────────────────────
    if whatif:
        story.append(divider())
        story.append(H("What-If Improvement Scenarios", size=12))
        story.append(
            P(
                "Estimated risk reduction if you make these lifestyle changes:",
                size=9,
                color=MUTED,
            )
        )

        wi_rows = [["Scenario", "Current Score", "Improved Score", "Risk Reduction"]]
        for sc in whatif:
            wi_rows.append(
                [
                    sc["label"],
                    f"{sc['original_score']:.0f}%",
                    f"{sc['new_score']:.0f}%",
                    Paragraph(
                        f"<b>-{sc['delta']:.1f}%</b>",
                        ParagraphStyle(
                            "wd", fontName="Helvetica-Bold", fontSize=9, textColor=TEAL
                        ),
                    ),
                ]
            )

        wi_tbl = Table(wi_rows, colWidths=["40%", "20%", "20%", "20%"])
        wi_tbl.setStyle(
            TableStyle(
                [
                    ("BACKGROUND", (0, 0), (-1, 0), DARK2),
                    ("TEXTCOLOR", (0, 0), (-1, 0), WHITE),
                    ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                    ("FONTSIZE", (0, 0), (-1, 0), 9),
                    ("ROWBACKGROUNDS", (0, 1), (-1, -1), [LIGHT, WHITE]),
                    ("GRID", (0, 0), (-1, -1), 0.3, BORDER),
                    ("ALIGN", (1, 0), (-1, -1), "CENTER"),
                    ("TOPPADDING", (0, 0), (-1, -1), 7),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 7),
                    ("LEFTPADDING", (0, 0), (-1, -1), 10),
                ]
            )
        )
        story.append(wi_tbl)
        story.append(Spacer(1, 14))

    # ── Health Recommendations ───────────────────────────────────────────────
    story.append(divider())
    story.append(H("Personalized Health Recommendations", size=12))

    tips = _generate_tips(features, risk_score)
    for tip in tips:
        icon = (
            "⚠"
            if tip["level"] == "warn"
            else ("✗" if tip["level"] == "danger" else "✓")
        )
        tip_color = (
            AMBER
            if tip["level"] == "warn"
            else (RED if tip["level"] == "danger" else TEAL)
        )
        tip_bg = (
            AMB_BG
            if tip["level"] == "warn"
            else (RED_BG if tip["level"] == "danger" else TEAL_BG)
        )
        tip_tbl = Table(
            [
                [
                    Paragraph(
                        f"<b>{icon}  {tip['text']}</b>",
                        ParagraphStyle(
                            "tt",
                            fontName="Helvetica",
                            fontSize=9,
                            textColor=tip_color,
                            leading=14,
                        ),
                    )
                ]
            ],
            colWidths=["100%"],
        )
        tip_tbl.setStyle(
            TableStyle(
                [
                    ("BACKGROUND", (0, 0), (-1, -1), tip_bg),
                    ("LEFTPADDING", (0, 0), (-1, -1), 12),
                    ("TOPPADDING", (0, 0), (-1, -1), 8),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
                    ("ROUNDEDCORNERS", [6]),
                ]
            )
        )
        story.append(tip_tbl)
        story.append(Spacer(1, 5))

    # ── Disclaimer ───────────────────────────────────────────────────────────
    story.append(Spacer(1, 16))
    story.append(divider())
    story.append(
        P(
            "<b>Disclaimer:</b> This report is generated by an AI/ML model for educational and screening purposes only. "
            "It is <b>not</b> a medical diagnosis. Please consult a qualified healthcare professional for proper medical advice.",
            size=8,
            color=MUTED,
        )
    )

    doc.build(story)
    return buf.getvalue()


def _generate_tips(features, score):
    tips = []
    g = features.get("Glucose", 100)
    b = features.get("BMI", 25)
    bp = features.get("BloodPressure", 72)
    dpf = features.get("DiabetesPedigreeFunction", 0.3)
    ins = features.get("Insulin", 80)

    if g >= 126:
        tips.append(
            {
                "level": "danger",
                "text": "Glucose ≥126 mg/dL may indicate diabetes. Seek medical evaluation immediately.",
            }
        )
    elif g >= 100:
        tips.append(
            {
                "level": "warn",
                "text": "Pre-diabetic glucose range (100–125 mg/dL). Reduce refined carbs and sugar intake.",
            }
        )
    else:
        tips.append(
            {
                "level": "ok",
                "text": "Glucose is in the normal range. Maintain a balanced diet.",
            }
        )

    if b >= 30:
        tips.append(
            {
                "level": "danger",
                "text": "BMI indicates obesity. Aim for 150 min/week of moderate aerobic exercise.",
            }
        )
    elif b >= 25:
        tips.append(
            {
                "level": "warn",
                "text": "Overweight BMI. Consider dietary adjustments and regular physical activity.",
            }
        )
    else:
        tips.append(
            {
                "level": "ok",
                "text": "BMI is in the healthy range. Keep up the good work!",
            }
        )

    if bp >= 90:
        tips.append(
            {
                "level": "danger",
                "text": "Elevated blood pressure. Reduce sodium intake and consult a doctor.",
            }
        )
    elif bp >= 80:
        tips.append(
            {
                "level": "warn",
                "text": "Borderline blood pressure. Monitor regularly and limit salt.",
            }
        )

    if dpf >= 1.0:
        tips.append(
            {
                "level": "warn",
                "text": "High family history factor. Annual diabetes screenings are strongly recommended.",
            }
        )

    if score >= 60:
        tips.append(
            {
                "level": "danger",
                "text": "High risk score. Schedule a comprehensive diabetes screening with your doctor.",
            }
        )
    elif score >= 30:
        tips.append(
            {
                "level": "warn",
                "text": "Moderate risk. Focus on diet, exercise, and regular health checkups.",
            }
        )
    else:
        tips.append(
            {
                "level": "ok",
                "text": "Low overall risk. Continue healthy lifestyle habits.",
            }
        )

    return tips
