from io import BytesIO
from html import escape
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle


def csv_bytes(result) -> bytes:
    """Export annotated reviews as UTF-8 CSV, preserving metadata."""
    return result.reviews.to_csv(index=False).encode("utf-8-sig")


def pdf_bytes(result) -> bytes:
    """Create a paginated evidence report for a single review or batch."""
    buf = BytesIO()
    styles = getSampleStyleSheet()
    doc = SimpleDocTemplate(
        buf, title="ReviewGuard AI Forensics Report", author="ReviewGuard AI"
    )
    rows = [
        Paragraph("ReviewGuard AI | Forensics report", styles["Title"]),
        Spacer(1, 14),
        Paragraph(
            "Research screening only. A suspicion score is not proof of deception. Review flagged items manually.",
            styles["Normal"],
        ),
        Spacer(1, 12),
    ]
    if hasattr(result, "reviews"):
        df = result.reviews
        rows.append(
            Paragraph(
                f"{len(df)} reviews | {len(result.clusters)} similarity clusters | Backend: {escape(result.embedding_backend)}",
                styles["Normal"],
            )
        )
        high = int(df.risk_band.isin(["High", "Very High"]).sum())
        rows.append(Spacer(1, 10))
        rows.append(
            Paragraph(
                f"Mean suspicion: {df.suspicion_score.mean():.1f}/100 | High / very high: {high} | Similarity pairs: {len(result.edges)}",
                styles["Normal"],
            )
        )
        from reportlab.graphics.shapes import Drawing, Rect, String

        chart = Drawing(450, 110)
        counts = [
            int((df.risk_band == b).sum())
            for b in ["Low", "Moderate", "High", "Very High"]
        ]
        for i, (band, count, color) in enumerate(
            zip(
                ["Low", "Moderate", "High", "Very High"],
                counts,
                ["#12b886", "#f59f00", "#f76707", "#e64980"],
            )
        ):
            height = 75 * count / max(max(counts), 1)
            x = 30 + i * 108
            chart.add(
                Rect(
                    x,
                    20,
                    65,
                    height,
                    fillColor=colors.HexColor(color),
                    strokeColor=None,
                )
            )
            chart.add(String(x, 5, band, fontSize=9))
            chart.add(String(x + 20, 24 + height, str(count), fontSize=9))
        rows.extend(
            [
                Spacer(1, 10),
                chart,
                Paragraph(
                    "Score distribution by operational risk band",
                    styles["Caption"] if "Caption" in styles else styles["Normal"],
                ),
            ]
        )
        if result.clusters:
            rows.append(Paragraph("Similarity clusters", styles["Heading2"]))
            for c in result.clusters[:10]:
                rows.append(
                    Paragraph(
                        f'Cluster {c["cluster_id"]+1}: {c["size"]} reviews; mean similarity {c["mean_similarity"]:.3f}; mean suspicion {c["mean_suspicion"]:.1f}. Shared terms: '
                        + escape(", ".join(c.get("shared_terms", []))),
                        styles["Normal"],
                    )
                )
        rows.append(Spacer(1, 8))
        weight_text = ", ".join(f"{k} {v:.0%}" for k, v in result.weights.items())
        rows.append(
            Paragraph(
                "Method: "
                + weight_text
                + ". Missing signals are omitted and available weights renormalized. Design weights are configurable, not scientific constants. Similarity groups do not establish shared authorship.",
                styles["Normal"],
            )
        )
        items = [
            (int(i) + 1, r["text"], r.suspicion_score, r.risk_band, r.explanation)
            for i, r in df.sort_values("suspicion_score", ascending=False).iterrows()
        ]
    else:
        items = [(1, result.text, result.score, result.band, " ".join(result.reasons))]
    for i, text, score, band, reason in items:
        rows.extend(
            [
                Spacer(1, 14),
                Paragraph(f"Review {i} | {score:.1f}/100 | {band}", styles["Heading2"]),
                Paragraph(escape(text), styles["Normal"]),
                Spacer(1, 6),
                Paragraph(escape(reason), styles["Normal"]),
            ]
        )
    doc.build(rows)
    return buf.getvalue()
