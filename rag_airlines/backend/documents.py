"""Fictional booking confirmations and simulated refund receipts."""

from io import BytesIO
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import A4
from reportlab.lib.colors import HexColor


def pdf_document(title, fields):
    output = BytesIO()
    pdf = canvas.Canvas(output, pagesize=A4)
    width, height = A4
    pdf.setTitle("AeroNova - " + title)
    pdf.setFillColor(HexColor("#f7f7f2"))
    pdf.rect(0, 0, width, height, fill=1, stroke=0)
    pdf.setFillColor(HexColor("#263c35"))
    pdf.setFont("Helvetica-Bold", 25)
    pdf.drawString(45, height - 60, "AeroNova")
    pdf.setFont("Helvetica", 9)
    pdf.drawRightString(width - 45, height - 55, "FICTIONAL DEMO DOCUMENT")
    pdf.setFont("Helvetica-Bold", 23)
    pdf.drawString(45, height - 125, title)
    y = height - 180
    for key, value in fields:
        pdf.setFont("Helvetica", 10)
        pdf.drawString(45, y, str(key))
        pdf.setFont("Helvetica-Bold", 11)
        pdf.drawString(215, y, str(value)[:48])
        pdf.setStrokeColor(HexColor("#dde3da"))
        pdf.line(45, y - 17, width - 45, y - 17)
        y -= 49
    pdf.setFont("Helvetica", 9)
    pdf.drawString(45, 80, "Fictional data. Not a boarding pass, real ticket or proof of payment.")
    pdf.drawString(45, 65, "Simulated refunds do not transfer money.")
    pdf.save()
    return output.getvalue()
