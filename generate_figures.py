"""Generate architecture diagram PNGs using Pillow.

Run:  python generate_figures.py
Output: assets/*.png
"""
import os
from PIL import Image, ImageDraw, ImageFont

# ── Try to load a good font, fallback to default ──
def _get_fonts():
    candidates = [
        "/System/Library/Fonts/Helvetica.ttc",
        "/System/Library/Fonts/SFNSMono.ttf",
        "/Library/Fonts/Arial.ttf",
    ]
    for path in candidates:
        if os.path.exists(path):
            try:
                return {
                    "title": ImageFont.truetype(path, 36),
                    "subtitle": ImageFont.truetype(path, 18),
                    "heading": ImageFont.truetype(path, 22),
                    "body": ImageFont.truetype(path, 17),
                    "small": ImageFont.truetype(path, 14),
                    "tiny": ImageFont.truetype(path, 12),
                    "label": ImageFont.truetype(path, 15),
                }
            except Exception:
                continue
    # fallback
    return {k: ImageFont.load_default() for k in
            ["title", "subtitle", "heading", "body", "small", "tiny", "label"]}

FONTS = _get_fonts()

# ── Colours ──
BG       = "#FAFBFC"
BLUE     = "#3B82F6"
BLUE_D   = "#1E40AF"
GREEN    = "#16A34A"
GREEN_D  = "#15803D"
ORANGE   = "#EA580C"
ORANGE_D = "#C2410C"
PURPLE   = "#7C3AED"
YELLOW   = "#F59E0B"
GRAY_L   = "#F3F4F6"
GRAY_B   = "#D1D5DB"
GRAY_T   = "#6B7280"
TEXT     = "#1F2937"
WHITE    = "#FFFFFF"

# section bg
BLUE_BG   = "#DBEAFE"
GREEN_BG  = "#DCFCE7"
ORANGE_BG = "#FFEDD5"
PURPLE_BG = "#F5F3FF"
YELLOW_BG = "#FEF9C3"


def _rounded_rect(draw, xy, radius, fill, outline=None, width=1):
    """Draw a rounded rectangle."""
    x1, y1, x2, y2 = xy
    r = radius
    draw.rounded_rectangle(xy, radius=r, fill=fill, outline=outline, width=width)


def _box(draw, x, y, w, h, text1, text2=None, bg=BLUE, fg=WHITE,
         outline=None, radius=12, font1=None, font2=None):
    """Draw a rounded box with centered text."""
    font1 = font1 or FONTS["body"]
    font2 = font2 or FONTS["small"]
    _rounded_rect(draw, (x, y, x+w, y+h), radius, fill=bg, outline=outline, width=2 if outline else 0)

    if text2:
        # two lines
        bb1 = draw.textbbox((0, 0), text1, font=font1)
        bb2 = draw.textbbox((0, 0), text2, font=font2)
        tw1 = bb1[2] - bb1[0]
        th1 = bb1[3] - bb1[1]
        tw2 = bb2[2] - bb2[0]
        th2 = bb2[3] - bb2[1]
        gap = 4
        total_h = th1 + gap + th2
        sy = y + (h - total_h) // 2
        draw.text((x + (w - tw1) // 2, sy), text1, fill=fg, font=font1)
        draw.text((x + (w - tw2) // 2, sy + th1 + gap), text2, fill=fg, font=font2)
    else:
        bb = draw.textbbox((0, 0), text1, font=font1)
        tw = bb[2] - bb[0]
        th = bb[3] - bb[1]
        draw.text((x + (w - tw) // 2, y + (h - th) // 2), text1, fill=fg, font=font1)


def _arrow_right(draw, x1, y, x2, color=GRAY_B, width=3):
    """Horizontal arrow →"""
    draw.line([(x1, y), (x2, y)], fill=color, width=width)
    # arrowhead
    draw.polygon([(x2, y), (x2 - 10, y - 6), (x2 - 10, y + 6)], fill=color)


def _arrow_down(draw, x, y1, y2, color=GRAY_B, width=3):
    """Vertical arrow ↓"""
    draw.line([(x, y1), (x, y2)], fill=color, width=width)
    draw.polygon([(x, y2), (x - 6, y2 - 10), (x + 6, y2 - 10)], fill=color)


def _arrow_line(draw, x1, y1, x2, y2, color=GRAY_B, width=3):
    """Diagonal arrow."""
    draw.line([(x1, y1), (x2, y2)], fill=color, width=width)
    # simple arrowhead at end
    import math
    angle = math.atan2(y2 - y1, x2 - x1)
    size = 10
    lx = x2 - size * math.cos(angle - 0.4)
    ly = y2 - size * math.sin(angle - 0.4)
    rx = x2 - size * math.cos(angle + 0.4)
    ry = y2 - size * math.sin(angle + 0.4)
    draw.polygon([(x2, y2), (int(lx), int(ly)), (int(rx), int(ry))], fill=color)


def _centered_text(draw, x, y, w, text, font=None, fill=TEXT):
    font = font or FONTS["body"]
    bb = draw.textbbox((0, 0), text, font=font)
    tw = bb[2] - bb[0]
    draw.text((x + (w - tw) // 2, y), text, fill=fill, font=font)


# ═══════════════════════════════════════════════════════════════
#  Figure 1 — RAG Query Pipeline
# ═══════════════════════════════════════════════════════════════
def fig_query_pipeline():
    W, H = 1400, 820
    img = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(img)

    # Title
    _centered_text(d, 0, 20, W, "RAG Query Pipeline", FONTS["title"], TEXT)
    _centered_text(d, 0, 65, W, "Medical Imaging Multimodal Retrieval-Augmented Generation",
                   FONTS["subtitle"], GRAY_T)

    # ── Section backgrounds ──
    _rounded_rect(d, (30, 110, 280, 520), 16, fill=BLUE_BG, outline="#93C5FD", width=2)
    _rounded_rect(d, (320, 110, 700, 520), 16, fill=GREEN_BG, outline="#86EFAC", width=2)
    _rounded_rect(d, (740, 110, 1080, 520), 16, fill=ORANGE_BG, outline="#FDBA74", width=2)

    # Section labels
    d.text((50, 118), "INPUT", fill=BLUE, font=FONTS["label"])
    d.text((340, 118), "RETRIEVAL", fill=GREEN, font=FONTS["label"])
    d.text((760, 118), "GENERATION", fill=ORANGE, font=FONTS["label"])

    # ── Input ──
    _box(d, 55, 170, 200, 65, "Text Query", "Natural language", bg=BLUE)
    _box(d, 55, 290, 200, 65, "X-ray Image", "Optional upload", bg=BLUE)

    # ── Retrieval ──
    _box(d, 350, 170, 200, 65, "Embedding", "TF-IDF / Gemini", bg=GREEN)
    _box(d, 350, 290, 200, 65, "Gemini Vision", "Image to text desc.", bg=GREEN)
    _box(d, 350, 420, 320, 60, "ChromaDB Vector Search", "Cosine similarity, Top-K chunks", bg=GREEN_D)

    # ── Generation ──
    _box(d, 770, 170, 280, 55, "Prompt Builder", "Context + query + image", bg=ORANGE)
    _box(d, 770, 280, 280, 70, "LLM Generation", "Gemini 2.0 Flash / Ollama", bg=ORANGE)
    _box(d, 770, 410, 280, 60, "Post-processing", "Citations + disclaimer", bg=ORANGE_D)

    # ── Output ──
    _box(d, 350, 600, 500, 70, "Answer with Source Citations", "Gradio UI or CLI",
         bg=PURPLE, radius=16)

    # ── Arrows ──
    _arrow_right(d, 255, 202, 350, color="#64748B")    # query → embedding
    _arrow_right(d, 255, 322, 350, color="#64748B")    # image → vision
    _arrow_down(d, 450, 355, 420, color=GREEN)         # vision ↑ merge
    _arrow_down(d, 450, 235, 290, color=GREEN)         # embedding ↓ (actually up, let's fix)

    # embedding & vision both → chromadb
    _arrow_down(d, 450, 235, 420, color=GREEN)

    # chromadb → prompt builder
    _arrow_line(d, 670, 445, 770, 197, color="#64748B")

    # prompt → LLM → post
    _arrow_down(d, 910, 225, 280, color=ORANGE)
    _arrow_down(d, 910, 350, 410, color=ORANGE)

    # post → output
    _arrow_line(d, 910, 470, 700, 610, color="#64748B")

    # ── Fallback box ──
    _box(d, 1110, 270, 240, 80, "Fallback chain:", "Gemini > Ollama > Local",
         bg="#FFF7ED", fg="#9A3412", outline="#FDBA74", radius=10,
         font1=FONTS["small"], font2=FONTS["small"])

    # ── Legend ──
    ly = 750
    for i, (label, color) in enumerate([
        ("Input", BLUE), ("Retrieval", GREEN),
        ("Generation", ORANGE), ("Output", PURPLE),
    ]):
        lx = 350 + i * 170
        _rounded_rect(d, (lx, ly, lx + 30, ly + 20), 4, fill=color)
        d.text((lx + 40, ly), label, fill=TEXT, font=FONTS["small"])

    img.save("assets/rag_query_pipeline.png", dpi=(200, 200))
    print("  Saved: assets/rag_query_pipeline.png")


# ═══════════════════════════════════════════════════════════════
#  Figure 2 — Data Ingestion Pipeline
# ═══════════════════════════════════════════════════════════════
def fig_ingestion_pipeline():
    W, H = 1400, 580
    img = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(img)

    _centered_text(d, 0, 20, W, "Data Ingestion Pipeline", FONTS["title"], TEXT)
    _centered_text(d, 0, 65, W, "One-time indexing of radiology reports into vector database",
                   FONTS["subtitle"], GRAY_T)

    # ── 5 stages ──
    stages = [
        ("Data Sources", "HuggingFace / JSON", BLUE),
        ("Data Loader", "MedicalReport obj.", BLUE),
        ("Chunking", "Section-based split", GREEN),
        ("Embedding", "TF-IDF / Gemini", GREEN),
        ("ChromaDB", "Persistent store", YELLOW),
    ]

    bw, bh = 210, 75
    gap = 40
    sx = (W - (5 * bw + 4 * gap)) // 2
    sy = 130

    for i, (label, sub, bg) in enumerate(stages):
        x = sx + i * (bw + gap)
        _box(d, x, sy, bw, bh, label, sub, bg=bg, fg=WHITE if bg != YELLOW else TEXT)
        if i > 0:
            _arrow_right(d, x - gap, sy + bh // 2, x, color="#64748B")

    # ── Detail items ──
    details = [
        ["openi_synthetic/reports.json", "OpenI dataset (HF)", "Auto-download"],
        ["Parse JSON fields", "Filter empty reports", "MedicalReport dataclass"],
        ["indication", "findings", "impression", "full_text"],
        ["TF-IDF to 768-dim", "Gemini text-embedding-004", "Batch size: 100"],
        ["Cosine similarity", "Persistent storage", "Skip-if-indexed"],
    ]

    for i, items in enumerate(details):
        x = sx + i * (bw + gap)
        for j, item in enumerate(items):
            dy = 240 + j * 36
            _box(d, x + 8, dy, bw - 16, 30, item,
                 bg="#F8FAFC", fg="#334155", outline="#E2E8F0", radius=6,
                 font1=FONTS["tiny"])

    # ── Metadata note ──
    note = "Each chunk carries metadata:   uid   |   section   |   has_images"
    _box(d, 300, 430, 800, 40, note, bg="#EEF2FF", fg="#4338CA",
         outline="#C7D2FE", radius=8, font1=FONTS["small"])

    # ── Down arrows from stages to details ──
    for i in range(5):
        x = sx + i * (bw + gap) + bw // 2
        _arrow_down(d, x, sy + bh, 240, color=GRAY_B)

    img.save("assets/data_ingestion_pipeline.png", dpi=(200, 200))
    print("  Saved: assets/data_ingestion_pipeline.png")


# ═══════════════════════════════════════════════════════════════
#  Figure 3 — Evaluation System
# ═══════════════════════════════════════════════════════════════
def fig_evaluation_system():
    W, H = 1200, 760
    img = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(img)

    _centered_text(d, 0, 20, W, "Evaluation System", FONTS["title"], TEXT)
    _centered_text(d, 0, 65, W, "RAGAS-style metrics + LLM-as-Judge",
                   FONTS["subtitle"], GRAY_T)

    # ── Input ──
    _box(d, 40, 150, 220, 65, "Golden QA Dataset", "40 questions, 5 types", bg=BLUE)
    _box(d, 40, 270, 220, 65, "RAG Pipeline", "Query each question", bg=BLUE)
    _arrow_down(d, 150, 215, 270, color=BLUE)

    # ── Retrieval Metrics ──
    _rounded_rect(d, (300, 125, 600, 380), 14, fill=GREEN_BG, outline="#86EFAC", width=2)
    d.text((315, 133), "RETRIEVAL METRICS", fill=GREEN, font=FONTS["label"])
    d.text((315, 153), "No API needed", fill=GRAY_T, font=FONTS["tiny"])
    _box(d, 325, 185, 250, 55, "Context Precision", "retrieved & relevant / retrieved",
         bg=GREEN, font1=FONTS["small"], font2=FONTS["tiny"])
    _box(d, 325, 275, 250, 55, "Context Recall", "retrieved & relevant / relevant",
         bg=GREEN, font1=FONTS["small"], font2=FONTS["tiny"])

    # ── Generation Metrics ──
    _rounded_rect(d, (640, 125, 940, 380), 14, fill=ORANGE_BG, outline="#FDBA74", width=2)
    d.text((655, 133), "GENERATION METRICS", fill=ORANGE, font=FONTS["label"])
    d.text((655, 153), "LLM required", fill=GRAY_T, font=FONTS["tiny"])
    _box(d, 665, 185, 250, 55, "Faithfulness", "Hallucination detection",
         bg=ORANGE, font1=FONTS["small"], font2=FONTS["tiny"])
    _box(d, 665, 275, 250, 55, "Answer Relevancy", "On-topic assessment",
         bg=ORANGE, font1=FONTS["small"], font2=FONTS["tiny"])

    # Arrows from pipeline to metrics
    _arrow_line(d, 260, 290, 325, 210, color="#64748B")
    _arrow_line(d, 260, 310, 325, 300, color="#64748B")
    _arrow_line(d, 260, 290, 665, 210, color="#64748B")
    _arrow_line(d, 260, 310, 665, 300, color="#64748B")

    # ── LLM Judge ──
    _rounded_rect(d, (200, 420, 950, 560), 14, fill=PURPLE_BG, outline="#C4B5FD", width=2)
    d.text((220, 428), "LLM-AS-JUDGE  (single LLM call)", fill=PURPLE, font=FONTS["label"])

    _box(d, 220, 470, 220, 55, "Medical", "Appropriateness", bg=PURPLE,
         font1=FONTS["small"], font2=FONTS["small"])
    _box(d, 470, 470, 200, 55, "Citation", "Accuracy", bg=PURPLE,
         font1=FONTS["small"], font2=FONTS["small"])
    _box(d, 700, 470, 220, 55, "Answer", "Completeness", bg=PURPLE,
         font1=FONTS["small"], font2=FONTS["small"])

    # Arrow from pipeline down to judge
    _arrow_down(d, 150, 335, 495, color="#64748B")
    _arrow_right(d, 150, 495, 220, color="#64748B")

    # ── Output ──
    _box(d, 280, 620, 550, 60, "HTML Report + JSON Results",
         "Radar chart | bar charts | summary | per-question",
         bg=YELLOW, fg=TEXT, radius=12)

    _arrow_down(d, 450, 380, 620, color="#64748B")
    _arrow_down(d, 575, 560, 620, color="#64748B")

    # Score note
    _box(d, 900, 625, 220, 50, "Score: 0.0 – 1.0", "-1.0 = skipped",
         bg="#FFF7ED", fg="#9A3412", outline="#FDBA74", radius=8,
         font1=FONTS["small"], font2=FONTS["tiny"])

    img.save("assets/evaluation_system.png", dpi=(200, 200))
    print("  Saved: assets/evaluation_system.png")


if __name__ == "__main__":
    os.makedirs("assets", exist_ok=True)
    fig_query_pipeline()
    fig_ingestion_pipeline()
    fig_evaluation_system()
    print("\nDone! PNGs saved in assets/")
