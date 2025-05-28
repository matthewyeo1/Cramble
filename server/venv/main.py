from fastapi import FastAPI, File, UploadFile, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from io import BytesIO
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from reportlab.lib.units import inch
import subprocess
import traceback
import fitz
import re
from tuning import load_model_and_tokenizer, apply_lora, train_lora, load_lora_model, generate_text
from datasets import load_dataset

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
async def load_models_once():
    print("[STARTUP] Loading model and tokenizer...")
    model, tokenizer = load_model_and_tokenizer()
    lora_model = load_lora_model("EleutherAI/gpt-neo-125m", "lora_output")

    # Save into app state
    app.state.model = lora_model
    app.state.tokenizer = tokenizer
    print("[STARTUP] Model and tokenizer loaded.")

def text_to_pdf_bytes(text: str) -> BytesIO:
    buffer = BytesIO()
    c = canvas.Canvas(buffer, pagesize=letter)
    width, height = letter

    left_margin = 0.5 * inch
    right_margin = 0.5 * inch
    top_margin = height - 0.5 * inch
    bottom_margin = 0.5 * inch
    usable_width = width - left_margin - right_margin
    y = top_margin

    font_name = "Helvetica"
    font_size = 7
    line_height = font_size + 1
    c.setFont(font_name, font_size)

    text_obj = c.beginText()
    text_obj.setTextOrigin(left_margin, y)
    text_obj.setFont(font_name, font_size)

    # Function to split a paragraph into lines without breaking words
    def wrap_line(line: str):
        words = line.split()
        lines = []
        current_line = ""
        for word in words:
            # Check width of current line + word + space
            test_line = current_line + (" " if current_line else "") + word
            if c.stringWidth(test_line, font_name, font_size) <= usable_width:
                current_line = test_line
            else:
                if current_line:
                    lines.append(current_line)
                current_line = word
        if current_line:
            lines.append(current_line)
        return lines

    for paragraph in text.split('\n'):
        if paragraph.strip() == "":
            # Empty line = paragraph break
            y -= line_height
            if y <= bottom_margin:
                c.drawText(text_obj)
                c.showPage()
                text_obj = c.beginText()
                text_obj.setTextOrigin(left_margin, top_margin)
                text_obj.setFont(font_name, font_size)
                y = top_margin
            else:
                text_obj.textLine("")  # blank line
            continue

        wrapped_lines = wrap_line(paragraph)
        for line in wrapped_lines:
            if y <= bottom_margin:
                c.drawText(text_obj)
                c.showPage()
                text_obj = c.beginText()
                text_obj.setTextOrigin(left_margin, top_margin)
                text_obj.setFont(font_name, font_size)
                y = top_margin

            text_obj.textLine(line)
            y -= line_height

    c.drawText(text_obj)
    c.save()
    buffer.seek(0)
    return buffer

def preprocess_text(text: str) -> str:
    lines = text.splitlines()
    cleaned_lines = []
    for line in lines:
        line = line.strip()
        if line == "":
            cleaned_lines.append("")  # Keep paragraph breaks
        elif cleaned_lines and not cleaned_lines[-1].endswith(('.', ':', ';')):
            cleaned_lines[-1] += " " + line
        else:
            cleaned_lines.append(line)
    return '\n'.join(cleaned_lines)

def extract_text_with_pymupdf(pdf_path: str) -> str:
    doc = fitz.open(pdf_path)
    text = ""
    for page in doc:
        page_text = page.get_text("text")  # better at preserving symbols
        text += page_text + "\n"
    return text

intro_phrases = [
    "the article discusses", "here are some key points", "this article provides",
    "the passage states", "this text outlines", "this document covers", "cheatsheet:"
    "key definitions", "it seems"
]
conclusion_phrases = [
    "in conclusion", "to conclude", "overall", "in summary", "summarizing",
    "to summarize", "as a whole", "to sum up", "finally", "conclusion:"
]

def remove_intro_and_conclusion(text: str) -> str:
    lines = text.strip().splitlines()
    filtered_lines = []

    # Skip intro lines
    skipping_intro = True
    for line in lines:
        lower = line.strip().lower()
        if skipping_intro and any(lower.startswith(p) for p in intro_phrases):
            continue
        skipping_intro = False

        if any(lower.startswith(p) for p in conclusion_phrases):
            break

        filtered_lines.append(line)

    return '\n'.join(filtered_lines).strip()

def strip_article_phrases(text: str) -> str:
    # Remove known undesired phrases
    text = re.sub(r"(?i)\bthis (article|passage|text|document) (says|discusses|provides|covers)[^\n]*", "", text)
    text = re.sub(r"(?i)\bhere (are|is) (some )?key points[^\n]*", "", text)
    return text.strip()

def starts_with_phrase(sentence, phrases):
    sentence_lower = sentence.lower().strip()
    return any(sentence_lower.startswith(phrase) for phrase in phrases)

def clean_summary(text):
    # Split into sentences
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    
    # Remove first sentence if it matches intro phrases
    if sentences and starts_with_phrase(sentences[0], intro_phrases):
        print(f"[LOG] Removed intro sentence: {sentences[0]}")
        sentences = sentences[1:]

    # Remove last sentence if it matches conclusion phrases
    if sentences and starts_with_phrase(sentences[-1], conclusion_phrases):
        print(f"[LOG] Removed conclusion sentence: {sentences[-1]}")
        sentences = sentences[:-1]

    return ' '.join(sentences)
    
@app.post("/upload")
async def upload(request: Request, file: UploadFile = File(...)):
    try:
        print(f"Received file: {file.filename}, content type: {file.content_type}")
        contents = await file.read()
        with open("temp.pdf", "wb") as f:
            f.write(contents)

        text = extract_text_with_pymupdf("temp.pdf")

        print(f"Extracted text length: {len(text)}")

        if not text.strip():
            return JSONResponse(content={"cheatsheet": "Failed to extract text from PDF."}, status_code=400)

        cleaned_text = preprocess_text(text)

        prompt = (
            "OBJECTIVE: Summarize the following content into a concise cheatsheet focusing on key definitions, formulas, and concepts.\n"
            "STRICT RULES:\n"
            "• Do NOT refer to the information given to you as an article, passage, text, etc.\n"
            "• Do NOT include any greetings, pleasantries, or meta descriptions (e.g., 'This article says...').\n"
            "• ONLY include key points, not summaries.\n"
            "• Replace all variable names with descriptive words for clarity.\n"
            "• Use bullet points and concise phrasing.\n"
            "• Separate each bullet point.\n"
            "• Use LaTeX-style notation for mathematical content.\n"
            "• DO NOT include introductions or conclusions.\n"
            "OUTPUT: Only the cheatsheet content.\n\n" + cleaned_text[:2048]       # Test value, varies from model to model
        )

        model = request.app.state.model
        tokenizer = request.app.state.tokenizer

        cheatsheet = generate_text(model, tokenizer, prompt)
        
        if not cheatsheet:
            return JSONResponse(content={"error": "Failed to generate cheatsheet"}, status_code=500)
        
        '''
        if cheatsheet.startswith(prompt.strip()):
            cheatsheet = cheatsheet[len(prompt.strip()):].strip()
        '''
        if prompt.strip() in cheatsheet:
            cheatsheet = cheatsheet.split(prompt.strip(), 1)[-1].strip()

        def remove_repeating_blocks(text, min_len=20):
            lines = text.strip().split('\n')
            seen = set()
            output = []
            for line in lines:
                line_stripped = line.strip()
                if len(line_stripped) >= min_len and line_stripped in seen:
                    continue
                seen.add(line_stripped)
                output.append(line)
            return '\n'.join(output).strip()

        cheatsheet = remove_repeating_blocks(cheatsheet)
        cheatsheet = strip_article_phrases(cheatsheet)
        cheatsheet = remove_intro_and_conclusion(cheatsheet)
        cheatsheet = clean_summary(cheatsheet)
        pdf_buffer = text_to_pdf_bytes(cheatsheet)

        return StreamingResponse(
            pdf_buffer,
            media_type="application/pdf",
            headers={"Content-Disposition": "attachment; filename=cheatsheet.pdf"}
        )

    except Exception as e:
        traceback.print_exc()
        return JSONResponse(content={"error": str(e)}, status_code=500)
