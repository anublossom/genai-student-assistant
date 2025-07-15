
import gradio as gr
import fitz  # PyMuPDF
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM

# Load FLAN-T5 for summary/quiz
flan_pipeline = pipeline("text2text-generation", model="google/flan-t5-base")

# Load English-to-Tamil translator
translator_checkpoint = "suriya7/English-to-Tamil"
ta_tokenizer = AutoTokenizer.from_pretrained(translator_checkpoint)
ta_model = AutoModelForSeq2SeqLM.from_pretrained(translator_checkpoint)

def translate_to_tamil(text):
    tokenized = ta_tokenizer([text], return_tensors='pt', truncation=True)
    out = ta_model.generate(**tokenized, max_length=128)
    return ta_tokenizer.decode(out[0], skip_special_tokens=True)

def extract_text(file):
    text = ""
    try:
        with fitz.open(file.name) as doc:
            for i, page in enumerate(doc):
                if i >= 3:
                    break
                text += page.get_text()
    except Exception as e:
        return f"⚠️ Error reading file: {e}"
    return text.strip()

def process(file, task):
    if not file:
        return "⚠️ Please upload a PDF file."
    
    text = extract_text(file)
    if not text:
        return "⚠️ No readable text found in PDF."

    try:
        if task == "Summarize":
            prompt = "Summarize this content:
" + text[:2000]
            result = flan_pipeline(prompt, max_length=512)[0]['generated_text']
            return result
        
        elif task == "Generate Quiz":
            prompt = "Generate 5 quiz questions from this content:
" + text[:2000]
            result = flan_pipeline(prompt, max_length=512)[0]['generated_text']
            return result

        elif task == "Translate to Tamil":
            return translate_to_tamil(text[:512])

    except Exception as e:
        return f"⚠️ Error during processing: {e}"

gr.Interface(
    fn=process,
    inputs=[
        gr.File(label="📂 Upload PDF"),
        gr.Radio(["Summarize", "Generate Quiz", "Translate to Tamil"], label="📌 Select Task")
    ],
    outputs="text",
    title="📚 Student Assistant (English + Tamil)",
    description="Upload a short PDF (1–5 pages, < 5MB)."
).launch()
