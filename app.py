import streamlit as st
import requests
from bs4 import BeautifulSoup
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from summa import summarizer as textrank_summarizer
import torch

device = "cuda" if torch.cuda.is_available() else "cpu"

# Load model
vit5_pretrained = "VietAI/vit5-base-vietnews-summarization"
model_finetuned_path = "nguyenduc05/ViT5_finetuned"

# tokenizer_vit5 = AutoTokenizer.from_pretrained(vit5_pretrained)
# model_vit5 = AutoModelForSeq2SeqLM.from_pretrained(vit5_pretrained)

# tokenizer_ft = AutoTokenizer.from_pretrained(model_finetuned_path)
# model_ft = AutoModelForSeq2SeqLM.from_pretrained(model_finetuned_path)

# Crawl data
def extract_text_from_url(url):
    res = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, verify=False)
    soup = BeautifulSoup(res.text, "html.parser")

    # VNExpress, Dân trí, Zing... thường có <p>
    paragraphs = [p.get_text() for p in soup.find_all("p")]
    text = " ".join(paragraphs)
    return text.strip()

# Summarize ViT5
def summarize_vit5(text, tokenizer, model, num_sent=3):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
    summary_ids = model.generate(
        inputs["input_ids"],
        max_length=50 * num_sent,
        num_beams=4,
        early_stopping=True
    )
    return tokenizer.decode(summary_ids[0], skip_special_tokens=True)

# Streamlit UI
st.set_page_config(page_title="Vietnamese News Summarizer", layout="wide")
st.title("📰 Vietnamese News Summarizer")

# Chọn input mode
mode = st.radio("Chọn chế độ nhập:", ["Nhập văn bản", "Link báo"])

# Nhập văn bản hoặc link
text_input = ""
if mode == "Nhập văn bản":
    text_input = st.text_area("Nhập đoạn văn cần tóm tắt:", height=200)
elif mode == "Link báo":
    link_input = st.text_input("Nhập link bài báo:")
    if link_input:
        text_input = extract_text_from_url(link_input)

# Chọn phương pháp
model_choice = st.selectbox("Chọn phương pháp tóm tắt:",
                            ["TextRank", "ViT5 (pretrained)", "ViT5 (fine-tuned)"])

# Số câu muốn tóm tắt
num_sent = st.slider("Số lượng câu tóm tắt:", 1, 10, 3)

# Nút chạy
if st.button("🚀 Tóm tắt ngay"):
    if not text_input.strip():
        st.warning("⚠️ Bạn chưa nhập nội dung hoặc link hợp lệ.")
    else:
        if model_choice == "TextRank":
            summary = textrank_summarizer.summarize(text_input, ratio=num_sent * 0.1)
            if not summary:
                summary = "❌ TextRank không tạo được tóm tắt (text quá ngắn)."
        # elif model_choice == "ViT5 (pretrained)":
        #     summary = summarize_vit5(text_input, tokenizer_vit5, model_vit5, num_sent)
        # elif model_choice == "ViT5 (fine-tuned)":
        #     summary = summarize_vit5(text_input, tokenizer_ft, model_ft, num_sent)
        else:
            summary = "❌ Model chưa hỗ trợ."

        st.subheader("✨ Kết quả tóm tắt")
        st.write(summary)
