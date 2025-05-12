import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import gradio as gr

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load tokenizer and model
model_dir = "summrizer_model"
tokenizer = AutoTokenizer.from_pretrained(model_dir)
model = AutoModelForSeq2SeqLM.from_pretrained(model_dir).to(device)

# Your custom summarization function with ratio
def summarize_paragraph(paragraph, ratio=0.47):
    tokens = tokenizer.encode_plus(
        paragraph,
        max_length=512,
        truncation=True,
        padding='max_length',
        return_tensors='pt'
    )
    input_ids = tokens['input_ids'].to(device)
    attention_mask = tokens['attention_mask'].to(device)

    target_length = max(1, int(ratio * len(tokenizer.tokenize(paragraph))))

    summary_tokens = model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        max_length=target_length
    )
    summary = tokenizer.decode(summary_tokens[0], skip_special_tokens=True)
    return summary

interface = gr.Interface(
    fn=summarize_paragraph,
    inputs=[
        gr.Textbox(label="Enter Arabic Text", lines=10),
        gr.Slider(0.1, 1.0, value=0.47, step=0.01, label="Summary Ratio")
    ],
    outputs=gr.Textbox(label="Summary"),
    title="Arabic Text Summarizer",
    allow_flagging="never",
    description="Paste Arabic text and adjust the summary ratio to get a concise result."
)

interface.launch()


