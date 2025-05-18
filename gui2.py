import torch
import os
import sys
import gradio as gr

# Setup path
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

from train import init_tokenizer, load_model, stream_text
from nano_transformer_class import transformerConfig

# Initialize model and tokenizer
tokenizer = init_tokenizer(vocab_size=5000, pretrained=True, tokenizer_name="bpe_tok_5k", log=False)

config = transformerConfig(
    num_layers=4,
    num_heads=4,
    embedding_dim=512,
    feed_forward_dim=1024,
    max_seq_len=256,
    vocab_size=tokenizer.vocab_size,
    dropout=0.1
)

model = load_model(config, model_name="nano_transformer_bpe_5k")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Define the chat function
def chat_function(message, history, max_len):
    full_prompt = ""
    for user_msg, bot_reply in history:
        full_prompt += f"User: {user_msg}\nAssistant: {bot_reply}\n"
    full_prompt += f"User: {message}\nAssistant:"

    response = ""
    for token in stream_text(prompt=full_prompt, max_len=int(max_len), model=model, bpe_tokenizer=tokenizer, device=device):
        response += token
        yield response

# Create the Gradio UI
with gr.Blocks() as demo:
    #gr.Markdown("# 🤖 Nano LLM Chatbot")
    gr.Markdown(
    """
    <h1 style='text-align: center; font-size: 2.5em; margin-bottom: 0.5em;'>
        🤖 Nano LLM Chatbot
    </h1>
    """,
    elem_id="title"
    )
    chatbot = gr.Chatbot(height=400)

    with gr.Row():
        max_len_slider = gr.Slider(minimum=64, maximum=512, value=256, step=32, label="Max Tokens")

    interface = gr.ChatInterface(
        fn=chat_function,
        chatbot=chatbot,
        additional_inputs=[max_len_slider],
        clear_btn="🧹 Clear",
        retry_btn="🔁 Retry",
    )

demo.launch()
