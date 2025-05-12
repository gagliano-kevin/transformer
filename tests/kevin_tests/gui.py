import torch
import os
import sys
import gradio as gr

# Set up system path
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

from train import init_tokenizer, load_model, stream_text
from nano_transformer_class import transformerConfig

# Load tokenizer and model
tokenizer = init_tokenizer(vocab_size=5000, pretrained=True, tokenizer_name="bpe_tok_1k", log=False)

config = transformerConfig(
    num_layers=4,
    num_heads=4,
    embedding_dim=512,
    feed_forward_dim=1024,
    max_seq_len=256,
    vocab_size=tokenizer.vocab_size,
    dropout=0.1
)

model = load_model(config, model_name="nano_transformer_bpe_1k")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Define the chat function
def chat_function(message, history):
    """
    message: str, latest user message
    history: list of tuples (user, bot)
    """
    # Rebuild full conversation context
    full_prompt = ""
    for user_msg, bot_reply in history:
        full_prompt += f"User: {user_msg}\nAssistant: {bot_reply}\n"
    full_prompt += f"User: {message}\nAssistant:"

    response = ""
    for token in stream_text(prompt=full_prompt, max_len=128, model=model, bpe_tokenizer=tokenizer, device=device):
        response += token
        yield response  # streaming output

# Create ChatInterface
chat = gr.ChatInterface(
    fn=chat_function,
    title="Nano LLM Chatbot",
    description="Chat with a lightweight transformer-based language model.",
    retry_btn="🔁 Retry",
    clear_btn="🧹 Clear",
    chatbot=gr.Chatbot(height=400),
)

if __name__ == "__main__":
    chat.launch()
