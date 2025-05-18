import torch
import os
import sys
import gradio as gr

from tests.kevin_tests.train import init_tokenizer, load_model, stream_text
from nano_transformer_class import transformerConfig


configs_dict = {
    "milli_transformer": transformerConfig(
        num_layers=8,
        num_heads=8,
        embedding_dim=1024,
        feed_forward_dim=1024,
        max_seq_len=256,
        vocab_size=10000,
    ),
    "micro_transformer": transformerConfig(
        num_layers=8,
        num_heads=4,
        embedding_dim=512,
        feed_forward_dim=1024,
        max_seq_len=256,
        vocab_size=10000
    ),
    "nano_transformer": transformerConfig(
        num_layers=4,
        num_heads=8,
        embedding_dim=512,
        feed_forward_dim=1024,
        max_seq_len=256,
        vocab_size=7000,
    ),
    "pico_transformer": transformerConfig(
        num_layers=4,
        num_heads=4,
        embedding_dim=256,
        feed_forward_dim=512,
        max_seq_len=128,
        vocab_size=7000,
    ),
    "femto_transformer": transformerConfig(
        num_layers=2,
        num_heads=2,
        embedding_dim=256,
        feed_forward_dim=512,
        max_seq_len=64,
        vocab_size=1000,
    )
}


model_to_tokenizer = {
    "milli_transformer": "bpe_10k",
    "micro_transformer": "bpe_10k",
    "nano_transformer": "bpe_7k",
    "pico_transformer": "bpe_7k",
    "femto_transformer": "bpe_1k",
}

model_to_vocab_size = {
    "milli_transformer": 10000,
    "micro_transformer": 10000,
    "nano_transformer": 7000,
    "pico_transformer": 7000,
    "femto_transformer": 1000,
}

# Define the chat function
def chat_function(message, history, max_len, model_name):  

    tokenizer = init_tokenizer(vocab_size=model_to_vocab_size[model_name], pretrained=True, tokenizer_param_dir_path="./tokenizers", tokenizer_name=model_to_tokenizer[model_name], log=False)

    model = load_model(configs_dict[model_name], model_name=model_name)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)    

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
        model_dropdown = gr.Dropdown(
        choices=list(configs_dict.keys()),
        value="nano_transformer",            
        label="Select Model"
        )

        max_len_slider = gr.Slider(minimum=64, maximum=512, value=256, step=32, label="Max Tokens")

    interface = gr.ChatInterface(
        fn=chat_function,
        chatbot=chatbot,
        additional_inputs=[max_len_slider, model_dropdown],
        clear_btn="🧹 Clear",
        retry_btn="🔁 Retry",
    )

demo.launch()
