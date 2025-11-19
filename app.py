import cache_manager
import gradio as gr
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# Load model + tokenizer
model_name = "Samurai719214/gptneo-mythology-storyteller"
device = "cuda" if torch.cuda.is_available() else "cpu"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name).to(device)

# Story generation with history
def generate_full_story_chunks(excerpt, history_state):
    if not excerpt or not excerpt.strip():
        history_state.append(("❌", "⚠️ Enter a story excerpt."))
        yield history_state, gr.update(visible=False), gr.update(interactive=True)
        return

    inputs = tokenizer(excerpt, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}

    output_ids = model.generate(
        inputs["input_ids"],
        attention_mask=inputs["attention_mask"],
        max_new_tokens=400,
        do_sample=True,
        temperature=0.1,
        top_p=0.9,
        no_repeat_ngram_size=2,
    )

    generated_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)

    # Append user input
    history_state.append(("You", excerpt))

    # Stream response in chunks
    response = ""
    for i in range(0, len(generated_text), 200):
        response += generated_text[i:i+200]
        if len(history_state) > 0 and history_state[-1][0] == "AI":
            history_state[-1] = ("AI", response)
        else:
            history_state.append(("AI", response))
        yield history_state, gr.update(visible=False), gr.update(interactive=True)

# Clear conversation
def clear_history():
    return [], gr.update(interactive=False)

# Enable/disable generate button
def toggle_generate_button(text):
    return gr.update(interactive=bool(text.strip()))

# Build UI
with gr.Blocks() as demo:
    gr.Markdown("## 🏺 Mythology Storyteller")
    gr.Markdown("Enter a phrase from a chapter of your choice (include Parv, key event, and section for better results).")

    with gr.Row():
        with gr.Column():
            user_input = gr.Textbox(
                label="Incomplete story excerpt",
                placeholder="Enter an excerpt from the Mahabharata here...",
                lines=4,
            )
            summary_input = gr.Textbox(
                label="Chapter summary (optional)",
                placeholder="Enter summary if available...",
                lines=2,
            )
            generate_btn = gr.Button("✨ Generate Story", interactive=False)

        with gr.Column():
            output_text = gr.Chatbot(
                label="Generated Story",
                height=400,
                placeholder="⚔️ Legends are being written..." 
            )
            spinner = gr.Markdown("", visible=False)  # spinner placeholder
            clear_btn = gr.Button("🗑️ Clear Conversation", interactive=False)

    gr.Markdown("---")
    gr.Markdown(
    """🔌 View the Privacy Policy & Terms- 
    <a href="https://github.com/SudarshanaSRao/GPTNeo_storyteller/blob/main/privacy_terms.md" target="_blank">here</a>""",
    unsafe_allow_html=True
)
    gr.Markdown("# Please read the Privacy Policy and Terms of this app before using the app. If you disagree with it, do not use this app. If you proceed to use this app, then your agreement to the Privacy Policy & Terms will be assumed.")

    # Toggle generate button when input changes
    user_input.change(
        fn=toggle_generate_button,
        inputs=user_input,
        outputs=generate_btn,
    )

    # Show spinner when generating
    def show_spinner():
        return gr.update(value="⏳ Generating story...", visible=True)

    def hide_spinner():
        return gr.update(visible=False)

    generate_btn.click(
        fn=show_spinner,
        inputs=None,
        outputs=spinner,
    ).then(
        fn=generate_full_story_chunks,
        inputs=[user_input, output_text],
        outputs=[output_text, spinner, clear_btn],
    ).then(
        fn=hide_spinner,
        inputs=None,
        outputs=spinner,
    )

    # Clear history
    clear_btn.click(
        fn=clear_history,
        inputs=None,
        outputs=[output_text, clear_btn],
    )

# Launch app
if __name__ == "__main__":
    demo.launch()