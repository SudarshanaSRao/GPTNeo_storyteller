# Mythology Storyteller Gradio App
This repository hosts a Gradio web application that uses the fine-tuned GPT-Neo Mythology Storyteller model (available at [Samurai719214/gptneo-mythology-storyteller](https://huggingface.co/spaces/Samurai719214/GPTNeo-storyteller)) to generate complete mythological narratives. Given an incomplete story excerpt, the model generates a complete narrative that includes the chapter (Parv), key event, section, and full story continuation.

## Features
- **Interactive Interface:** Built with Gradio, making it easy to use directly in your browser.
- **Complete Narrative Generation:** Automatically generates header details (Parv, Key Event, Section) along with the story continuation.
- **Customizable Parameters:** Generation parameters such as `max_new_tokens`, `temperature`, and `top_p` can be adjusted within the code if needed.
