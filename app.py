import gradio as gr
import base64
import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def generate_content(image, prompt):
    # image is filepath
    with open(image, "rb") as f:
        img_base64 = base64.b64encode(f.read()).decode()

    response = client.chat.completions.create(
        model="gpt-4o-mini",   # vision model
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{img_base64}"
                        }
                    }
                ]
            }
        ],
        max_tokens=300
    )

    return response.choices[0].message.content


with gr.Blocks() as demo:
    gr.Markdown("## 🖼️ Image → Caption / Story Generator (OpenAI)")

    with gr.Row():
        image = gr.Image(
            type="filepath",
            label="Upload Image",
            scale=1
        )
        output = gr.Textbox(
            label="Generated Result",
            lines=15,        # ⬅️ makes it tall
            max_lines=25,
            scale=2          # ⬅️ makes it wider
        )

    prompt = gr.Textbox(
        value="Generate:\n1. Caption\n2. Short story\n3. Social media post\n4. Hashtags",
        label="Prompt"
    )

    btn = gr.Button("Generate")
    btn.click(generate_content, inputs=[image, prompt], outputs=output)

demo.launch()
