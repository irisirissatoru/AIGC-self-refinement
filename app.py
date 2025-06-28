import uuid
from PIL import Image
import requests
import openai
from diffusers import StableDiffusionPipeline
import torch
import streamlit as st
import io

# Initialize SD model (optional)
@st.cache_resource
def load_sd_model():
    return StableDiffusionPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5",
        torch_dtype=torch.float16
    ).to("cuda")

pipe = load_sd_model()

# Image scoring function: call your API
SCORE_API_URL = "http://localhost:8000/score"  # Replace with your public API if needed
def score_image_via_api(image: Image.Image) -> float:
    with io.BytesIO() as output:
        image.save(output, format="PNG")
        output.seek(0)
        files = {"file": ("image.png", output, "image/png")}
        try:
            response = requests.post(SCORE_API_URL, files=files)
            response.raise_for_status()
            return response.json().get("score", 0.0)
        except Exception as e:
            st.error(f"Scoring API request failed: {e}")
            return 0.0

# LLM feedback function
openai.api_key = st.secrets.get("OPENAI_API_KEY", "sk-xxx")
def get_feedback(prompt: str, score: float) -> str:
    system_msg = "You are an expert in prompt engineering and image generation. Help improve prompts for Stable Diffusion."
    user_msg = f"""
Prompt: {prompt}
Score: {score}/10

Please:
1. Briefly comment on the image quality;
2. Analyze any shortcomings of the prompt;
3. Suggest an improved version of the prompt.
"""
    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_msg}
        ]
    )
    return response["choices"][0]["message"]["content"]

# Streamlit UI
st.title("🧠 Prompt Feedback Assistant for Stable Diffusion")
prompt_input = st.text_area("Enter your text prompt below:")

if st.button("Generate and Evaluate") and prompt_input.strip():
    with st.spinner("Generating image..."):
        image = pipe(prompt_input).images[0]
        filename = f"output_{uuid.uuid4().hex[:8]}.png"
        image.save(filename)

    with st.spinner("Scoring image and getting feedback..."):
        score = score_image_via_api(image)
        feedback = get_feedback(prompt_input, score)

    st.image(image, caption=f"Image Score: {score}/10", use_column_width=True)
    st.markdown("### 📝 GPT Feedback and Prompt Suggestions")
    st.write(feedback)
else:
    st.info("Enter a prompt and click the button to start.")
