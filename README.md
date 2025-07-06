# Large-Bonk-Model
The Large Bonk Model (LBM) is a custom-built Large Language Model designed to power the LetsBonkFun ecosystem with intelligence, humor, and community-first interaction. Built in the spirit of BONK, LBM is not just smart—it’s fun, fast, and meme-native.
# 🧠 Large Bonk Model (LBM)

The official BONK-powered LLM for the LetsBonkFun ecosystem. Designed to create memes, moderate mayhem, and guide degens with wizardly wisdom.

## Features
- Meme & lore generation
- Community Q&A bot
- BONKonomics explainer
- Custom personas like "Wizard Bonk"

## Setup
```bash
git clone https://github.com/bonktoshi/large-bonk-model.git
cd large-bonk-model
pip install -r requirements.txt
uvicorn app.main:app --reload


### 📄 `requirements.txt`
```text
fastapi
uvicorn
openai
python-dotenv

__pycache__/
.env

from fastapi import FastAPI, Request
from app.bonk_agent import generate_bonk_response

app = FastAPI(title="Large Bonk Model (LBM)")

@app.post("/generate")
async def generate(request: Request):
    body = await request.json()
    prompt = body.get("prompt", "")
    persona = body.get("persona", "wizard_bonk")
    response = generate_bonk_response(prompt, persona)
    return {"response": response}
import openai
import os
import json
from dotenv import load_dotenv

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

with open("app/prompts/bonk_personas.json", "r") as f:
    PERSONAS = json.load(f)

def generate_bonk_response(prompt: str, persona: str = "wizard_bonk"):
    system_prompt = PERSONAS.get(persona, PERSONAS["wizard_bonk"])
    
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": prompt}
    ]

    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=messages,
        temperature=0.8
    )
    return response.choices[0].message["content"]
{
  "wizard_bonk": "You are Wizard Bonk, a quirky and magical meme-master who speaks like a chaotic Gandalf. Your job is to entertain, explain, and enchant the BONK community.",
  "shiba_sidekick": "You're a hyper-enthusiastic shiba inu sidekick who helps users navigate the BONK ecosystem. You speak in emojis and short hype bursts!"
}
OPENAI_API_KEY=your-key-here
