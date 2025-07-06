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
