# NL Terminal — Natural Language to Bash

A terminal assistant that converts plain English (and Malayalam/Manglish) into bash commands using a multi-stage AI pipeline.

## Demo

> "list all hidden files in current directory"
> → `ls -la`

> "find all python files recursively"
> → `find . -name "*.py"`

---

## How It Works

```
User Input → [Malayalam Translator] → RAG Retrieval → T5 Model → Ollama Correction → Safety Check → Execute
```

1. **Malayalam support** — detects Manglish/Malayalam input, transliterates and translates to English using AI4Bharat + Facebook NLLB
2. **RAG retrieval** — finds similar known commands using SBERT sentence embeddings and cosine similarity
3. **T5 model** — fine-tuned T5 model translates natural language to bash
4. **Ollama correction** — uses local llama3.2 to verify and correct T5 output, provides alternative commands
5. **Safety checker** — 3-tier system (SAFE / WARN / BLOCK) that catches destructive commands before execution
6. **Memory** — learns from successful commands and updates the vector index

---

## Tech Stack

- **T5** — fine-tuned seq2seq model for NL to bash translation
- **SBERT** (`all-MiniLM-L6-v2`) — sentence embeddings for RAG retrieval
- **Ollama** (`llama3.2`) — local LLM for command correction
- **AI4Bharat** — Manglish to Malayalam transliteration
- **Facebook NLLB** — Malayalam to English translation
- **PyTorch** — model inference
- **Rich** — terminal UI

---

## Setup

### Requirements
- Python 3.10+
- [Ollama](https://ollama.com) running locally with llama3.2

```bash
ollama pull llama3.2
ollama serve
```

### Install

```bash
git clone https://github.com/Pranav-22-codes/nl-terminal.git
cd nl-terminal
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Run

```bash
python3 main_v2.py
```

---

## Project Structure

```
nl-terminal/
├── main_v2.py       # Entry point and main loop
├── translator.py    # T5 model + RAG pipeline
├── corrector.py     # Ollama-based command correction
├── memory.py        # SBERT vector index and learning
├── executor.py      # Command execution and cd handling
├── safety.py        # 3-tier safety checker
├── native.py        # Malayalam/Manglish translator
└── LINUX_TERMINAL_COMMANDS_CLEANED.jsonl  # Training/RAG dataset
```

---

## Features

- Natural language to bash in plain English
- Malayalam and Manglish input support
- Suggests best command + one alternative
- Blocks destructive commands automatically
- Learns and remembers successful commands
- Fully local — no cloud API needed

---

## Environment Variables

| Variable | Default | Description |
|---|---|---|
| `OLLAMA_URL` | `http://localhost:11434` | Ollama server URL |
| `OLLAMA_MODEL` | `llama3.2` | Model to use for correction |

---

Built as a college project by [Pranav](https://github.com/Pranav-22-codes).
