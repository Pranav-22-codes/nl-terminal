import os
import logging
import re
import json
import urllib.request
import urllib.error


OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434")
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "llama3.2")


def _call_ollama(prompt: str) -> list[tuple[str, str]]:
    """
    Returns a list of (command, description) tuples.
    Ollama is asked to return exactly 2 lines in the format:
        command|||short description
    """
    payload = json.dumps({
        "model": OLLAMA_MODEL,
        "stream": False,
        "options": {"temperature": 0},
        "messages": [
            {
                "role": "system",
                "content": (
                    "You are a Linux bash expert. "
                    "Reply with EXACTLY 2 lines, no markdown, no backticks, no numbering:\n"
                    "best_command|||short description (max 6 words)\n"
                    "alternative_command|||short description (max 6 words)\n\n"
                    "Example:\n"
                    "ls -la|||list all files with details\n"
                    "find . -maxdepth 1|||list files using find"
                ),
            },
            {"role": "user", "content": prompt},
        ],
    }).encode()

    req = urllib.request.Request(
        f"{OLLAMA_URL}/api/chat",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=30) as resp:
        body = json.loads(resp.read())

    raw = body["message"]["content"].strip()
    results = []
    for line in raw.splitlines():
        line = line.strip()
        if not line:
            continue
        if "|||" in line:
            cmd, desc = line.split("|||", 1)
            cmd = _clean(cmd.strip())
            if cmd:
                results.append((cmd, desc.strip()))
        if len(results) == 2:
            break
    return results


def _build_prompt(user_input: str, t5_candidate: str, rag_matches: list) -> str:
    rag_block = ""
    if rag_matches:
        lines = "\n".join(
            f'  "{m["prompt"]}" → {m["cmd"]}' for m in rag_matches[:3]
        )
        rag_block = f"\nSimilar examples:\n{lines}\n"

    return (
        f'User instruction: "{user_input}"\n'
        f"Local model suggested: {t5_candidate!r}\n"
        f"{rag_block}\n"
        f"Give the best bash command for this instruction, and one alternative approach.\n"
        f"Reply with EXACTLY 2 lines in the format: command|||short description"
    )


def _clean(raw: str) -> str:
    raw = re.sub(r"```[a-z]*\n?", "", raw).strip("`").strip()
    # strip leading list markers like "1. " or "- "
    raw = re.sub(r"^[\d]+\.\s*|^[-*]\s*", "", raw)
    for line in raw.splitlines():
        line = line.strip()
        if line:
            return line
    return raw.strip()


def _check_ollama_available() -> bool:
    try:
        req = urllib.request.Request(f"{OLLAMA_URL}/api/tags", method="GET")
        with urllib.request.urlopen(req, timeout=5) as resp:
            return resp.status == 200
    except Exception:
        return False


class CommandCorrector:
    """
    Uses a local Ollama instance (llama3.2 by default) to fix T5 hallucinations
    and generate a meaningful alternative command.

    Optional env vars:
        OLLAMA_URL   — default: http://localhost:11434
        OLLAMA_MODEL — default: llama3.2
    """

    def __init__(self, api_key: str | None = None):
        self.enabled = _check_ollama_available()
        if not self.enabled:
            print(f"[Corrector] Ollama not reachable at {OLLAMA_URL} — running without correction.")  # keep — startup warning
            print("[Corrector] Make sure Ollama is running: ollama serve")  # keep — startup warning
        else:
            print(f"[Corrector] Ollama connected — using model '{OLLAMA_MODEL}'")  # keep — startup confirmation

    def correct(
        self,
        user_input:   str,
        t5_candidate: str,
        rag_matches:  list | None = None,
    ) -> tuple[list[tuple[str, str]], bool]:
        """
        Returns (candidates, was_corrected) where candidates is a list of
        (command, description) tuples — best first, alternative second.
        """
        if not self.enabled:
            return [(t5_candidate, "")], False
        try:
            prompt = _build_prompt(user_input, t5_candidate, rag_matches or [])
            pairs = _call_ollama(prompt)

            if not pairs:
                return [(t5_candidate, "")], False

            best_cmd = pairs[0][0]
            was_corrected = best_cmd.strip() != t5_candidate.strip()
            if was_corrected:
                logging.debug(f"[Corrector] {t5_candidate!r}  →  {best_cmd!r}")
            return pairs, was_corrected

        except urllib.error.URLError as e:
            logging.warning(f"[Corrector] Connection error: {e}")
            return [(t5_candidate, "")], False
        except Exception as e:
            logging.warning(f"[Corrector] Error: {e}")
            return [(t5_candidate, "")], False