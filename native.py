"""
Malayalam (Manglish) → English translator
Used as a preprocessing step in the NL terminal pipeline.
"""
import argparse
import torch
torch.serialization.add_safe_globals([argparse.Namespace])

from ai4bharat.transliteration import XlitEngine
from transformers import NllbTokenizer, AutoModelForSeq2SeqLM


class MalayalamTranslator:
    """
    Converts Manglish (romanized Malayalam) → English
    in one call, ready to drop into the terminal pipeline.
    """

    # Fallback keyword hints if lingua is unavailable
    _MANGLISH_HINTS = {
        "undakku", "undak", "cheyyuka", "cheyyu", "kanikku",
        "nokku", "upayogam", "venam", "venda", "edukku",
        "tirakkuka", "adukkuka", "maykuka", "peril", "ennoru",
        "evidanu", "evide", "evidey", "evidaanu", "parayu",
        "tharu", "kanaan", "aanu", "illa", "ente",
        "njan", "ningal", "enthanu", "thora", "ittu",
        "cheyyoo", "kaanaan", "poyi", "varu", "varoo",
        "nokkoo", "thaa", "edu", "edukku", "kodukkoo",
        "sahaayam", "veedu", "padam", "computer", "open",
        "cheyyuka", "close", "delete", "save", "send",
        "adipoli", "pinne", "eppol", "engane", "enthokke",
        "sheriyaa", "athe", "alle", "athu", "ithu",
        "folder", "file", "cheyyan", "kaanikkan",
    }

    def __init__(self):
        print("[Malayalam] Loading transliteration engine...")
        self.xlit = XlitEngine("ml", beam_width=5)

        print("[Malayalam] Loading NLLB translation model...")
        model_id        = "facebook/nllb-200-distilled-600M"
        self.tokenizer  = NllbTokenizer.from_pretrained(model_id)
        # Force CPU — SBERT already uses GPU, NLLB on GPU causes OOM on 4GB VRAM
        self.model      = AutoModelForSeq2SeqLM.from_pretrained(model_id).to("cpu")
        self.english_id = self.tokenizer.convert_tokens_to_ids("eng_Latn")
        print("[Malayalam] Ready.")

    def is_manglish(self, text: str) -> bool:
        """
        Detect Manglish/Malayalam using multiple strategies:
        1. Malayalam unicode script check
        2. lingua language detector (best for short Manglish text)
        3. Keyword fallback
        """
        # Strategy 1: actual Malayalam script characters
        if any('\u0d00' <= c <= '\u0d7f' for c in text):
            return True

        # Strategy 2: lingua detector (handles romanized Malayalam well)
        try:
            from lingua import Language, LanguageDetectorBuilder
            detector = LanguageDetectorBuilder.from_languages(
                Language.ENGLISH, Language.MALAYALAM
            ).build()
            result = detector.detect_language_of(text)
            if result == Language.MALAYALAM:
                return True
            # Also check confidence — mixed Manglish scores partial Malayalam
            confidences = detector.compute_language_confidence_values(text)
            for conf in confidences:
                if conf.language == Language.MALAYALAM and conf.value > 0.2:
                    return True
        except ImportError:
            pass  # lingua not installed, fall through to keyword check
        except Exception:
            pass

        # Strategy 3: keyword fallback
        words = set(text.lower().split())
        return bool(words & self._MANGLISH_HINTS)

    def translate(self, text: str) -> tuple[str, str]:
        """
        Translates Manglish/Malayalam to English.
        Returns (malayalam_script, english_text)
        """
        # Step 1: Manglish → Malayalam script
        mal_script = self.xlit.translit_sentence(text)['ml']

        # Step 2: Malayalam script → English
        self.tokenizer.src_lang = "mal_Mlym"
        inputs = self.tokenizer(mal_script, return_tensors="pt")
        inputs = {k: v.to("cpu") for k, v in inputs.items()}
        with torch.no_grad():
            tokens = self.model.generate(
                **inputs,
                forced_bos_token_id=self.english_id,
                max_length=100,
            )
        english = self.tokenizer.batch_decode(tokens, skip_special_tokens=True)[0]
        return mal_script, english