from typing import Optional

from .gemini import Gemini
from .openai import GPT
from .documentai import DocumentAI


def get_model_provider(name: Optional[str], output_dir: Optional[str] = None):
    """Return a provider instance for a given model name.

    Heuristics-based factory: matches model name patterns to available
    provider implementations. Returns None for special sentinel values
    like "ground-truth". If no provider is implemented for the given
    name, raises NotImplementedError.
    """
    if not name:
        return None

    key = name.lower()
    if key == "ground-truth":
        return None

    # Google / Gemini family
    if "gemini" in key or key.startswith("g-"):
        return Gemini(name, output_dir)
    
    if "google" in key:
        return DocumentAI(name, output_dir)

    # OpenAI-like models (gpt, gpt-4, etc.)
    if "openai" in key or key.startswith("gpt") or "gpt" in key:
        return GPT(name, output_dir)

    # No provider implemented for this model name
    raise NotImplementedError(f"No provider implemented for model '{name}'")


__all__ = ["get_model_provider", "Gemini", "GPT", "DocumentAI"]
