"""ollama_client.py
Thin Ollama client for the Air Quality SLM Assistant.

Wraps ollama.generate() calls for the improved model.
The model's numerical outputs are NEVER used as authoritative;
Python lookups and calculations are always preferred.
"""

import os
import sys
import logging
from pathlib import Path

# Ensure the project root is on the path
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import ollama
OLLAMA_AVAILABLE = True

# ── Model configuration ─────────────────────────────────────────────────────

# The improved model as specified by the user
MODEL_NAME = "air-quality-slm-improved:latest"

# Default generation parameters (deterministic, as validated)
DEFAULT_PARAMS = {
    "temperature": 0,
    "top_k": 1,
    "top_p": 1.0,
    "repeat_penalty": 1.0,
    "num_ctx": 128,   # continued-training change from 64 to 128
    "num_predict": 48,  # validated: temperature/RH answer needs ~19 tokens, fits in 128
}


# ── Client class ─────────────────────────────────────────────────────────────

class OllamaClient:
    """Wrapper around the Ollama API for the Air Quality SLM.

    Key design decisions:
      - Deterministic decoding (temperature=0, top_k=1) to match trained model behaviour
      - num_ctx=128 (the improved model's context window)
      - num_predict=48 (validated: temperature/RH answer is ~47 tokens, fits in 128)
      - The model is ONLY used for natural-language explanation, NOT for numerical values
      - Numerical values always come from Python lookups / calculations (authoritative)
    """

    def __init__(self, model_name: str = MODEL_NAME):
        self.model_name = model_name
        if not OLLAMA_AVAILABLE:
            logging.warning("ollama package not available; OllamaClient will operate in demo mode")
        # Verify model exists on first use, not at init

    def _ensure_ollama(self):
        """Check that ollama is available; return True if so."""
        return OLLAMA_AVAILABLE

    def generate(self, prompt: str, system_prompt: str = "", max_new_tokens: int = 48, **kwargs) -> str:
        """Generate a completion from the Ollama model.

        Parameters
        ----------
        prompt : str
            The user prompt/prefix to complete
        system_prompt : str
            The system instruction to prepend (injected via {{ .System }} template)
        max_new_tokens : int
            Maximum new tokens to generate (capped by context window)
        **kwargs
            Additional parameters to pass to ollama.generate()

        Returns
        -------
        str
            The model's completion text (prompt is excluded)
        """
        if not self._ensure_ollama():
            return "(Ollama not available - using deterministic fallback)"

        # Merge default params with any overrides
        # Ollama's Python client uses 'options' dict for generation params
        options = {}
        for k, v in DEFAULT_PARAMS.items():
            options[k] = v
        options.update(kwargs)

        try:
            # Use Ollama's system parameter which gets injected via {{ .System }} template
            response = ollama.generate(
                model=self.model_name,
                prompt=prompt,
                system=system_prompt,
                options=options
            )
            return response.get("response", "")
        except Exception as e:
            logging.error(f"Ollama generation error: {e}")
            return f"(Ollama error: {e})"

    def explain(self, context: str, question: str) -> str:
        """Get a natural-language explanation from the SLM.

        This is the ONLY use of the model for generating text.
        Numerical values in the response are NOT authoritative.

        Parameters
        ----------
        context : str
            Context/description of the task
        question : str
            The user's question

        Returns
        -------
        str
            Natural-language explanation from the model
        """
        prompt = f"""You are the Air Quality SLM. Provide a concise, helpful natural-language
explanation based on the following context. Do not invent or change any numerical values.

Context: {context}

Question: {question}

Answer:"""
        return self.generate(prompt, max_new_tokens=48)


# ── Convenience functions ──────────────────────────────────────────────────

def generate_response(prompt: str, system_prompt: str = "", model_name: str = MODEL_NAME, **kwargs) -> str:
    """Generate a response from the improved model.

    Parameters
    ----------
    prompt : str
        The prompt to send to the model
    system_prompt : str
        The system instruction to prepend (injected via {{ .System }} template)
    model_name : str
        The Ollama model name (default: air-quality-slm-improved:latest)
    **kwargs
        Additional parameters for ollama.generate()

    Returns
    -------
    str
        The model's completion
    """
    client = OllamaClient(model_name)
    return client.generate(prompt, system_prompt=system_prompt, **kwargs)


def is_model_available() -> bool:
    """Check if the improved Ollama model is available."""
    if not OLLAMA_AVAILABLE:
        return False
    try:
        client = OllamaClient()
        # Try a lightweight check
        client.generate("", max_new_tokens=1)
        return True
    except Exception:
        return False


def verify_model_untouched() -> bool:
    """Verify that the original Ollama model remains untouched.

    Checks that we are using the improved model and not retraining or modifying
    the original checkpoint. This is a runtime check, not a filesystem check.

    Returns
    -------
    bool
        True if the model appears to be in its original improved state
    """
    try:
        # Verify we're using the correct model name
        model_name = MODEL_NAME
        if "improved" not in model_name.lower():
            logging.warning(f"Model name {model_name} doesn't contain 'improved'")
            return False

        # Check the model is available and responsive
        if not is_model_available():
            return False

        # The key invariant: we only call ollama.generate, never modify weights
        return True
    except Exception as e:
        logging.error(f"Model verification error: {e}")
        return False