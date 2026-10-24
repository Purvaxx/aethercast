"""Language-model provider for the explanation layer. Follows the project's existing convention (app/assistant.py):
the official Anthropic SDK, model id `claude-opus-5-5`, key from ANTHROPIC_API_KEY. The provider and model are explicit
and never switched silently. A live call happens ONLY in AnthropicProvider.complete(); everything else in the package
works without credentials. Errors are mapped to stable codes; keys, prompts and evidence are never logged.
"""
from __future__ import annotations

import os
from dataclasses import dataclass

PROVIDER = "anthropic"
MODEL = "claude-opus-5-5"
MAX_TOKENS = 4000
TIMEOUT_S = 60.0


class LLMError(Exception):
    def __init__(self, code: str, detail: str = ""):
        super().__init__(code)
        self.code = code            # LLM_UNAVAILABLE | LLM_AUTH | LLM_RATE_LIMIT | LLM_TIMEOUT | LLM_CONNECTION | LLM_REFUSAL | LLM_TRUNCATED | LLM_API_ERROR
        self.detail = detail        # exception class name only; never the message body, the prompt or a credential


def credentials_configured() -> bool:
    """True only when an explicit credential is present in the environment (profiles on disk are not probed)."""
    return bool(os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_AUTH_TOKEN"))


@dataclass
class AnthropicProvider:
    model: str = MODEL
    timeout: float = TIMEOUT_S
    max_retries: int = 1
    name: str = PROVIDER

    def complete(self, system: str, user: str) -> str:
        """One request; returns the text of the reply. REQUIRES A LIVE API KEY (never called by the test suite)."""
        if not credentials_configured():
            raise LLMError("LLM_UNAVAILABLE", "no credential in the environment")
        try:
            import anthropic
        except ImportError as ex:                              # the SDK is an optional dependency of this path
            raise LLMError("LLM_UNAVAILABLE", type(ex).__name__) from None
        try:
            client = anthropic.Anthropic(timeout=self.timeout, max_retries=self.max_retries)
            # Thinking is adaptive by default on this model; effort is set explicitly (the task is a short, grounded rewrite).
            resp = client.messages.create(model=self.model, max_tokens=MAX_TOKENS, system=system, output_config={"effort": "low"},
                                          messages=[{"role": "user", "content": user}])
        except anthropic.AuthenticationError as ex:
            raise LLMError("LLM_AUTH", type(ex).__name__) from None
        except anthropic.PermissionDeniedError as ex:
            raise LLMError("LLM_AUTH", type(ex).__name__) from None
        except anthropic.RateLimitError as ex:
            raise LLMError("LLM_RATE_LIMIT", type(ex).__name__) from None
        except anthropic.APITimeoutError as ex:
            raise LLMError("LLM_TIMEOUT", type(ex).__name__) from None
        except anthropic.APIConnectionError as ex:
            raise LLMError("LLM_CONNECTION", type(ex).__name__) from None
        except anthropic.APIStatusError as ex:
            raise LLMError("LLM_API_ERROR", f"{type(ex).__name__}:{getattr(ex, 'status_code', '')}") from None
        except Exception as ex:                                # noqa: BLE001  (any other SDK failure: mapped, not leaked)
            raise LLMError("LLM_API_ERROR", type(ex).__name__) from None
        if resp.stop_reason == "refusal":
            raise LLMError("LLM_REFUSAL", "refusal")
        if resp.stop_reason == "max_tokens":
            raise LLMError("LLM_TRUNCATED", "max_tokens")
        text = "".join(b.text for b in resp.content if getattr(b, "type", "") == "text")
        if not text.strip():
            raise LLMError("LLM_API_ERROR", "empty reply")
        return text


def default_provider():
    return AnthropicProvider() if credentials_configured() else None
