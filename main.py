import json
import logging
from collections.abc import Iterator
from dataclasses import dataclass
from functools import lru_cache
from typing import Any

import requests
from ulauncher.api.client.EventListener import EventListener
from ulauncher.api.client.Extension import Extension
from ulauncher.api.shared.action.CopyToClipboardAction import CopyToClipboardAction
from ulauncher.api.shared.action.DoNothingAction import DoNothingAction
from ulauncher.api.shared.action.HideWindowAction import HideWindowAction
from ulauncher.api.shared.action.RenderResultListAction import RenderResultListAction
from ulauncher.api.shared.event import ItemEnterEvent, KeywordQueryEvent
from ulauncher.api.shared.item.ExtensionResultItem import ExtensionResultItem

logger = logging.getLogger(__name__)
EXTENSION_ICON = "images/icon.png"
CHUNK_ACCUMULATION_SIZE = 50
MAX_NAME_LENGTH = 50
MAX_DESCRIPTION_LENGTH = 200


class ChatGPTError(Exception):
    """Base exception for ChatGPT extension errors."""

    pass


@dataclass(frozen=True)
class ChatGPTConfig:
    """Configuration for ChatGPT API calls."""

    api_key: str
    api_endpoint: str
    max_tokens: int
    frequency_penalty: float
    presence_penalty: float
    temperature: float
    top_p: float
    system_prompt: str
    line_wrap: int
    model: str

    @classmethod
    def from_preferences(cls, preferences: dict[str, Any]) -> "ChatGPTConfig":
        """Create config from extension preferences."""
        try:
            return cls(
                api_key=str(preferences["api_key"]),
                api_endpoint=str(
                    preferences.get(
                        "api_endpoint", "https://api.openai.com/v1/chat/completions"
                    )
                ),
                max_tokens=int(preferences.get("max_tokens", 150)),
                frequency_penalty=float(preferences.get("frequency_penalty", 0.0)),
                presence_penalty=float(preferences.get("presence_penalty", 0.0)),
                temperature=float(preferences.get("temperature", 0.7)),
                top_p=float(preferences.get("top_p", 1.0)),
                system_prompt=str(preferences.get("system_prompt", "")),
                line_wrap=int(preferences.get("line_wrap", 64)),
                model=str(preferences.get("model", "gpt-4o")),
            )
        except (KeyError, ValueError) as e:
            raise ChatGPTError(f"Invalid preferences: {str(e)}") from e

    @property
    def api_headers(self) -> dict[str, str]:
        return {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }

    @property
    def base_payload(self) -> dict:
        return {
            "model": self.model,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "top_p": self.top_p,
            "frequency_penalty": self.frequency_penalty,
            "presence_penalty": self.presence_penalty,
            "stream": True,
        }


def sanitize_text(text: str) -> str:
    """Clean and format command output for display."""
    # Preserve escaped newlines and other special characters
    text = text.strip("`").strip()
    # Only remove unescaped newlines
    text = " ".join(line.strip() for line in text.splitlines())
    return text


@lru_cache(maxsize=100)
def wrap_text(text: str, max_width: int) -> str:
    """Wrap text to specified width, with caching for performance."""
    if not text:
        return text

    words = text.split()
    lines = []
    current_line = []
    current_length = 0

    for word in words:
        word_length = len(word)
        if current_length + word_length + 1 <= max_width:
            current_line.append(word)
            current_length += word_length + 1
        else:
            if current_line:
                lines.append(" ".join(current_line))
            current_line = [word]
            current_length = word_length + 1

    if current_line:
        lines.append(" ".join(current_line))

    return "\n".join(lines)


class KeywordQueryEventListener(EventListener):
    def _create_result_item(
        self, text: str, config: ChatGPTConfig
    ) -> ExtensionResultItem:
        """Create a result item with clean, formatted text."""
        clean_text = sanitize_text(text)

        # For display, we'll show a preview
        display_text = clean_text[:MAX_DESCRIPTION_LENGTH]
        if len(clean_text) > MAX_DESCRIPTION_LENGTH:
            display_text += "..."

        return ExtensionResultItem(
            icon=EXTENSION_ICON,
            name=(
                display_text[:MAX_NAME_LENGTH] + "..."
                if len(display_text) > MAX_NAME_LENGTH
                else display_text
            ),
            description=display_text,
            on_enter=CopyToClipboardAction(clean_text),  # Full text for clipboard
        )

    def _stream_api_request(self, config: ChatGPTConfig, prompt: str) -> Iterator[str]:
        payload = {
            **config.base_payload,
            "messages": [
                {"role": "system", "content": config.system_prompt},
                {"role": "user", "content": prompt},
            ],
        }

        try:
            with requests.post(
                config.api_endpoint,
                headers=config.api_headers,
                json=payload,
                stream=True,
                timeout=10,
            ) as response:
                response.raise_for_status()

                for line in response.iter_lines():
                    if not line or line.strip() == b"data: [DONE]":
                        continue
                    if line.startswith(b"data: "):
                        try:
                            data = json.loads(line[6:])
                            if (
                                content := data["choices"][0]
                                .get("delta", {})
                                .get("content")
                            ):
                                yield content
                        except (json.JSONDecodeError, KeyError):
                            continue

        except requests.RequestException as e:
            raise ChatGPTError(f"API request failed: {str(e)}")

    def on_event(
        self, event: KeywordQueryEvent, extension: Extension
    ) -> RenderResultListAction:
        query = event.get_argument()
        if not query:
            return RenderResultListAction(
                [
                    ExtensionResultItem(
                        icon=EXTENSION_ICON,
                        name="Type in a prompt...",
                        description="Enter your question or prompt",
                        on_enter=DoNothingAction(),
                    )
                ]
            )

        try:
            config = ChatGPTConfig.from_preferences(extension.preferences)
            accumulated_text = []

            for chunk in self._stream_api_request(config, query):
                accumulated_text.append(chunk)
                if len(accumulated_text) >= CHUNK_ACCUMULATION_SIZE:
                    break

            text = "".join(accumulated_text)
            if not text:
                return RenderResultListAction(
                    [
                        ExtensionResultItem(
                            icon=EXTENSION_ICON,
                            name="No response received",
                            description="Try again or check your API key",
                            on_enter=DoNothingAction(),
                        )
                    ]
                )

            return RenderResultListAction([self._create_result_item(text, config)])

        except ChatGPTError as e:
            return RenderResultListAction(
                [
                    ExtensionResultItem(
                        icon=EXTENSION_ICON,
                        name=f"Error: {str(e)}",
                        on_enter=HideWindowAction(),
                    )
                ]
            )


class ItemEnterEventListener(EventListener):
    """Listener for handling item enter events."""

    def on_event(
        self, event: ItemEnterEvent, extension: Extension
    ) -> RenderResultListAction:
        """Handle item enter event."""
        data = event.get_data()
        return RenderResultListAction(
            [
                ExtensionResultItem(
                    icon=EXTENSION_ICON,
                    name="Text copied to clipboard",
                    description=(
                        data[:MAX_DESCRIPTION_LENGTH] + "..."
                        if len(data) > MAX_DESCRIPTION_LENGTH
                        else data
                    ),
                    on_enter=HideWindowAction(),
                )
            ]
        )


class GPTExtension(Extension):
    def __init__(self):
        super().__init__()
        self.subscribe(KeywordQueryEvent, KeywordQueryEventListener())
        self.subscribe(ItemEnterEvent, ItemEnterEventListener())


if __name__ == "__main__":
    GPTExtension().run()
