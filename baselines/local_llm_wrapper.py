"""Local LLM wrapper compatible with GraphRAG's ChatOpenAI interface."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, Any, List, Optional
from dataclasses import dataclass

# Add trident to path to import LLMInterface
TRIDENT_PATH = Path(__file__).parent.parent
sys.path.insert(0, str(TRIDENT_PATH))

from trident.llm_interface import LLMInterface, LLMOutput


@dataclass
class LLMResponse:
    """Response from LLM generation (compatible with ChatOpenAI)."""
    content: str
    usage: Dict[str, int]


class LocalLLMWrapper:
    """
    Wrapper around TRIDENT's LLMInterface to be compatible with GraphRAG's ChatOpenAI.

    This adapter allows GraphRAG and KET-RAG to use local LLMs instead of OpenAI API.
    """

    def __init__(
        self,
        model_name: str = "Qwen/Qwen2.5-7B-Instruct",
        device: str = "cuda:0",
        temperature: float = 0.0,
        max_tokens: int = 512,
        load_in_8bit: bool = False,
        load_in_4bit: bool = False,
        **kwargs
    ):
        """
        Initialize local LLM wrapper.

        Args:
            model_name: HuggingFace model name (default: Qwen2.5-7B-Instruct)
            device: Device to use (cuda:0, cuda:1, cpu)
            temperature: Sampling temperature
            max_tokens: Max tokens for generation
            load_in_8bit: Use 8-bit quantization
            load_in_4bit: Use 4-bit quantization
            **kwargs: Additional arguments (ignored for compatibility)
        """
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens

        # Initialize TRIDENT's LLMInterface
        self.llm = LLMInterface(
            model_name=model_name,
            device=device,
            temperature=temperature,
            max_new_tokens=max_tokens,
            load_in_8bit=load_in_8bit,
            load_in_4bit=load_in_4bit,
        )

    def generate(
        self,
        messages: List[Dict[str, str]],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> LLMResponse:
        """
        Generate response from messages (compatible with ChatOpenAI interface).

        Args:
            messages: List of message dicts with 'role' and 'content'
            temperature: Override temperature
            max_tokens: Override max_tokens
            **kwargs: Additional arguments (ignored for compatibility)

        Returns:
            LLMResponse with content and usage
        """
        # Convert messages to single prompt
        prompt = self._messages_to_prompt(messages)

        # Generate using local LLM
        output = self.llm.generate(
            prompt=prompt,
            temperature=temperature if temperature is not None else self.temperature,
            max_new_tokens=max_tokens if max_tokens is not None else self.max_tokens,
        )

        # Return in ChatOpenAI-compatible format
        return LLMResponse(
            content=output.text.strip(),
            usage={
                'prompt_tokens': output.tokens_used // 2,  # Approximate
                'completion_tokens': output.tokens_used // 2,
                'total_tokens': output.tokens_used,
            }
        )

    def _messages_to_prompt(self, messages: List[Dict[str, str]]) -> str:
        """
        Convert ChatOpenAI-style messages to a single prompt.

        Args:
            messages: List of message dicts with 'role' and 'content'

        Returns:
            Single prompt string
        """
        # Simple conversion - can be made more sophisticated with chat templates
        prompt_parts = []

        for msg in messages:
            role = msg.get('role', 'user')
            content = msg.get('content', '')

            if role == 'system':
                prompt_parts.append(f"System: {content}")
            elif role == 'user':
                prompt_parts.append(f"{content}")
            elif role == 'assistant':
                prompt_parts.append(f"Assistant: {content}")

        return "\n\n".join(prompt_parts)

    def get_model_info(self) -> Dict[str, Any]:
        """Get model information."""
        return {
            'model_name': self.model_name,
            'temperature': self.temperature,
            'max_tokens': self.max_tokens,
            'backend': 'local_llm',
        }


def create_local_llm(
    model_name: str = "Qwen/Qwen2.5-7B-Instruct",
    device: str = "cuda:0",
    temperature: float = 0.0,
    max_tokens: int = 512,
    load_in_8bit: bool = False,
    load_in_4bit: bool = False,
) -> LocalLLMWrapper:
    """
    Factory function to create a local LLM wrapper.

    Args:
        model_name: HuggingFace model name
        device: Device to use
        temperature: Sampling temperature
        max_tokens: Max tokens for generation
        load_in_8bit: Use 8-bit quantization
        load_in_4bit: Use 4-bit quantization

    Returns:
        LocalLLMWrapper instance
    """
    return LocalLLMWrapper(
        model_name=model_name,
        device=device,
        temperature=temperature,
        max_tokens=max_tokens,
        load_in_8bit=load_in_8bit,
        load_in_4bit=load_in_4bit,
    )
