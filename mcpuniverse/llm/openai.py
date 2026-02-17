"""
OpenAI LLMs
"""
# pylint: disable=broad-exception-caught
import os
import time
import logging
from dataclasses import dataclass
from typing import Dict, Union, Optional, Type, List
from openai import OpenAI
from openai import RateLimitError, APIError, APITimeoutError, InternalServerError
from dotenv import load_dotenv
from pydantic import BaseModel as PydanticBaseModel

from mcpuniverse.common.config import BaseConfig
from mcpuniverse.common.context import Context
from .base import BaseLLM

load_dotenv()

logging.getLogger("OpenAIModel").setLevel(logging.DEBUG)


@dataclass
class OpenAIConfig(BaseConfig):
    """
    Configuration for OpenAI language models.

    Attributes:
        model_name (str): The name of the OpenAI model to use (default: "gpt-4o").
        api_key (str): The OpenAI API key (default: environment variable OPENAI_API_KEY).
        temperature (float): Controls randomness in output (default: 1.0).
        top_p (float): Controls diversity of output (default: 1.0).
        frequency_penalty (float): Penalizes frequent token use (default: 0.0).
        presence_penalty (float): Penalizes repeated topics (default: 0.0).
        max_completion_tokens (int): Maximum number of tokens in the completion (default: 2048).
        reasoning_effort (str): The reasoning effort to use (default: "medium").
        seed (int): Random seed for reproducibility (default: 12345).
    """
    model_name: str = "gpt-4.1"
    api_key: str = os.getenv("OPENAI_API_KEY", "")
    temperature: float = 1.0
    top_p: float = 1.0
    frequency_penalty: float = 0.0
    presence_penalty: float = 0.0
    max_completion_tokens: int = 10000
    reasoning_effort: str = "medium"
    seed: int = 12345


class OpenAIModel(BaseLLM):
    """
    OpenAI language models.

    This class provides methods to interact with OpenAI's language models,
    including generating responses based on input messages.

    Attributes:
        config_class (Type[OpenAIConfig]): Configuration class for the model.
        alias (str): Alias for the model, used for identification.
    """
    config_class = OpenAIConfig
    alias = "openai"
    env_vars = ["OPENAI_API_KEY"]

    def __init__(self, config: Optional[Union[Dict, str]] = None):
        super().__init__()
        self.config = OpenAIModel.config_class.load(config)

    def _generate(
            self,
            messages: List[dict[str, str]],
            response_format: Type[PydanticBaseModel] = None,
            **kwargs
    ):
        """
        Generates content using the OpenAI model.

        Args:
            messages (List[dict[str, str]]): List of message dictionaries,
                each containing 'role' and 'content' keys.
            response_format (Type[PydanticBaseModel], optional): Pydantic model
                defining the structure of the desired output. If None, generates
                free-form text.
            **kwargs: Additional keyword arguments including:
                - max_retries (int): Maximum number of retry attempts (default: 720)
                - base_delay (float): Delay in seconds between retries, linear backoff (default: 60.0)
                - timeout (int): Request timeout in seconds (default: 600)

        Returns:
            Union[str, PydanticBaseModel, None]: Generated content as a string
                if no response_format is provided, a Pydantic model instance if
                response_format is provided, or None if parsing structured output fails.
                Returns None if all retry attempts fail or non-retryable errors occur.
        """
        max_retries = kwargs.get("max_retries", 480)
        base_delay = kwargs.get("base_delay", 60.0)
        env_timeout = os.getenv("OPENAI_API_TIMEOUT_SECONDS")
        if env_timeout is not None:
            timeout = int(env_timeout)
            self.logger.info("[OpenAI] Using OPENAI_API_TIMEOUT_SECONDS=%d from environment", timeout)
        else:
            # Set a long timeout for each request so that the model server has sufficient time to respond.
            timeout = int(kwargs.get("timeout", 900))
            self.logger.debug("[OpenAI] Using default timeout=%d", timeout)

        for attempt in range(max_retries + 1):
            try:
                client = OpenAI(api_key=self.config.api_key)
                # Models support the 'reasoning_effort' parameter.
                # This set can be extended as new models are introduced.
                _models_with_reasoning_effort_support = {"gpt-5", "o3", "o4-mini", "gpt-5-high"}
                if any(prefix in self.config.model_name
                       for prefix in _models_with_reasoning_effort_support):
                    kwargs["reasoning_effort"] = self.config.reasoning_effort

                if "high" in self.config.model_name:
                    kwargs["reasoning_effort"] = "high"
                    self.config.model_name = "gpt-5"

                self.logger.debug("[OpenAI Attempt %d/%d] Making API call to model %s", 
                                 attempt + 1, max_retries + 1, self.config.model_name)

                if response_format is None:
                    chat = client.chat.completions.create(
                        messages=messages,
                        model=self.config.model_name,
                        temperature=self.config.temperature,
                        timeout=timeout,
                        top_p=self.config.top_p,
                        frequency_penalty=self.config.frequency_penalty,
                        presence_penalty=self.config.presence_penalty,
                        max_completion_tokens=self.config.max_completion_tokens,
                        seed=self.config.seed,
                        **kwargs
                    )
                    # If tools are provided, return the entire response object
                    # so the caller can handle both content and tool_calls
                    if 'tools' in kwargs:
                        self.logger.debug("[OpenAI] Returning tool response object")
                        return chat
                    # For backward compatibility, return just content when no tools
                    content = chat.choices[0].message.content
                    self.logger.debug("[OpenAI Attempt %d/%d] Response content type: %s", 
                                     attempt + 1, max_retries + 1, type(content).__name__)
                    self.logger.debug("[OpenAI Attempt %d/%d] Response is None: %s", 
                                     attempt + 1, max_retries + 1, content is None)
                    if content is not None:
                        self.logger.debug("[OpenAI Attempt %d/%d] Response length: %d", 
                                         attempt + 1, max_retries + 1, len(content))
                        self.logger.debug("[OpenAI Attempt %d/%d] Response (first 300 chars): %s", 
                                         attempt + 1, max_retries + 1, repr(content[:300]))
                    else:
                        self.logger.error("[OpenAI Attempt %d/%d] API returned None content!", 
                                         attempt + 1, max_retries + 1)
                        self.logger.error("[OpenAI Attempt %d/%d] Full chat response: %s", 
                                         attempt + 1, max_retries + 1, chat)
                    return content

                chat = client.beta.chat.completions.parse(
                    messages=messages,
                    model=self.config.model_name,
                    temperature=self.config.temperature,
                    timeout=timeout,
                    top_p=self.config.top_p,
                    frequency_penalty=self.config.frequency_penalty,
                    presence_penalty=self.config.presence_penalty,
                    max_completion_tokens=self.config.max_completion_tokens,
                    seed=self.config.seed,
                    response_format=response_format,
                    **kwargs
                )
                # If tools are provided, return the entire response object
                # so the caller can handle both content and tool_calls
                if 'tools' in kwargs:
                    return chat
                # For backward compatibility, return just parsed content when no tools
                parsed = chat.choices[0].message.parsed
                self.logger.debug("[OpenAI Attempt %d/%d] Parsed response: %s", 
                                 attempt + 1, max_retries + 1, type(parsed).__name__)
                return parsed

            except (RateLimitError, APIError, APITimeoutError, InternalServerError, ConnectionError) as e:
                if attempt == max_retries:
                    # Last attempt failed, return None instead of raising
                    self.logger.error("[OpenAI] All %d attempts failed. Last error: %s", max_retries + 1, e)
                    self.logger.error("[OpenAI] Returning None - this will cause JSON decode errors downstream!")
                    return None

                # Linear backoff: constant delay between retries
                delay = base_delay
                self.logger.warning("[OpenAI Attempt %d/%d] Failed with error: %s. Retrying in %.1f seconds...",
                           attempt + 1, max_retries + 1, e, delay)
                time.sleep(delay)

            except Exception as e:
                # For non-retryable errors, return None instead of raising
                self.logger.error("[OpenAI] Non-retryable error occurred: %s", e)
                self.logger.error("[OpenAI] Returning None - this will cause JSON decode errors downstream!")
                return None

    def set_context(self, context: Context):
        """
        Set context, e.g., environment variables (API keys).
        """
        super().set_context(context)
        self.config.api_key = context.env.get("OPENAI_API_KEY", self.config.api_key)
