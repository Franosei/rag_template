import logging
import json
import time
import random
from typing import TypeVar, Type, Optional, Dict, Any, List
from openai import OpenAI, OpenAIError, RateLimitError, APITimeoutError
from pydantic import BaseModel, ValidationError
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
    before_sleep_log
)

from src.app.settings import settings

logger = logging.getLogger(__name__)

T = TypeVar('T', bound=BaseModel)


class LLMResponse(BaseModel):
    """Raw LLM response wrapper"""
    content: str
    model: str
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    finish_reason: str


class LLMError(Exception):
    """Base exception for LLM client errors"""
    pass


class LLMClient:
    """
    Wrapper around OpenAI client with:
    - JSON mode enforcement
    - Automatic retry with exponential backoff
    - Schema validation and repair
    - Token tracking
    """
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None
    ):
        self.client = OpenAI(api_key=api_key or settings.openai_api_key)
        self.model = model or settings.openai_model
        self.temperature = temperature or settings.openai_temperature
        self.max_tokens = max_tokens or settings.openai_max_tokens
        
        # Track usage
        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=0.5, min=0.5, max=10),
        retry=retry_if_exception_type((RateLimitError, APITimeoutError, OpenAIError)),
        before_sleep=before_sleep_log(logger, logging.WARNING),
        reraise=True
    )
    def _call_openai(
        self,
        messages: List[Dict[str, str]],
        response_format: Optional[Dict[str, str]] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None
    ) -> LLMResponse:
        """
        Low-level OpenAI call with transport-level retry.
        Handles rate limits, timeouts, transient errors.
        """
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=temperature or self.temperature,
                max_tokens=max_tokens or self.max_tokens,
                response_format=response_format or {"type": "text"}
            )
            
            # Track tokens
            usage = response.usage
            self.total_prompt_tokens += usage.prompt_tokens
            self.total_completion_tokens += usage.completion_tokens
            
            logger.debug(
                "LLM call succeeded",
                extra={
                    "model": self.model,
                    "prompt_tokens": usage.prompt_tokens,
                    "completion_tokens": usage.completion_tokens,
                    "finish_reason": response.choices[0].finish_reason
                }
            )
            
            return LLMResponse(
                content=response.choices[0].message.content,
                model=response.model,
                prompt_tokens=usage.prompt_tokens,
                completion_tokens=usage.completion_tokens,
                total_tokens=usage.total_tokens,
                finish_reason=response.choices[0].finish_reason
            )
            
        except (RateLimitError, APITimeoutError, OpenAIError) as e:
            logger.warning(f"LLM call failed (will retry): {e}")
            raise
        except Exception as e:
            logger.error(f"Unexpected LLM error: {e}")
            raise LLMError(f"LLM call failed: {e}") from e
    
    def generate_text(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: Optional[float] = None
    ) -> str:
        """Simple text generation (non-structured)"""
        messages = []
        
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        
        messages.append({"role": "user", "content": prompt})
        
        response = self._call_openai(messages, temperature=temperature)
        return response.content
    
    def generate_structured(
        self,
        prompt: str,
        response_model: Type[T],
        system_prompt: Optional[str] = None,
        temperature: Optional[float] = None,
        max_repair_attempts: int = 2
    ) -> T:
        """
        Generate structured output conforming to a Pydantic model.
        
        Implements semantic retry with repair loop:
        1. Ask for JSON matching schema
        2. If invalid JSON: repair prompt
        3. If schema mismatch: repair with validation errors
        4. Max repair_attempts, then fail
        """
        messages = []
        
        # Build system prompt with schema
        schema_json = json.dumps(response_model.model_json_schema(), indent=2)
        
        full_system_prompt = f"""{system_prompt or "You are a helpful assistant."}

You must respond with valid JSON that matches this schema exactly:

{schema_json}

CRITICAL: Respond ONLY with the JSON object. No markdown, no backticks, no preamble, no explanation.
"""
        
        messages.append({"role": "system", "content": full_system_prompt})
        messages.append({"role": "user", "content": prompt})
        
        for attempt in range(1 + max_repair_attempts):
            try:
                # Call OpenAI in JSON mode
                response = self._call_openai(
                    messages,
                    response_format={"type": "json_object"}
                )
                
                # Parse JSON
                try:
                    raw_json = json.loads(response.content)
                except json.JSONDecodeError as e:
                    if attempt < max_repair_attempts:
                        logger.warning(f"Invalid JSON on attempt {attempt + 1}, repairing...")
                        messages.append({"role": "assistant", "content": response.content})
                        messages.append({
                            "role": "user",
                            "content": f"That was invalid JSON. Error: {e}. Please return valid JSON matching the schema with no other text."
                        })
                        continue
                    else:
                        raise LLMError(f"Failed to generate valid JSON after {max_repair_attempts + 1} attempts") from e
                
                # Validate against Pydantic model
                try:
                    validated = response_model.model_validate(raw_json)
                    logger.info(
                        f"Successfully generated {response_model.__name__}",
                        extra={"attempt": attempt + 1}
                    )
                    return validated
                
                except ValidationError as e:
                    if attempt < max_repair_attempts:
                        logger.warning(f"Schema validation failed on attempt {attempt + 1}, repairing...")
                        
                        # Format validation errors for repair
                        error_details = "\n".join([
                            f"- Field '{err['loc'][0]}': {err['msg']}"
                            for err in e.errors()
                        ])
                        
                        messages.append({"role": "assistant", "content": response.content})
                        messages.append({
                            "role": "user",
                            "content": f"""That JSON didn't match the schema. Validation errors:

{error_details}

Please return corrected JSON matching the schema exactly."""
                        })
                        continue
                    else:
                        raise LLMError(
                            f"Failed to generate valid schema after {max_repair_attempts + 1} attempts. "
                            f"Errors: {e}"
                        ) from e
            
            except LLMError:
                # Don't retry LLMErrors (they're already retried at transport level)
                raise
            except Exception as e:
                logger.error(f"Unexpected error in generate_structured: {e}")
                raise LLMError(f"Structured generation failed: {e}") from e
        
        # Should never reach here due to loop logic, but satisfy type checker
        raise LLMError("Structured generation failed")
    
    def get_usage_stats(self) -> Dict[str, int]:
        """Return token usage stats"""
        return {
            "prompt_tokens": self.total_prompt_tokens,
            "completion_tokens": self.total_completion_tokens,
            "total_tokens": self.total_prompt_tokens + self.total_completion_tokens
        }