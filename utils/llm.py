import json
import re

from google import genai
from google.genai import types

from groq import Groq

from ollama import Client as OllamaClient

from utils.config import (
    GEMINI_API_KEY,
    GEMINI_DEFAULT_MODEL,
    GROQ_API_KEY,
    GROQ_DEFAULT_MODEL,
    OLLAMA_HOST,
    OLLAMA_DEFAULT_MODEL
)


# ==================================================
# 1. PROVIDER CONFIGURATION
# ==================================================

DEFAULT_MODELS = {
    "Gemini": GEMINI_DEFAULT_MODEL,
    "Groq": GROQ_DEFAULT_MODEL,
    "Ollama": OLLAMA_DEFAULT_MODEL
}


def get_configured_providers():

    providers = []

    if GEMINI_API_KEY:
        providers.append("Gemini")

    if GROQ_API_KEY:
        providers.append("Groq")

    if OLLAMA_DEFAULT_MODEL:
        providers.append("Ollama")

    return providers


# ==================================================
# 2. DYNAMIC MODEL DISCOVERY
# ==================================================

def list_available_models(provider):

    models = []

    try:

        # ------------------------------------------
        # GEMINI MODELS
        # ------------------------------------------

        if provider == "Gemini":

            if not GEMINI_API_KEY:
                raise ValueError("Gemini API key is missing.")

            client = genai.Client(
                api_key=GEMINI_API_KEY
            )

            for item in client.models.list():

                name = getattr(item, "name", None)

                if not name:
                    continue

                # Read supported methods where available.
                details = {}

                if hasattr(item, "model_dump"):
                    details = item.model_dump(by_alias=True)

                supported = (
                    details.get("supportedGenerationMethods")
                    or details.get("supported_generation_methods")
                    or getattr(item, "supported_actions", None)
                    or getattr(
                        item,
                        "supported_generation_methods",
                        None
                    )
                )

                if supported and not any(
                    "generateContent" in str(method)
                    for method in supported
                ):
                    continue

                model_id = name.removeprefix("models/")

                models.append(model_id)

        # ------------------------------------------
        # GROQ MODELS
        # ------------------------------------------

        elif provider == "Groq":

            if not GROQ_API_KEY:
                raise ValueError("Groq API key is missing.")

            client = Groq(
                api_key=GROQ_API_KEY
            )

            response = client.models.list()

            excluded_words = (
                "whisper",
                "audio",
                "speech",
                "tts",
                "transcription",
                "embedding",
                "guard"
            )

            for item in response.data:

                model_id = getattr(item, "id", None)

                if not model_id:
                    continue

                if getattr(item, "active", True) is False:
                    continue

                if any(
                    word in model_id.lower()
                    for word in excluded_words
                ):
                    continue

                models.append(model_id)

        # ------------------------------------------
        # OLLAMA LOCAL MODELS
        # ------------------------------------------

        elif provider == "Ollama":

            client = OllamaClient(
                host=OLLAMA_HOST
            )

            response = client.list()

            for item in response.models:

                model_name = (
                    getattr(item, "model", None)
                    or getattr(item, "name", None)
                )

                if model_name:
                    models.append(model_name)

        else:

            raise ValueError(
                f"Unknown provider: {provider}"
            )

    except Exception as error:

        print(
            f"Could not list {provider} models: {error}"
        )

    # Use the configured default as a fallback in the UI
    # when the model-list endpoint is temporarily unavailable.
    if not models:

        default_model = DEFAULT_MODELS.get(
            provider,
            ""
        )

        if default_model:
            models = [default_model]

    # Put the configured default at the top of the list.
    default_model = DEFAULT_MODELS.get(
        provider,
        ""
    )

    models = list(dict.fromkeys(models))

    if default_model and default_model in models:

        models.remove(default_model)

        models.insert(0, default_model)

    return models


# ==================================================
# 3. LLM CLIENT
# ==================================================

class LLMClient:

    def __init__(
        self,
        provider,
        model
    ):

        self.provider = provider

        self.model = model

        self.last_provider = None

        self.last_model = None

        self._clients = {}

    # ----------------------------------------------
    # GET PROVIDER CLIENT
    # ----------------------------------------------

    def _get_client(self, provider):

        if provider in self._clients:
            return self._clients[provider]

        if provider == "Gemini":

            if not GEMINI_API_KEY:
                raise ValueError("Gemini API key is missing.")

            client = genai.Client(
                api_key=GEMINI_API_KEY
            )

        elif provider == "Groq":

            if not GROQ_API_KEY:
                raise ValueError("Groq API key is missing.")

            client = Groq(
                api_key=GROQ_API_KEY
            )

        elif provider == "Ollama":

            if not OLLAMA_DEFAULT_MODEL:
                raise ValueError(
                    "Ollama default model is not configured."
                )

            client = OllamaClient(
                host=OLLAMA_HOST
            )

        else:

            raise ValueError(
                f"Unsupported provider: {provider}"
            )

        self._clients[provider] = client

        return client

    # ----------------------------------------------
    # GENERATE RESPONSE USING ONE PROVIDER
    # ----------------------------------------------

    def _generate_with_provider(
        self,
        provider,
        model,
        prompt,
        json_mode=False,
        temperature=0.2
    ):

        client = self._get_client(provider)

        # GEMINI
        if provider == "Gemini":

            config_options = {
                "temperature": temperature
            }

            if json_mode:

                config_options["response_mime_type"] = (
                    "application/json"
                )

            response = client.models.generate_content(
                model=model,
                contents=prompt,
                config=types.GenerateContentConfig(
                    **config_options
                )
            )

            if not response.text:
                raise ValueError(
                    "Gemini returned an empty response."
                )

            return response.text.strip()

        # GROQ
        elif provider == "Groq":

            arguments = {
                "model": model,
                "messages": [
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                "temperature": temperature
            }

            if json_mode:

                arguments["response_format"] = {
                    "type": "json_object"
                }

            response = client.chat.completions.create(
                **arguments
            )

            content = response.choices[0].message.content

            if not content:
                raise ValueError(
                    "Groq returned an empty response."
                )

            return content.strip()

        # OLLAMA
        elif provider == "Ollama":

            arguments = {
                "model": model,
                "messages": [
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                "options": {
                    "temperature": temperature
                },
                "stream": False
            }

            if json_mode:
                arguments["format"] = "json"

            response = client.chat(
                **arguments
            )

            content = response.message.content

            if not content:
                raise ValueError(
                    "Ollama returned an empty response."
                )

            return content.strip()

    # ----------------------------------------------
    # PROVIDER FALLBACK ORDER
    # ----------------------------------------------

    def _get_provider_sequence(self):

        providers = get_configured_providers()

        sequence = [self.provider]

        for provider in providers:

            if provider != self.provider:
                sequence.append(provider)

        return sequence

    # ----------------------------------------------
    # GENERATE WITH AUTOMATIC FALLBACK
    # ----------------------------------------------

    def _generate(
        self,
        prompt,
        json_mode=False,
        temperature=0.2
    ):

        errors = []

        sequence = self._get_provider_sequence()

        for provider in sequence:

            if provider == self.provider:

                model = self.model

            else:

                model = DEFAULT_MODELS.get(
                    provider,
                    ""
                )

            # Skip fallback providers without a default model.
            if not model:
                continue

            try:

                response = self._generate_with_provider(
                    provider=provider,
                    model=model,
                    prompt=prompt,
                    json_mode=json_mode,
                    temperature=temperature
                )

                if json_mode:

                    response = self._parse_json(
                        response
                    )

                self.last_provider = provider

                self.last_model = model

                return response

            except Exception as error:

                errors.append(
                    f"{provider} ({model}): {str(error)}"
                )

        raise RuntimeError(
            "All available LLM providers failed:\n"
            + "\n".join(errors)
        )

    # ----------------------------------------------
    # PARSE JSON
    # ----------------------------------------------

    @staticmethod
    def _parse_json(response):

        cleaned = response.strip()

        cleaned = re.sub(
            r"^```(?:json)?\s*",
            "",
            cleaned,
            flags=re.IGNORECASE
        )

        cleaned = re.sub(
            r"\s*```$",
            "",
            cleaned
        )

        try:

            result = json.loads(cleaned)

        except json.JSONDecodeError:

            start = cleaned.find("{")

            end = cleaned.rfind("}")

            if start == -1 or end == -1:

                raise ValueError(
                    "No valid JSON object found."
                )

            result = json.loads(
                cleaned[start:end + 1]
            )

        if not isinstance(result, dict):

            raise ValueError(
                "Expected a JSON object."
            )

        return result

    # ----------------------------------------------
    # PUBLIC METHODS
    # ----------------------------------------------

    def generate_text(
        self,
        prompt,
        temperature=0.4
    ):

        return self._generate(
            prompt,
            json_mode=False,
            temperature=temperature
        )

    def generate_json(
        self,
        prompt,
        temperature=0.1
    ):

        return self._generate(
            prompt,
            json_mode=True,
            temperature=temperature
        )