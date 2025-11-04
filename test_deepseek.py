import os
import litellm

# Set provider & API
litellm.api_base = os.environ.get("CHUTES_API_BASE", "https://api.chutes.ai/v1")
litellm.api_key = os.environ.get("CHUTES_API_KEY")
litellm.provider = "chutes"

# Test model
response = litellm.completion(
    model="deepseek/deepseek-r1:free",
    messages=[{"role": "user", "content": "hello"}]
)

print(response)
