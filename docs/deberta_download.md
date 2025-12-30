# Downloading DeBERTa models locally

If the mirror endpoint returns `429 Too Many Requests`, download the model directly from Hugging Face and cache it locally so repeated runs do not hit the API:

```bash
# Authenticate once if needed (ensures higher rate limits)
huggingface-cli login

# Download the model weights to the default cache (~/.cache/huggingface/hub)
huggingface-cli download microsoft/deberta-v2-xlarge-mnli --local-dir /tmp/deberta-v2-xlarge-mnli --local-dir-use-symlinks False
```

Then point the environment to the local copy (no further network calls):

```bash
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_HOME=/tmp/hf-cache
# Move or symlink the downloaded folder into $HF_HOME/hub if desired
```

Alternatively, clone with Git LFS if you prefer a specific location:

```bash
git lfs install
git clone https://huggingface.co/microsoft/deberta-v2-xlarge-mnli /tmp/deberta-v2-xlarge-mnli
```

Use that directory path in your config or by setting `HF_HOME` so the pipeline resolves the model without calling the mirror.
