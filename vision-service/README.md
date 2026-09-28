# Small Vision AI (Ollama + Railway)

An isolated vision service for legitimate image captioning, OCR, document
analysis, and visual question answering. It does not modify or import any of
the repository's existing application code.

The public gateway is authenticated, fixes the model server-side, validates
image sizes, and refuses CAPTCHA/security-challenge prompts. Raw Ollama is only
bound to `127.0.0.1` inside the container.

## Railway settings

**Root Directory:** `/vision-service`

Railway will discover this directory's `Dockerfile` and `railway.toml`.

### Required variables

```text
OLLAMA_MODEL=ahmadwaqar/smolvlm2-256m-video:q8_0
VISION_API_KEY=<generate-a-long-random-secret>
```

`MODEL` is accepted as an alias for `OLLAMA_MODEL`. Prefer `OLLAMA_MODEL`; if
you set both, make their values identical. Do not set `PORT` because Railway
provides it automatically.

`VISION_API_KEY` must be identical in every client and in this gateway. For two
services in one Railway project, define it once under **Project Settings →
Shared Variables**, then attach/reference that shared value from both services:

```text
VISION_API_KEY=${{ shared.VISION_API_KEY }}
```

Do not create two service-local values with the same name: they can drift and
produce HTTP 401 even though both services report the variable as configured.
Saving the shared reference redeploys the affected services. Never paste the
secret into application logs, source code, or support messages. A probe that
returns HTTP 401 means the endpoint is running and the keys do not match; it is
not a cold-start or reachability failure.

### Optional variables

```text
VISION_API_BASE=https://your-service.up.railway.app
OLLAMA_REQUEST_TIMEOUT=45
```

`OLLAMA_REQUEST_TIMEOUT` is the gateway→Ollama wait (default 45s). The
bot asks SmolVLM2 one tile at a time with a 12s client timeout; a 180s
gateway wait is how the old 9-image JSON path 504'd.

Set `VISION_API_BASE` to the public URL of this service. When provided, the
gateway returns it in the `base_url` field of its `GET /` health-check
response and includes it in the startup log. Railway's own
`RAILWAY_PUBLIC_DOMAIN` variable is also recognised if `VISION_API_BASE` is
not set.

### Recommended volume

Attach a Railway volume mounted at:

```text
/data
```

The model cache is stored in `/data/ollama`, so the model does not need to be
downloaded again on every deployment. The selected quantized model is well
below the requested 1.3 GB download-size ceiling, but inference uses additional
RAM. Allocate at least 2 GB RAM; download size and runtime memory are not the
same measurement.

The first deployment waits while Ollama downloads the model. Subsequent starts
are faster when the `/data` volume is attached.

## API

### Ollama-compatible (for the bot's vision_solver)

The bot speaks Ollama's native API; the gateway proxies it to local Ollama
with the model forced to `OLLAMA_MODEL` (fixed-model server-side). Both
endpoints require the API key:

```bash
# health / model list (the bot probes this before solving)
curl https://YOUR-SERVICE.up.railway.app/api/tags \
  -H "Authorization: Bearer $VISION_API_KEY"

# chat (native Ollama payload: messages[], base64 images).
# The bot's SmolVLM2 path does NOT send format:"json" — it hangs 256M models.
curl -X POST https://YOUR-SERVICE.up.railway.app/api/chat \
  -H "Authorization: Bearer $VISION_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"model":"ignored-forced-server-side","messages":[{"role":"user","content":"What is this?","images":["<base64>"]}],"stream":false}'
```

Point the bot at this service with `VISION_API_BASE` + `VISION_API_KEY`.
The bot's configured `OLLAMA_MODEL` name does not need to match — the
gateway forces the local model.

### Analyze an image (legacy safe endpoint)

Note: `POST /v1/analyze` refuses CAPTCHA / security-challenge prompts
(its contract is lawful captioning / OCR / visual Q&A). Use the
Ollama-compatible `/api/chat` above for the bot.

```bash
IMAGE=$(base64 < photo.jpg | tr -d '\n')

curl -X POST https://YOUR-SERVICE.up.railway.app/v1/analyze \
  -H "Authorization: Bearer $VISION_API_KEY" \
  -H "Content-Type: application/json" \
  -d "{\"prompt\":\"Describe this image precisely.\",\"image\":\"$IMAGE\"}"
```

The `image` value can also be a `data:image/jpeg;base64,...` URI. For multiple
images, send an `images` array instead.

Example response:

```json
{
  "model": "ahmadwaqar/smolvlm2-256m-video:q8_0",
  "response": "The image shows ...",
  "done": true
}
```

## Model choice and realistic limits

`SmolVLM2-256M-Video-Instruct Q8_0` is selected for Railway Trial limits.
Its Ollama package is about 279 MB including the vision projector, so it fits a
0.5 GB Trial volume. It is suitable for lightweight captioning, OCR, and visual
Q&A, but it is less capable than the 500M variant, will not recognize
everything, and cannot safely "do everything." Smaller models trade accuracy
for low size and cost.

Ollama performs inference; it does not train the model. A later lawful training
workflow would be:

1. Gather a licensed, task-specific image/text dataset.
2. Split it into training, validation, and held-out test sets.
3. Fine-tune the original Hugging Face checkpoint outside Ollama.
4. Evaluate accuracy and failure cases before deployment.
5. Convert the resulting checkpoint to GGUF and point `OLLAMA_MODEL` to it.

Use a database or retrieval system for persistent knowledge rather than trying
to make the model memorize every fact through fine-tuning.
