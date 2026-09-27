# OpenMed local NER and de-identification demo

Browser UI and a thin loopback API for clinical NER and de-identification.
Pasted text is handled by the public SDK (`analyze_text`, `extract_pii`,
`deidentify`) on this machine. This is a local-first demo, not a HIPAA
compliance claim.

The demo does not wrap `openmed.service`, does not bind `0.0.0.0`, and does
not send pasted text to a cloud model, webhook, or analytics service. Model
weights may be downloaded from the Hugging Face Hub into the local cache.
After the active models are cached, the API prefers Hub offline mode.

## What you run

| Piece | Role |
| --- | --- |
| `demo/api` | FastAPI process bound to loopback only |
| `demo/web` | Vite/React UI. Talks to the API through one `InferencePort` |

Paste routes:

- `POST /analyze` → `openmed.analyze_text` (`cache_results=False`)
- `POST /pii/extract` → `openmed.extract_pii` (`cache_results=False`)
- `POST /pii/deidentify` → `openmed.deidentify` (`cache_results=False`)

Catalog routes never accept clinical text. They list registry models, download
into the local cache, and switch the active NER and PII models on the same
loader (no process restart).

De-identification in the UI is `mask` or `replace`. The reversible mapping is
not returned.

## Configure

From the repository root:

```bash
cp demo/.env.example demo/.env
```

`demo/.env.example` names every setting the processes read: API and web
hosts and ports, the CORS origin, the UI's API origin, NER and PII model ids,
cache directories, Hub offline flags, and the backend (`auto`, `hf`, or
`mlx`). The API exits if any name is missing or if the bind host is not
loopback. `auto` uses MLX on Apple Silicon when the `mlx` extra imports, and
local Hugging Face weights otherwise. `remote` is rejected.

Keep `OPENMED_DEMO_CORS_ORIGIN` on the web origin and
`VITE_OPENMED_DEMO_API_ORIGIN` on the API origin. The checked-in example uses
`127.0.0.1` only. Registry aliases for the same weights collapse to one picker
row. With `OPENMED_DEMO_BACKEND=auto` on Apple Silicon, the API loads an MLX
sibling of the selected model when the `mlx` extra is installed.

## Install

Python 3.10 or newer. From the repository root:

```bash
python -m pip install -e ".[hf]"
python -m pip install -r demo/api/requirements.txt
```

On Apple Silicon, install the MLX extra as well so `auto` can use it:

```bash
python -m pip install -e ".[hf,mlx]"
```

The web UI needs Node.js 20 or newer:

```bash
cd demo/web
npm install
```

## Run

Use two terminals from the repository root. The API loads `demo/.env` itself.

```bash
python -m demo.api
```

```bash
cd demo/web
npm run dev
```

Open the origin printed by Vite (the example is `http://127.0.0.1:5173`).
The status line reads `Local · 127.0.0.1 · no cloud PHI` when the API origin
host is `127.0.0.1`.

Download the NER and PII models in the model panel before the first run if
they are not cached yet. The API keeps one `ModelLoader` for the process and
warms a pipeline when the weights are already on disk, so later notes do not
reload weights. Switching the active model in the panel does not restart the
API.

Paste only the built-in synthetic samples, or your own fake note. Do not paste
real patient information.

## Tests

```bash
python -m pytest tests/unit/demo -q
cd demo/web && npm test && npm run build
```
