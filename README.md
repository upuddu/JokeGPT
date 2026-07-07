# JokeGPT

A fine-tuned language model that generates jokes, packaged for deployment with Cog/Replicate.

## Overview

JokeGPT is a LoRA (Low-Rank Adaptation) fine-tune of Google's **Gemma 2 2B** trained to
produce short jokes. The repository ships the trained adapter and a
[Cog](https://github.com/replicate/cog) configuration so the model can be built into a
container and served locally or on [Replicate](https://replicate.com).

## Contents

| File | Description |
|------|-------------|
| `adapter_model.safetensors` | The trained LoRA adapter weights. |
| `predict.py` | The Cog `Predictor`: loads the base model plus the adapter and runs generation. |
| `cog.yaml` | Cog build configuration (GPU image, Python version, and dependencies). |

## Model details

- **Base model:** Google Gemma 2 2B (26 transformer layers, hidden size 2304).
- **Adaptation:** LoRA (rank 8) applied to the attention projections
  (`q_proj`, `k_proj`, `v_proj`, `o_proj`) and the MLP projections
  (`gate_proj`, `up_proj`, `down_proj`), loaded at inference time with
  [PEFT](https://github.com/huggingface/peft).
- **Prompt format:** Gemma chat format. Prompts include the Gemma turn tokens
  (for example `<start_of_turn>`), and the predictor returns the generated text with the
  prompt stripped from the front.

The adapter is applied on top of the base Gemma 2 2B weights and tokenizer, which the
predictor loads from the model directory. Those base files (along with the adapter
configuration) must be present in the build context when the container is built.

## Inference

`predict.py` defines a Cog `Predictor` with two stages:

- `setup()` loads the tokenizer and the base causal-LM from the model directory, then
  attaches the LoRA adapter with `PeftModel.from_pretrained` and switches the model to
  evaluation mode.
- `predict(prompt)` tokenizes the prompt, generates up to 64 new tokens
  (`temperature=0.7`), decodes the result, and returns the continuation with the original
  prompt removed.

### Input

| Name | Type | Description |
|------|------|-------------|
| `prompt` | string | The full prompt, including any Gemma turn tokens. |

## Running with Cog

Install [Cog](https://github.com/replicate/cog) and, from the repository directory, build
and run a prediction (a GPU is required, per `cog.yaml`):

```bash
cog predict -i prompt="<start_of_turn>user
Tell me a joke.<end_of_turn>
<start_of_turn>model
"
```

## Deploying to Replicate

Push the model to a Replicate destination to serve it via the API:

```bash
cog login
cog push r8.im/<your-username>/jokegpt
```

## Dependencies

Declared in `cog.yaml`: `torch`, `transformers`, `peft`, and `lion-pytorch`, on Python 3.10
with GPU support enabled.
