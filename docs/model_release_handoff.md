# Model release handoff

This document is for the teammate responsible for training the next-token
language model. As of the 400k release, **no trained checkpoint exists** for
this project's model, so nothing was uploaded to a model repository. This
file explains exactly what is needed and how to publish it.

## 1. Current status (measured, not assumed)

- No training code exists in `ct219-final-report` (this repo is preprocessing
  only, by design).
- No next-token model checkpoint exists locally: `~/projects` was searched
  for `*.safetensors`, `pytorch_model.bin`, `config.json`, and checkpoint
  directories. The only checkpoints found belong to the W07 *assignment*
  (NER classification models: `DistilBertForTokenClassification` and a ViMed
  NER model). Those are token-classification models for a different
  assignment and cannot serve as the next-token generation model.
- `CT219_MODEL_DIR` is not set.
- `transformers` is not installed in the preprocessing environment.

## 2. Who provides the checkpoint

The model teammate (responsible for "model architecture, model fine-tuning,
tokenizer training" per the project split) must provide a trained
checkpoint. Expected location conventions, in order:

1. `CT219_MODEL_DIR` environment variable pointing at the checkpoint
   directory, or
2. a directory under the group repo, or
3. a Hugging Face model repository URL.

## 3. Required files for a reloadable checkpoint

The validation utility (`scripts/hf_release.py model-validate`) checks for:

| Group | Files |
| --- | --- |
| config | `config.json` |
| weights | `model.safetensors` (preferred), or `pytorch_model.bin` / `tf_model.h5` |
| tokenizer | `tokenizer.json`, or `vocab.txt` + `tokenizer_config.json` + `special_tokens_map.json` |
| optional | `generation_config.json` |

Additional requirements for a causal/next-token LM (verify before upload):

- architecture is causal (e.g., GPT-style decoder), not encoder-only;
- tokenizer loads and vocabulary size matches the model config;
- special-token IDs exist (bos/eos/pad as required);
- one short generation smoke test succeeds on CPU;
- no truncated weight files (all `*.safetensors` load);
- no secrets or absolute local paths in `config.json` or tokenizer files.

## 4. Exact commands to run after training

```bash
# 1. Structural validation (no upload)
python scripts/hf_release.py model-validate --model-dir "$CT219_MODEL_DIR"

# 2. Upload (private by default; resumable uploader)
python scripts/hf_release.py model --model-dir "$CT219_MODEL_DIR"

# Optional: choose a different repo id
HF_MODEL_REPO_ID=wheevu/ct219-vietnamese-next-token-model \
  python scripts/hf_release.py model --model-dir "$CT219_MODEL_DIR"
```

The upload utility refuses to run when validation fails, refuses to upload
empty weight files, and scans config/tokenizer files for secrets and local
paths. It never creates a repository full of preprocessing code and calls it
a model.

## 5. Model card contents

The model card must state (unknown values marked as unknown, never invented):

- architecture and parameter count;
- next-token-generation task, Vietnamese language;
- tokenizer reference;
- dataset repository (`wheevu/ct219-vietnamese-raw-400k`, revision
  `b81fcce58945970117a1b56d50ec81be2628a5c3` - see the dataset card);
- preprocessing code revision (this repo);
- training configuration, hardware, duration, evaluation results;
- limitations, licence status, citation and authorship.

## 6. Token requirement

Creating and uploading to a new repository requires a Hugging Face token
with **write** scope. The token used for the 400k dataset work
(`nlp-project`) has role `read` and cannot create repositories. Use a
write-scoped token via the standard `huggingface-cli login` / `hf auth
login` flow; never paste tokens into commands, files, or manifests.
