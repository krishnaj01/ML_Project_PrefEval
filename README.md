# ML_Project_PrefEval

**Improving Long-Context Preference Following in LLMs via a Trained Preference-Detection Classifier and Enhanced Retrieval**
Course project by **Krishna Jhanwar (12341250)**, built on *PrefEval* (Zhao et al., ICLR 2025).

This repository is the clean continuation of the project. **Phase 1 was carried out in the working fork
[github.com/krishnaj01/PrefEval](https://github.com/krishnaj01/PrefEval), branch `phase1-local-ollama`**, which also holds the full
development history, the mid-term report and slides, and the legacy files removed here. The git history is preserved in this
repository too.

| Phase | Goal | Status |
|---|---|---|
| **1** | Rebuild PrefEval's evaluation to run locally and reproduce its baselines (classification task) | ✅ complete |
| 2 | Train a small "does this turn state a preference?" classifier (DistilBERT / ModernBERT / MiniLM) | planned |
| 3 | Classifier → structured preference memory + modern retriever (bge / e5), evaluated against the Phase 1 baselines | planned |

---

## Contents
1. [Background: what PrefEval is](#1-background-what-prefeval-is)
2. [What the original repository contained](#2-what-the-original-repository-contained)
3. [Why it could not be used as-is, and what went wrong at first](#3-why-it-could-not-be-used-as-is-and-what-went-wrong-at-first)
4. [What I built in Phase 1, and how](#4-what-i-built-in-phase-1-and-how)
5. [Issues found and fixed](#5-issues-found-and-fixed)
6. [Experiments run](#6-experiments-run)
7. [Results](#7-results)
8. [Deviations from the paper](#8-deviations-from-the-paper)
9. [How to set up and run](#9-how-to-set-up-and-run)
10. [Repository layout](#10-repository-layout)
11. [What was removed compared with the original repository](#11-what-was-removed-compared-with-the-original-repository)
12. [Next steps (Phases 2 and 3)](#12-next-steps-phases-2-and-3)
13. [References, license and AI assistance](#13-references-license-and-ai-assistance)

---

## 1. Background: what PrefEval is

PrefEval ([paper](https://arxiv.org/abs/2502.09597), [original code](https://github.com/amazon-science/PrefEval)) measures whether
an LLM keeps following a user's preference as a conversation grows. It has 1,000 preference–query pairs over 20 everyday topics,
each in three forms:

- **Explicit:** *"I strictly avoid restaurants that serve foods containing gluten…"*
- **Implicit, choice-based:** the preference is revealed by which options the user accepts or rejects in a short dialogue.
- **Implicit, persona-driven:** the preference is mentioned in passing in a longer persona conversation.

The preference comes first. Then `inter_turns` unrelated conversation turns from LMSYS-Chat-1M are inserted, and then a query
that a generic answer would get wrong (*"I'll be visiting Rome soon. What restaurants would you recommend?"*). In the
**classification task** the model picks one of four options, and exactly one respects the preference. The paper compares five
baselines: **Zero-shot, Reminder, Chain-of-Thought (CoT), Self-Critic and RAG** (top-5 exchanges retrieved with SimCSE).

## 2. What the original repository contained

The upstream repository ([amazon-science/PrefEval](https://github.com/amazon-science/PrefEval), last upstream commit `5079505`):

| Folder / file | Purpose |
|---|---|
| `classification_task/` | Classification (MCQ) benchmark scripts and the five baselines |
| `generation_task/` | Generation benchmark, plus LLM-judge scoring of four error types with Claude 3 Sonnet |
| `utils/` | Prompt construction for Claude / Llama / Mistral / GPT / Gemini, plus three copies of `generate_message` (Bedrock, OpenAI, Gemini) |
| `SFT/` | Fine-tuning Mistral-7B and vLLM inference |
| `error_type/` | Judge prompts for the generation task |
| `example_scripts/` | Shell scripts that call the above with Bedrock model names |
| `benchmark_dataset/` | The dataset, LMSYS distractor turns, MCQ options, and **precomputed SimCSE retrieval scores** for RAG |
| `requirements.txt` | vLLM 0.6.3, torch 2.4, boto3, … |

All model calls go through **Amazon Bedrock** (Claude, Llama, Mistral) or the OpenAI / Gemini APIs.

## 3. Why it could not be used as-is, and what went wrong at first

- **No Bedrock access** for this project, and the compute is one desktop: **RTX 3060 (12 GB), i7-12700, 15 GB RAM**.
- A first attempt to patch the original scripts for local vLLM inference failed. The logs showed:

  | Error | Root cause |
  |---|---|
  | `cannot import name 'genai' from 'google'` | `utils/common_utils.py` imports the Gemini SDK at top level |
  | `name 'random' is not defined` | an upstream bug: `benchmark_classification.py` calls `random.seed` without importing `random` |
  | `Invalid model_type: local_vllm` | there are three duplicate `generate_message` functions, and the patch reached only some of them |
  | `UnboundLocalError: model_id` | the duplicate `get_model_info` functions had no entry for the new model |
  | Grid included the generation task and "turns" 2 / 10 / 20 | the generation task needs a Claude judge, and 2 / 10 / 20 don't match the paper's rows |

**Decision:** don't patch the original scripts. Reuse the original *prompt-building* code verbatim, and replace only the
*inference call* with a single local backend.

## 4. What I built in Phase 1, and how

```
configs/phase1/<tier>.yaml ─▶ phase1/runner.py ── for each (model, form, method, context length, topic) ──┐
                                                                                                          │
  benchmark_dataset/ ─▶ phase1/upstream/  (ORIGINAL prompt builders, unchanged) ─▶ raw prompt string      │
                                                                                                          ▼
                    phase1/backend.py ─ Ollama /api/generate (raw=true, temperature 0, seed 41, explicit num_ctx)
                                                                                                          │
              original extract_choice("<choice>B</choice>") ◀── model reply ◀────────────────────────────┘
                                     │
                                     ▼
   results/phase1/<model>/<form>/<method>/inter<N>/<topic>.jsonl   (1 line per question, fsync'd → resumable)
                                     │
                                     ▼
   phase1/aggregate.py ─▶ results/phase1/summary/  (vs_paper.md, summary.csv, per_topic.csv, plots)
```

- **Inference:** [Ollama](https://ollama.com) 0.34.0 (llama.cpp backend) instead of Bedrock or vLLM. No vLLM, torch or boto3 is needed.
  The original code builds raw chat-template strings, which is exactly what Bedrock received. They are sent in Ollama's **raw mode**, so the model sees the same text.
- **Models: two of the paper's own models**, so results compare directly with the paper's Figure 6:
  - `llama3:8b-instruct-q8_0` = Llama-3-8B-Instruct (Bedrock `meta.llama3-8b-instruct-v1:0`). 8k context, so 0.2k–3k only, as in the paper.
  - `mistral:7b-instruct-v0.2-q8_0` = Mistral-7B-Instruct-v0.2 (Bedrock `mistral.mistral-7b-instruct-v0:2`). 32k context, which covers up to 23k tokens (70 turns).
  - Mixtral-8x7B was ruled out: 28 GB at 4-bit is more than 12 GB of GPU memory plus about 10 GB of free RAM.
- **RAG** uses the authors' **precomputed SimCSE scores** in `benchmark_dataset/rag_retrieval/`, so no embedding model is needed.
- **Context lengths:** the paper's rows 0.2k / 1k / 3k / 10k / 16k / 23k correspond to `inter_turns` = 0 / 3 / 8 / 28 / 48 / 68.
  The paper's "N turns" equals `inter_turns` + 2. I verified this against the data and the measured prompt sizes.
- **Engineering for runs that last many hours:**
  - every answered question is appended to its result file and flushed immediately, so re-running the same command resumes and at most one question is lost
  - a run manifest records the config, git commit, Ollama version, model digest and server settings of every run
  - a `BACKEND_REV` tag prevents results from different prompt-normalisation versions being mixed
  - a **settings guard** refuses to start a run if the Ollama server settings differ from those the model's baselines were produced with
  - `scripts/phase1_tmux.sh` runs a tier inside tmux, so it survives terminal disconnects

## 5. Issues found and fixed

1. **Double start-of-sequence token.** The original prompts begin with a literal `<|begin_of_text|>` (Llama) or `<s>` (Mistral), and Ollama
   adds its own. Token counts confirmed the duplication (`"<s>hello"` = 4 tokens vs `"hello"` = 3). The literal is stripped.
2. **Empty self-critiques (Llama).** The original Llama self-critic templates are indented strings ending in
   `"<|end_header_id|>\n        "`. With that trailing whitespace Llama-3 ends its turn immediately, giving an empty critique and about 0% accuracy.
   Trailing spaces are now trimmed. The model still sometimes *generates* the indentation and stops, leaving 25% empty revisions at 1k for
   Llama explicit. This is documented, not hidden.
3. **Silent truncation.** Ollama drops the *start* of over-long prompts, which is exactly where the preference sits. The context size is now set
   explicitly, and every call checks the prompt length. **0 prompts were truncated** in all runs.
4. **Defects in the released data:**
   - `simcse_explicit_pref/entertain_games_overall300_topk_history.json` is truncated mid-file (4 of 51 questions parse), so explicit RAG uses 19 topics.
   - In 7 topics, the implicit retrieval files end a few questions early, so implicit RAG covers 973 of 1,000 questions. The entries that exist were checked and are correctly aligned.
5. **GPU memory at 32k context.** Mistral spilled onto the CPU, taking 11–695 s per question. It now uses flash attention plus an 8-bit KV cache
   (100% GPU, about 2,000 tokens/s). **All Mistral runs use this setting, and all Llama runs use Ollama's defaults.** The guard enforces this per model.

## 6. Experiments run

All cells cover **all 20 topics** (RAG exceptions above). RAG at 0.2k is undefined, and the paper has no value there either.

| Model | Explicit | Implicit-choice |
|---|---|---|
| Llama-3-8B | 5 methods × 0.2k / 1k / 3k | 5 methods × 0.2k / 1k / 3k |
| Mistral-7B | 5 methods × 0.2k / 1k / 3k, plus Zero-shot / Reminder / RAG at 10k / 16k / 23k | 5 methods × 0.2k / 1k / 3k |

**64,535 questions in 65 cells, about 71 GPU-hours.** Implicit-persona was not run: it is not required by the proposal, and the paper reports
implicit forms only for the generation task.

## 7. Results

Classification accuracy (%), macro-averaged over topics: **ours (paper Figure 6)**.

**Llama-3-8B-Instruct, explicit**

| Method | 0.2k | 1k | 3k |
|---|---|---|---|
| Zero-shot | 94.2 (85) | 52.7 (55) | 43.3 (42) |
| Reminder | 97.4 (94) | 78.0 (80) | 59.4 (59) |
| CoT | 93.7 (83) | 78.0 (67) | 63.9 (50) |
| Self-Critic | 78.8 (79) | 50.1 (73) | 46.7 (62) |
| RAG | – | 88.6 (87) | 74.8 (80) |

**Mistral-7B-Instruct-v0.2, explicit**

| Method | 0.2k | 1k | 3k | 10k | 16k | 23k |
|---|---|---|---|---|---|---|
| Zero-shot | 95.5 (87) | 61.5 (66) | 55.5 (57) | 42.5 (50) | 37.1 (42) | 37.6 (47) |
| Reminder | 96.2 (95) | 78.9 (85) | 71.3 (75) | 56.7 (63) | 43.9 (56) | 46.9 (56) |
| CoT | 88.0 (90) | 74.9 (79) | 67.0 (73) | – | – | – |
| Self-Critic | 79.8 (75) | 67.0 (65) | 61.0 (58) | – | – | – |
| RAG | – | 81.0 (87) | 75.7 (86) | 64.6 (81) | 59.0 (80) | 53.7 (75) |

**Implicit choice-based preferences (own baselines; the paper has no classification numbers for these)**

| Method | Llama 0.2k | Llama 1k | Llama 3k | Mistral 0.2k | Mistral 1k | Mistral 3k |
|---|---|---|---|---|---|---|
| Zero-shot | 77.0 | 53.3 | 46.0 | 81.0 | 50.8 | 47.5 |
| Reminder | 91.0 | 77.0 | 60.2 | 88.4 | 71.9 | 63.1 |
| CoT | 84.4 | 70.3 | 61.7 | 74.0 | 60.1 | 59.5 |
| Self-Critic | 66.1 | 50.1 | 38.7 | 67.2 | 51.8 | 50.9 |
| RAG | – | 82.8 | 73.7 | – | 69.0 | 64.0 |

**Key findings**
- **Reproduction:** excluding the trivial 0.2k row, 12 of 29 cells with a paper value are within 5 points and 20 of 29 within 10 (median
  |Δ| = 6.0). The paper's qualitative findings reproduce: accuracy collapses with context, Reminder and RAG are the strongest
  baselines, and implicit preferences are harder.
- **Known gaps:**
  - Llama Self-Critic is low because of the template issue (#2); excluding empty revisions gives 78.8 / 67.1 / 47.8.
  - CoT is +11 to +14 points for Llama but −2 to −6 for Mistral, which points to serving differences rather than a pipeline bias.
  - Mistral RAG falls increasingly below the paper at long context (−21 at 16k / 23k). The likely cause is the 8-bit KV cache; this is not yet verified.
- **Self-critique overturns correct answers:** at 0.2k, Llama's revisions turned 205 right answers wrong and fixed only 51 (Mistral: 188 vs 29).
- **Answer-position prior at long context:** for Mistral zero-shot at 23k, accuracy is 42% when the correct option is A but 23% when it is D. The model
  increasingly picks B and avoids D.
- **RAG can trigger refusals:** 19.1% of Llama's RAG answers at 3k contain no choice ("I cannot recommend…"), concentrated in food and travel topics.
- **Topic difficulty varies a lot.** A 3-topic pilot gave about 23% at 23k; all 20 topics give 37.6%. Always report results over all topics.

All numbers are generated from `results/phase1/` by `python -m phase1.aggregate`, and the full tables are in `results/phase1/summary/vs_paper.md`.

## 8. Deviations from the paper

- 8-bit (Q8_0) weights on llama.cpp instead of full precision on Bedrock. Mistral also uses an 8-bit KV cache.
- Two prompt normalisations (single start token, trailing-whitespace trim). The prompt *text* is identical to the original.
- Options are shuffled with a per-question seed (`seed:topic:task_id`) instead of one global random stream. They are still random, but identical
  across methods and lengths, and safe to resume.
- RAG coverage: 19 topics for explicit, 973 / 1,000 questions for implicit, because of data defects.
- One deterministic run per cell. The binomial standard error is at most about 1.6 points per cell.
- Classification task only. The generation task needs an LLM judge (Claude 3 Sonnet in the paper).

## 9. How to set up and run

```bash
# environment
conda create -y -n prefeval python=3.11 && conda activate prefeval
pip install -r requirements-phase1.txt

# models (Ollama must be running: systemctl status ollama)
ollama pull llama3:8b-instruct-q8_0
ollama pull mistral:7b-instruct-v0.2-q8_0
```

**Ollama server settings per model** (the runner refuses to start on a mismatch and prints the switch command):
```bash
# Mistral-7B: flash attention + 8-bit KV cache
sudo mkdir -p /etc/systemd/system/ollama.service.d
printf '[Service]\nEnvironment="OLLAMA_FLASH_ATTENTION=1"\nEnvironment="OLLAMA_KV_CACHE_TYPE=q8_0"\n' | sudo tee /etc/systemd/system/ollama.service.d/override.conf
sudo systemctl daemon-reload && sudo systemctl restart ollama
# Llama-3-8B: Ollama defaults
sudo rm /etc/systemd/system/ollama.service.d/override.conf && sudo systemctl daemon-reload && sudo systemctl restart ollama
```

**Running**
```bash
python -m phase1.runner show-prompt --model llama3-8b --method remind --inter-turns 3    # inspect a prompt (no model call)
python -m phase1.runner run --config configs/phase1/smoke.yaml --dry-run                # list cells and progress
scripts/phase1_tmux.sh tierA                                                            # run a tier in tmux (resumable)
python -m phase1.runner status --config configs/phase1/tierA.yaml                       # progress, accuracy, ETA
python -m phase1.aggregate                                                              # tables + plots in results/phase1/summary/
```
Re-running the same `run` command resumes where it stopped. The results in this repository are complete, so `status` shows 100% for every tier.
`scripts/phase1_tmux.sh` finds conda automatically (anaconda / miniconda / miniforge). Override the environment or session name with
`CONDA_ENV=myenv SESSION=myrun scripts/phase1_tmux.sh tierA`.

**Configs (`configs/phase1/`)**

| Config | Model | Contents | Measured time |
|---|---|---|---|
| `smoke.yaml`, `mistral_smoke.yaml` | both | every method, 5 questions per cell | minutes |
| `tierA.yaml` / `tierB.yaml` (+ `_pilot`) | Llama | explicit; Zero-shot / Reminder / RAG; CoT / Self-Critic | 3.3 h / 5.7 h |
| `mistral_tierA.yaml` / `mistral_tierB.yaml` | Mistral | same | 3.6 h / 5.0 h |
| `mistral_tierC_long(_pilot).yaml` | Mistral | explicit, 10k / 16k / 23k, Zero-shot / Reminder / RAG | 36 h |
| `llama_implicitA/B.yaml`, `mistral_implicitA/B.yaml` | both | implicit-choice; A = Zero-shot / Reminder / RAG, B = CoT / Self-Critic | 3–7 h each |
| `models.yaml` | – | Ollama tags, context sizes, required server settings | – |

## 10. Repository layout

```
phase1/                     Phase 1 pipeline
  runner.py                 run / status / show-prompt CLI; resume, manifest, settings guard
  backend.py                Ollama client (raw mode), prompt normalisation, BACKEND_REV
  aggregate.py              per-topic and averaged accuracy, comparison with paper, plots
  paper_baselines.json      paper Figure 6 numbers (all 6 models), keyed by inter_turns
  upstream/                 verbatim copies of the original prompt code (commit 5079505; only imports edited)
configs/phase1/             experiment tiers + models.yaml
scripts/phase1_tmux.sh      start a tier inside tmux
results/phase1/             all Phase 1 results (JSONL per cell), run_manifest.jsonl, summary/
benchmark_dataset/          PrefEval data, LMSYS distractor turns, MCQ options, precomputed SimCSE retrieval
docs/PHASE1.md              detailed Phase 1 notes (plan, flows, commands, run history)
config.yaml                 original system prompt / token limits (read by the runner)
CLAUDE.md, phase1/CLAUDE.md working notes for AI-assisted development
LICENSE                     original PrefEval license (CC BY-NC 4.0)
```

## 11. What was removed compared with the original repository

Removed deliberately. All of it remains available in the [working fork](https://github.com/krishnaj01/PrefEval) and in this repository's git history.

| Removed | Why |
|---|---|
| `classification_task/`, `generation_task/`, `utils/`, `SFT/`, `example_scripts/`, `error_type/` | Bedrock / OpenAI / vLLM code paths. The prompt builders actually needed are vendored unchanged in `phase1/upstream/` |
| `requirements.txt` | vLLM / torch / boto3 stack, replaced by `requirements-phase1.txt` |
| `benchmark_dataset/claude_generate_preferences_questions.py`, `gen_data_prompt.txt`, `rag_retrieval/count.py`, `mcq_options/travel_hotel copy.json` | Bedrock dataset-generation script, a helper with the authors' private paths, a duplicate file |

## 12. Next steps (Phases 2 and 3)

- **Phase 2:** a preference-detection classifier. Positives: explicit preference sentences and the preference-revealing implicit turns. Negatives:
  LMSYS turns and the other dialogue turns. **Use a topic-level train/test split, so Phase 3 is evaluated on topics the classifier never saw.**
  Report per-form precision / recall, the confusion matrix, and latency vs. an LLM-prompted detector.
- **Phase 3:** the classifier writes preferences to a memory list. At query time, memory + top-k from bge / e5 + query go into the prompt. It will be evaluated with this
  same runner, prompts, parser and settings guard against the Phase 1 baselines (both models, both forms, all context lengths).
- Optional: CoT / Self-Critic for Mistral at 10k–23k; a full-precision KV-cache check for the Mistral RAG gap; implicit-persona.

## 13. References, license and AI assistance

- S. Zhao, M. Hong, Y. Liu, D. Hazarika, K. Lin. *Do LLMs Recognize Your Preferences? Evaluating Personalized Preference Following
  in LLMs.* ICLR 2025. [arXiv:2502.09597](https://arxiv.org/abs/2502.09597). Code: [amazon-science/PrefEval](https://github.com/amazon-science/PrefEval).
- L. Zheng et al. *LMSYS-Chat-1M.* ICLR 2024. T. Gao et al. *SimCSE.* EMNLP 2021.
- Phase 1 working repository: [github.com/krishnaj01/PrefEval](https://github.com/krishnaj01/PrefEval) (branch `phase1-local-ollama`).

The dataset and the vendored upstream code are under the original PrefEval license (see `LICENSE`, CC BY-NC 4.0).

*AI assistance:* Claude (Anthropic) was used as a coding and writing assistant for the local pipeline, debugging and documentation.
All experiments were run locally. Every reported number is produced by the scripts in this repository from the logged results.
