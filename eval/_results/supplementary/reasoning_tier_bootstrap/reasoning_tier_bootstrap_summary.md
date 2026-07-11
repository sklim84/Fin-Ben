# Reasoning Capability-Tier Separation (paired bootstrap)

- Per-item score: mean of 6 reasoning criteria
- Models: 19 proprietary (Table 6)
- Items: 575
- Bootstrap iterations: 5000, seed 42

- Top tier: 6 models (mean 6.96)
- Middle tier: 7 models (mean 6.59)
- Bottom tier: 6 models (mean 5.88)

| Comparison | Δ | 95% CI |
|---|:---:|:---:|
| Top vs Middle | +0.36 | [+0.31, +0.42] |
| Top vs Bottom | +1.07 | [+1.01, +1.14] |
| Middle vs Bottom | +0.71 | [+0.64, +0.78] |

All pairwise tier differences are significant (95% CI excludes 0).

## Adjacent-pair significance

Of the 18 rank-adjacent proprietary model pairs, only **4** are significantly separated (95% CI excludes 0); the rest, mostly closely-ranked frontier models, are not forced apart.

## Per-model mean reasoning (ranked)

| Model | Mean |
|---|:---:|
| openai_gpt-5 | 7.27 |
| gpt-5.2_reasoning | 7.18 |
| gpt-5 | 6.86 |
| gpt-5_2 | 6.81 |
| gpt-5-mini_reasoning | 6.80 |
| gemini-3-flash-preview | 6.80 |
| gemini-2.5-pro | 6.79 |
| grok-4.1-fast-reasoning | 6.72 |
| claude-opus-4-5 | 6.63 |
| gemini-2.5-flash | 6.59 |
| grok-4-fast-reasoning | 6.58 |
| gpt-4.1 | 6.45 |
| claude-sonnet-4-5 | 6.42 |
| gemini-3.1-pro-preview_2602 | 6.38 |
| gpt-5-mini | 6.34 |
| mistralai_mistral-medium-3.1 | 6.02 |
| claude-haiku-4-5 | 5.97 |
| gpt-5-nano_reasoning | 5.64 |
| gpt-5-nano | 4.95 |
