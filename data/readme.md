# Data Pipeline — Task 0: The Library of Babel

## Run Order

1. `data_human/src/data_human.ipynb` — Extract and chunk Project Gutenberg essays → `data_human/processed/human_class1.parquet`
2. `data_ai/src/data_ai.ipynb` — Generate Class 2 & 3 AI data via Gemini API → `data_ai/processed/ai_class2.parquet`, `ai_class3.parquet`
3. `data_merge.ipynb` — Concatenate all three classes → `data.parquet`

## Dataset Overview

| Class | Label | Source | Rows |
|-------|-------|--------|------|
| 1 | Human | Project Gutenberg Essays | 500 |
| 2 | Standard AI | Gemini (topic-only prompt) | 504 |
| 3 | Imposter AI | Gemini (author-mimicry prompt) | 504 |

**Human Authors:** Francis Bacon (1597), Ralph Waldo Emerson (1841), William James (1907), Bertrand Russell (1912)

**AI Models:** Gemini 3 Flash, Gemini 2.5 Flash, Gemini 2.5 Flash Lite

**Topics:** Ethics & Conduct, General Philosophy, Mind & Knowledge, Religion & Spirit, Society & Politics, Truth & Reality

## `data.parquet` Column Reference

| Column | Type | Description |
|--------|------|-------------|
| `id` | str | UUID per row |
| `class` | int | 1 = Human, 2 = Standard AI, 3 = Imposter AI |
| `topic` | str | One of six topic categories |
| `text` | str | 100–200 word text chunk |
| `feature_cache` | dict | Computed features (populated by `RowFeatureEnrichment.ipynb`) |

### `feature_cache` Keys (after enrichment)

**Lexical (5):** `n_tokens`, `n_types`, `ttr`, `hapax_5k`, `mtld`, `sent_len_std`

**Syntactic (4):** `adj_noun_ratio`, `tree_depth`, `fk_grade`, `discourse_density_per_100_words`

**Stylometric (1):** `function_word_pca` → `{dim1, dim2}`

**Punctuation (11):** `punct_semicolon`, `punct_colon`, `punct_exclamation`, `punct_question`, `punct_hyphen`, `punct_emdash`, `punct_asterisk`, `punct_apos`, `punct_apos_curly`, `punct_paren_open`, `punct_quote`