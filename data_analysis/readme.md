# Data Analysis - Task 1: The Fingerprint

## Run order
1. `LexicalRichness.ipynb` - TTR, Hapax, MTLD, sentence length std (group level)
2. `SyntacticComplexity.ipynb` - adj/noun ratio, tree depth, FK grade, discourse density, punctuation heatmap, function word PCA
3. `RowFeatureEnrichment.ipynb` - computes all 26 features per row, writes `data.parquet`

## Features

| # | Feature | Key |
|---|---------|-----|
| 1 | Type-Token Ratio | ttr |
| 2 | Hapax Legomena | hapax_5k |
| 3 | MTLD | mtld |
| 4 | Sentence Length Std Dev | sent_len_std |
| 5 | Adj/Noun Ratio | adj_noun_ratio |
| 6 | Dependency Tree Depth | tree_depth |
| 7 | Flesch-Kincaid Grade | fk_grade |
| 8 | Discourse Connective Density | discourse_density_per_100_words |
| 9 | Punctuation Density | punct_* (11 sub-features) |
| 10 | PCA Function Words | function_word_pca dim1, dim2 |

## Output
`data_analysis/data.parquet` - enriched dataset, all features in each rows feature_cache dict.
