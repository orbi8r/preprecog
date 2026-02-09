# Precog NLP Task
_Le style peut donc être enlevé_ (the style, therefore, can be removed)

**Video:** [Watch PPT Generated video for more context](https://iiithydresearch-my.sharepoint.com/:v:/g/personal/shuban_biswas_research_iiit_ac_in/IQBxgxTBdsvSTZDUNMMtvIGDAZbQA7CcjFX3fr4QkXLk6uk?nav=eyJyZWZlcnJhbEluZm8iOnsicmVmZXJyYWxBcHAiOiJPbmVEcml2ZUZvckJ1c2luZXNzIiwicmVmZXJyYWxBcHBQbGF0Zm9ybSI6IldlYiIsInJlZmVycmFsTW9kZSI6InZpZXciLCJyZWZlcnJhbFZpZXciOiJNeUZpbGVzTGlua0NvcHkifX0&e=1FL7U0)

## Structure

```
data/                       corpus creation (Task 0)
  data_human/               Project Gutenberg essays (Bacon, Emerson, James, Russell)
  data_ai/                  Gemini API generation (standard + imposter)
  data_merge.ipynb          merge into data.parquet, 1508 rows

data_analysis/              feature engineering (Task 1)
  LexicalRichness.ipynb     TTR, Hapax, MTLD, sentence length std
  SyntacticComplexity.ipynb FK grade, tree depth, punctuation density, PCA
  RowFeatureEnrichment.ipynb per-row 26 features -> data.parquet

models/                     detection models (Task 2) + interpretability (Task 3)
  Statistician/             XGBoost on 26 features
  Semanticist/              GloVe 300d + feedforward NN
  Transformer/              DistilBERT + LoRA fine-tuning (+ Saliency_Mapping.ipynb)
  Contrastive/              triplet loss on MiniLM embeddings
  Variance/                 sentence transition features
  Structure/                2D similarity heatmaps

imposter/                   adversarial attacks (Task 4-5)
  genetic/                  GA evolves AI text past XGBoost detector
  clauses_assemble/         Monte Carlo punctuation humanizer
```

## How to run

### Install deps
```
pip install pandas numpy matplotlib seaborn scikit-learn xgboost
pip install torch transformers peft
pip install sentence-transformers gensim captum
pip install spacy nltk
pip install google-genai python-dotenv
python -m spacy download en_core_web_sm
```

### Setup .env
For notebooks that use Gemini API (data_ai, genetic algorithm, clause assembler), create `.env` in project root:
```
GEMINI_API_KEY=your_api_key_here
```

### Run order

Task 0 (data):
1. `data/data_human/src/data_human.ipynb`
2. `data/data_ai/src/data_ai.ipynb` (needs Gemini API key in .env)
3. `data/data_merge.ipynb`

Task 1 (features):
4. `data_analysis/LexicalRichness.ipynb`
5. `data_analysis/SyntacticComplexity.ipynb`
6. `data_analysis/RowFeatureEnrichment.ipynb` -> writes `data_analysis/data.parquet`

Task 2 (models, any order, all read from data_analysis/data.parquet):
7. `models/Statistician/StatisticianModel_Clean.ipynb`
8. `models/Semanticist/Semanticist_Model.ipynb`
9. `models/Transformer/Transformer_Model.ipynb` (needs GPU)
10. `models/Contrastive/Contrastive_Model.ipynb`
11. `models/Variance/Variance_Model.ipynb`
12. `models/Structure/Structure_Analysis.ipynb`

Task 3 (interpretability):
13. `models/Transformer/Saliency_Mapping.ipynb` (needs trained transformer)

Task 4 (genetic algorithm):
14. `imposter/genetic/Genetic_Algorithm.ipynb` (needs Gemini API key)

Task 5 (clause assembler):
15. `imposter/clauses_assemble/Clause_Assembler.ipynb` (needs Gemini API key)

## Dataset

| Class | Label | Source | Rows |
|-------|-------|--------|------|
| 1 | Human | Project Gutenberg (Bacon, Emerson, James, Russell) | 500 |
| 2 | Standard AI | Gemini (topic only prompt) | 504 |
| 3 | Imposter AI | Gemini (mimics author style) | 504 |

100-200 word chunks, 6 topics: Ethics & Conduct, General Philosophy, Mind & Knowledge, Religion & Spirit, Society & Politics, Truth & Reality.


## Dependencies

pandas, numpy, matplotlib, seaborn, scikit-learn, xgboost, torch, transformers, peft, sentence-transformers, gensim, captum, spacy, nltk, google-genai, python-dotenv
