# Models - Task 2: The Detectors

6 independent models, each testing a different hypothesis about human vs AI writing. All load from `data_analysis/data.parquet` (1508 rows, 26 features). Stratified 70/15/15 splits by class x author.

| Model | Notebook | Approach | Test AUC |
|-------|----------|----------|---------|
| Statistician | `Statistician/StatisticianModel_Clean.ipynb` | XGBoost on 26 features | 0.9791 |
| Semanticist | `Semanticist/Semanticist_Model.ipynb` | GloVe 300d + feedforward NN | 0.9989 |
| Transformer | `Transformer/Transformer_Model.ipynb` | DistilBERT + LoRA | 0.9983 |
| Contrastive | `Contrastive/Contrastive_Model.ipynb` | Triplet loss MiniLM-L6-v2 | 0.9886 |
| Variance | `Variance/Variance_Model.ipynb` | Sentence transition features | weak |
| Structure | `Structure/Structure_Analysis.ipynb` | 2D similarity heatmaps | 0.81 |

## Task 3: Interpretability
`Transformer/Saliency_Mapping.ipynb` - Captum IntegratedGradients on the LoRA model. Token attribution reveals AI relies on punctuation microstructure + specific trigger words ("realm", "perennial", "ultimately") rather than abstract rhythm.

`Transformer/train_transformer_gpu.py` - standalone GPU training script, equivalent to the notebook but for remote server use.

## Model details

**Statistician**: 1000 estimators, lr 0.05, max depth 6, early stop 50. Top features: semicolon, PCA, MTLD, sent_len_std, asterisk.

**Semanticist**: GloVe 300d avg embedding -> 4 layer NN [512,256,128,64] with ReLU, Dropout 0.3, BatchNorm. 50 epochs Adam. Context-free embeddings still nearly perfect bc LLM vocabulary is that predictable.

**Transformer**: DistilBERT + LoRA (r=16, alpha=32, dropout=0.1) on q_lin/v_lin. 10 epochs, batch 4, grad accum 4, fp16, lr 2e-4. 887K trainable params out of 66M.

**Contrastive**: MiniLM-L6-v2 384d -> projector to 128d L2-normalized -> classifier head. Combined triplet loss (margin 0.5) + BCE. Underperforms transformer bc sentence-level embeddings lose sub-sentence patterns.

**Variance**: Tests if AI has smoother sentence transitions (it does, but signal is weak at 100-200 word level). 13 features: cosine sims, gradients, burst analysis, boundary overlap. GradientBoosting.

**Structure**: NxN sentence similarity matrices as fingerprints. 11 features from the matrices. Human writing is modular with sharp pivots, AI is uniformly coherent. RandomForest. Weak on short fragments.

## Deps
xgboost, scikit-learn, torch, transformers, peft, sentence-transformers, captum, gensim, spacy, nltk
