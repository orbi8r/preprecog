# Imposter - Task 4 & 5: Adversarial Attacks

## Run order
1. `genetic/Genetic_Algorithm.ipynb` - Task 4: evolve AI text past XGBoost detector
2. `clauses_assemble/Clause_Assembler.ipynb` - Task 5: Monte Carlo punctuation humanizer

## Genetic Algorithm (Task 4)

Evolves AI text until Statistician XGBoost labels it Human with >95% confidence.

Pipeline: read AI text -> generate 5 diverse rewrites via Gemini (1 call) -> run 5 generations of elitist GA with feature-guided mutations (5 calls) -> precision strike targeting vocab repetition (1 call) -> save to output.txt

Result: 0.2% -> 99.2% P(Human) in 7 API calls.

Key insight: MTLD and TTR were the bottleneck. AIs synonym-hunting bias (avoiding word repetition) is the critical vulnerability. Once you force repetition, diversity scores crash to human levels.

| File | What |
|------|------|
| `genetic/Genetic_Algorithm.ipynb` | main notebook |
| `genetic/fitness_detector.py` | reusable AIDetector class, XGBoost on 26 features |
| `genetic/input.txt` | original AI paragraph |
| `genetic/output.txt` | evolved human-passing paragraph |

## Clause Assembler (Task 5)

Uses Gemini as a parser (not writer) to identify thought units, then Python applies probabilistic punctuation via log-normal distribution. Separates recognition from decision-making.

Pipeline: deconstruction into thought units with weight/prosodic scores -> Monte Carlo assembly (mu=1.8, sigma=1.2) -> pattern disruption (thinking commas, parallel structure breaking, 25% fragments / 25% run-ons / 50% normal)

Result: CopyLeaks still 100% AI. QuillBot drops to 0-20% AI.

| File | What |
|------|------|
| `clauses_assemble/Clause_Assembler.ipynb` | main notebook |
| `clauses_assemble/input.txt` | original AI paragraph |
| `clauses_assemble/output.txt` | punctuation-humanized output |

## Deps
google-genai, python-dotenv, xgboost, scikit-learn, spacy, nltk, numpy
