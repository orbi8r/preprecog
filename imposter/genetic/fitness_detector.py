"""
fitness_detector.py — AI Text Detector (Fitness Function for Genetic Algorithm)
================================================================================

Combines the Statistician (XGBoost on 26 features) approach from the PKOG
research project. Trains on data_analysis/data.parquet and provides a fitness
scoring function for the Genetic Algorithm.

Features used (26 total):
  - 7 Lexical:   ttr, hapax_5k, mtld, yules_k, n_tokens, n_types, word_count
  - 2 Sentence:  sent_len_std, avg_sent_length
  - 4 Syntactic: adj_noun_ratio, tree_depth, fk_grade, discourse_density
  - 11 Punctuation: semicolon, colon, exclamation, question, hyphen, emdash,
                     asterisk, apos, apos_curly, paren_open, quote
  - 2 Stylometric: function_word_pca.dim1, function_word_pca.dim2

Usage:
    from fitness_detector import AIDetector
    detector = AIDetector()
    score = detector.score_human("Some text here")  # Returns P(Human) ∈ [0,1]
"""

import os
import re
import math
import random
import warnings
import numpy as np
import pandas as pd
from collections import Counter

import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import CountVectorizer

warnings.filterwarnings("ignore")

# ─── NLP Setup ──────────────────────────────────────────────────────────────
import spacy
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    print("Downloading spaCy model en_core_web_sm...")
    from spacy.cli import download
    download("en_core_web_sm")
    nlp = spacy.load("en_core_web_sm")

import nltk
for resource in ['punkt_tab', 'punkt']:
    try:
        nltk.data.find(f'tokenizers/{resource}')
        break
    except LookupError:
        nltk.download(resource, quiet=True)

# ─── Constants ──────────────────────────────────────────────────────────────
WORD_RE = re.compile(r"\b\w+\b", re.UNICODE)

PUNCT_MAP = {
    ';':      'punct_semicolon',
    ':':      'punct_colon',
    '!':      'punct_exclamation',
    '?':      'punct_question',
    '-':      'punct_hyphen',
    '\u2014': 'punct_emdash',       # —
    '*':      'punct_asterisk',
    "'":      'punct_apos',
    '\u2019': 'punct_apos_curly',   # '
    '(':      'punct_paren_open',
    '"':      'punct_quote',
}

DISCOURSE_MARKERS = {
    "however", "therefore", "consequently", "furthermore", "moreover",
    "nevertheless", "thus", "hence", "accordingly", "subsequently",
    "conversely", "meanwhile", "nonetheless", "notwithstanding",
    "additionally", "alternatively", "undoubtedly", "specifically",
    "similarly", "finally", "indeed",
}

# The 26 features the model uses (sorted alphabetically for consistency)
FEATURE_COLS = [
    'adj_noun_ratio', 'avg_sent_length', 'discourse_density_per_100_words',
    'fk_grade', 'function_word_pca.dim1', 'function_word_pca.dim2',
    'hapax_5k', 'mtld', 'n_tokens', 'n_types',
    'punct_apos', 'punct_apos_curly', 'punct_asterisk', 'punct_colon',
    'punct_emdash', 'punct_exclamation', 'punct_hyphen', 'punct_paren_open',
    'punct_question', 'punct_quote', 'punct_semicolon',
    'sent_len_std', 'tree_depth', 'ttr', 'word_count', 'yules_k',
]


# ═══════════════════════════════════════════════════════════════════════════
# FEATURE COMPUTATION FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════

def tokenize(text):
    """Regex-based tokenizer matching the original pipeline."""
    return [w.lower() for w in WORD_RE.findall(text)] if isinstance(text, str) else []


# ─── Lexical Features ───────────────────────────────────────────────────────

def type_token_ratio(tokens):
    """V/N — higher = more varied vocabulary."""
    N = len(tokens)
    return len(set(tokens)) / N if N > 0 else 0.0


def hapax_in_sample(tokens, sample_size=5000, seed=42):
    """Count of words appearing exactly once in a 5000-word sample."""
    N = len(tokens)
    if N == 0:
        return 0
    if N <= sample_size:
        sample = tokens
    else:
        rng = random.Random(seed)
        start = rng.randint(0, N - sample_size)
        sample = tokens[start:start + sample_size]
    freq = Counter(sample)
    return sum(1 for _, c in freq.items() if c == 1)


def mtld_calc(tokens, ttr_threshold=0.72):
    """Measure of Textual Lexical Diversity — robust to text length."""
    def _single_pass(token_list):
        factor_count = 0
        token_count = 0
        types = set()
        for w in token_list:
            token_count += 1
            types.add(w)
            ttr = len(types) / token_count
            if ttr <= ttr_threshold:
                factor_count += 1
                token_count = 0
                types = set()
        if token_count > 0:
            ttr_val = len(types) / token_count if token_count else 0
            partial = (1 - ttr_val) / (1 - ttr_threshold) if (1 - ttr_threshold) != 0 else 0
            factor_count += partial
        return (len(token_list) / factor_count) if factor_count != 0 else float('inf')

    if not tokens:
        return 0.0
    forward = _single_pass(tokens)
    backward = _single_pass(list(reversed(tokens)))
    val = (forward + backward) / 2
    return val if val != float('inf') else 0.0


def yules_k(tokens):
    """Yule's K — measures vocabulary concentration. Lower = richer."""
    N = len(tokens)
    if N == 0:
        return 0.0
    freq = Counter(tokens)
    freq_of_freq = Counter(freq.values())
    s = sum(r * r * Vr for r, Vr in freq_of_freq.items())
    return 10000 * (s - N) / (N * N) if N > 0 else 0.0


# ─── Sentence Features ──────────────────────────────────────────────────────

def sentence_stats(text):
    """Return (avg_sent_length, sent_len_std) using NLTK sentence tokenizer."""
    sents = nltk.sent_tokenize(text)
    if not sents:
        return 0.0, 0.0
    lengths = [len(tokenize(s)) for s in sents]
    if not lengths:
        return 0.0, 0.0
    mean = sum(lengths) / len(lengths)
    var = sum((l - mean) ** 2 for l in lengths) / len(lengths)
    return mean, math.sqrt(var)


# ─── Syntactic Features (spaCy) ─────────────────────────────────────────────

def _tree_depth(token):
    """Recursive dependency tree depth."""
    children = list(token.children)
    if not children:
        return 1
    return 1 + max(_tree_depth(c) for c in children)


def _syllable_count(word):
    """Approximate syllable count for Flesch-Kincaid."""
    word = word.lower()
    count = len(re.findall(r'[aeiouy]+', word))
    if word.endswith('e'):
        count -= 1
    return max(1, count)


def analyze_syntax(text):
    """Compute adj_noun_ratio, tree_depth, fk_grade, discourse_density."""
    if not text or not text.strip():
        return {
            "adj_noun_ratio": 0.0, "tree_depth": 0.0,
            "fk_grade": 0.0, "discourse_density_per_100_words": 0.0,
        }
    doc = nlp(text)

    adjs = len([t for t in doc if t.pos_ == "ADJ"])
    nouns = len([t for t in doc if t.pos_ == "NOUN"])
    adj_noun_ratio = adjs / nouns if nouns > 0 else 0.0

    # Average dependency tree depth
    depths = []
    for sent in doc.sents:
        try:
            depths.append(_tree_depth(sent.root))
        except RecursionError:
            depths.append(0)
    avg_depth = sum(depths) / len(depths) if depths else 0.0

    # Flesch-Kincaid grade level
    n_words = len([t for t in doc if not t.is_punct])
    n_sents = max(len(list(doc.sents)), 1)
    n_syllables = sum(_syllable_count(t.text) for t in doc if not t.is_punct)
    fk = (0.39 * (n_words / n_sents) + 11.8 * (n_syllables / max(n_words, 1)) - 15.59) \
        if n_words > 0 else 0.0

    # Discourse marker density
    disc_count = len([t for t in doc if t.lower_ in DISCOURSE_MARKERS])
    disc_density = (disc_count / n_words) * 100 if n_words > 0 else 0.0

    return {
        "adj_noun_ratio": adj_noun_ratio,
        "tree_depth": avg_depth,
        "fk_grade": fk,
        "discourse_density_per_100_words": disc_density,
    }


# ─── Punctuation Features ───────────────────────────────────────────────────

def punct_density(text):
    """Per-1000-character density of 11 punctuation types."""
    text = text or ''
    L = len(text)
    if L == 0:
        return {name: 0.0 for name in PUNCT_MAP.values()}
    c = Counter(text)
    return {name: (c.get(sym, 0) / L) * 1000 for sym, name in PUNCT_MAP.items()}


# ═══════════════════════════════════════════════════════════════════════════
# AI DETECTOR CLASS
# ═══════════════════════════════════════════════════════════════════════════

class AIDetector:
    """
    XGBoost-based AI text detector trained on 26 linguistic features.

    Combines insights from:
      - Statistician Model (XGBoost on handcrafted features)
      - Feature engineering from LexicalRichness, SyntacticComplexity,
        RowFeatureEnrichment notebooks
      - Function-word PCA stylometry

    Usage:
        detector = AIDetector()
        p_human = detector.score_human("Some paragraph text")
        details = detector.score_detailed("Some paragraph text")
    """

    def __init__(self, data_path=None):
        if data_path is None:
            script_dir = os.path.dirname(os.path.abspath(__file__))
            data_path = os.path.join(script_dir, '..', '..', 'data_analysis', 'data.parquet')
        self.data_path = os.path.abspath(data_path)

        self.model = None
        self.feature_cols = FEATURE_COLS
        self.pca = None
        self.vectorizer = None
        self.function_indices = None

        self._load_and_train()

    # ─── Training Pipeline ───────────────────────────────────────────────

    def _load_and_train(self):
        """Load corpus, extract features, fit PCA, train XGBoost."""
        print("[Detector] Loading data from:", self.data_path)
        df = pd.read_parquet(self.data_path)
        print(f"  Loaded {len(df)} samples ({df['class'].value_counts().to_dict()})")

        # 1. Expand pre-computed feature_cache
        features = pd.json_normalize(df['feature_cache'])

        # Drop non-numeric metadata
        drop_cols = ['author', 'book_title', 'persona_mimicked']
        features = features.drop(
            [c for c in drop_cols if c in features.columns], axis=1, errors='ignore'
        )

        # Drop pre-computed PCA (we'll recompute for consistency)
        pca_cols = [c for c in features.columns if 'function_word_pca' in c]
        features = features.drop(pca_cols, axis=1, errors='ignore')

        # 2. Fit PCA pipeline on corpus texts
        texts = df['text'].fillna('').astype(str)
        corpus_pca = self._fit_pca_pipeline(texts)

        # Add recomputed PCA values
        features['function_word_pca.dim1'] = corpus_pca[:, 0]
        features['function_word_pca.dim2'] = corpus_pca[:, 1]

        # 3. Select exactly the 26 expected features
        missing = [c for c in self.feature_cols if c not in features.columns]
        if missing:
            print(f"  WARNING: Missing features (will zero-fill): {missing}")
            for c in missing:
                features[c] = 0.0

        X = features[self.feature_cols].values.astype(np.float32)
        y = (df['class'].values - 1)  # 0=Human, 1=StdAI, 2=ImposterAI

        # 4. Train/Val split
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.15, random_state=42, stratify=y
        )

        # 5. Train XGBoost
        print("[Detector] Training XGBoost (26 features, 3 classes)...")
        self.model = xgb.XGBClassifier(
            objective='multi:softprob',
            num_class=3,
            n_estimators=1000,
            learning_rate=0.05,
            max_depth=6,
            subsample=0.8,
            colsample_bytree=0.8,
            gamma=0.1,
            random_state=42,
            early_stopping_rounds=50,
            n_jobs=-1,
            eval_metric='mlogloss',
        )
        self.model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)

        # 6. Report accuracy
        y_prob = self.model.predict_proba(X_val)
        y_prob_ai = y_prob[:, 1] + y_prob[:, 2]
        acc = ((y_prob_ai > 0.5).astype(int) == (y_val > 0).astype(int)).mean()
        print(f"  Binary validation accuracy: {acc:.4f}")
        print("[Detector] ✓ Ready")

    def _fit_pca_pipeline(self, texts):
        """Fit CountVectorizer + PCA on corpus function words. Returns (N, 2) array."""
        self.vectorizer = CountVectorizer(stop_words=None, max_features=100)
        dtm = self.vectorizer.fit_transform(texts)
        all_words = self.vectorizer.get_feature_names_out()

        # Select function words (spaCy stop words)
        std_stops = spacy.lang.en.stop_words.STOP_WORDS
        self.function_indices = [i for i, w in enumerate(all_words) if w in std_stops]
        if len(self.function_indices) < 20:
            self.function_indices = list(range(min(50, len(all_words))))
        else:
            self.function_indices = self.function_indices[:50]

        fn_dtm = dtm[:, self.function_indices]
        row_sums = np.array(fn_dtm.sum(axis=1)).ravel()
        row_sums[row_sums == 0] = 1
        freq_matrix = fn_dtm.toarray() / row_sums[:, None]

        self.pca = PCA(n_components=2)
        self.pca.fit(freq_matrix)
        return self.pca.transform(freq_matrix)

    # ─── Single-Text Feature Computation ─────────────────────────────────

    def _compute_pca(self, text):
        """Compute function-word PCA for a single text."""
        dtm = self.vectorizer.transform([text])
        fn_dtm = dtm[:, self.function_indices]
        row_sum = fn_dtm.sum()
        if row_sum == 0:
            freq = np.zeros((1, len(self.function_indices)))
        else:
            freq = fn_dtm.toarray() / row_sum
        result = self.pca.transform(freq)
        return float(result[0, 0]), float(result[0, 1])

    def compute_features(self, text):
        """
        Compute all 26 features for a single text from scratch.
        Returns a dict keyed by feature name.
        """
        tokens = tokenize(text)
        avg_sent_len, sent_std = sentence_stats(text)
        syntax = analyze_syntax(text)
        punct = punct_density(text)
        pca_dim1, pca_dim2 = self._compute_pca(text)

        return {
            'adj_noun_ratio':                syntax['adj_noun_ratio'],
            'avg_sent_length':               avg_sent_len,
            'discourse_density_per_100_words': syntax['discourse_density_per_100_words'],
            'fk_grade':                      syntax['fk_grade'],
            'function_word_pca.dim1':        pca_dim1,
            'function_word_pca.dim2':        pca_dim2,
            'hapax_5k':                      hapax_in_sample(tokens),
            'mtld':                          mtld_calc(tokens),
            'n_tokens':                      len(tokens),
            'n_types':                       len(set(tokens)),
            'punct_apos':                    punct['punct_apos'],
            'punct_apos_curly':              punct['punct_apos_curly'],
            'punct_asterisk':                punct['punct_asterisk'],
            'punct_colon':                   punct['punct_colon'],
            'punct_emdash':                  punct['punct_emdash'],
            'punct_exclamation':             punct['punct_exclamation'],
            'punct_hyphen':                  punct['punct_hyphen'],
            'punct_paren_open':              punct['punct_paren_open'],
            'punct_question':                punct['punct_question'],
            'punct_quote':                   punct['punct_quote'],
            'punct_semicolon':               punct['punct_semicolon'],
            'sent_len_std':                  sent_std,
            'tree_depth':                    syntax['tree_depth'],
            'ttr':                           type_token_ratio(tokens),
            'word_count':                    len(text.split()),
            'yules_k':                       yules_k(tokens),
        }

    # ─── Scoring Interface ───────────────────────────────────────────────

    def score_human(self, text):
        """
        Score text for human-likeness.
        Returns P(Human) ∈ [0, 1].
        Higher = more human-like.
        """
        features = self.compute_features(text)
        X = np.array([[features[col] for col in self.feature_cols]], dtype=np.float32)
        probs = self.model.predict_proba(X)[0]
        return float(probs[0])  # P(class 0 = Human)

    def score_detailed(self, text):
        """
        Return a detailed scoring breakdown with all probabilities and features.
        """
        features = self.compute_features(text)
        X = np.array([[features[col] for col in self.feature_cols]], dtype=np.float32)
        probs = self.model.predict_proba(X)[0]
        return {
            'p_human':       float(probs[0]),
            'p_standard_ai': float(probs[1]),
            'p_imposter_ai': float(probs[2]),
            'features':      features,
        }

    def score_batch(self, texts):
        """Score multiple texts at once. Returns list of P(Human) values."""
        all_features = [self.compute_features(t) for t in texts]
        X = np.array(
            [[f[col] for col in self.feature_cols] for f in all_features],
            dtype=np.float32
        )
        probs = self.model.predict_proba(X)
        return [float(p[0]) for p in probs]

    def diagnose(self, text):
        """
        Identify which features are most AI-like for a given text.
        Useful for guiding targeted mutations.
        Returns list of (feature_name, value, human_mean, direction) sorted by severity.
        """
        features = self.compute_features(text)

        # Known human means from analysis (per-row averages)
        human_means = {
            'sent_len_std':     13.13,  # The "smoking gun" — AI is ~6.46
            'fk_grade':         14.0,   # Human complexity
            'mtld':             72.83,  # Human vocabulary fatigue
            'ttr':              0.617,
            'yules_k':          134.96,
            'hapax_5k':         75.89,
            'punct_semicolon':  2.46,   # Humans use 3.9× more semicolons
            'punct_apos':       1.97,   # Humans use 3.7× more apostrophes
            'punct_asterisk':   0.0,    # Asterisks are AI artifacts
            'adj_noun_ratio':   0.43,
            'tree_depth':       5.61,
            'avg_sent_length':  23.0,
            'discourse_density_per_100_words': 0.72,
        }

        # Known AI means
        ai_means = {
            'sent_len_std':     6.46,
            'fk_grade':         10.4,
            'mtld':             119.27,
            'ttr':              0.672,
            'yules_k':          89.26,
            'hapax_5k':         89.02,
            'punct_semicolon':  0.63,
            'punct_apos':       0.53,
            'punct_asterisk':   1.44,
            'adj_noun_ratio':   0.36,
            'tree_depth':       4.63,
            'avg_sent_length':  24.0,
            'discourse_density_per_100_words': 0.58,
        }

        issues = []
        for feat, human_val in human_means.items():
            if feat not in features:
                continue
            actual = features[feat]
            ai_val = ai_means.get(feat, human_val)

            # How far is this feature from human vs AI?
            human_dist = abs(actual - human_val)
            ai_dist = abs(actual - ai_val)
            total_range = abs(human_val - ai_val)

            if total_range > 0:
                # Score: 0 = perfectly human, 1 = perfectly AI
                ai_likeness = human_dist / (human_dist + ai_dist + 1e-9)
            else:
                ai_likeness = 0.0

            if ai_likeness > 0.4:  # More AI-like than human-like
                direction = "↑" if human_val > ai_val else "↓"
                issues.append((feat, actual, human_val, direction, ai_likeness))

        issues.sort(key=lambda x: x[4], reverse=True)
        return issues


# ═══════════════════════════════════════════════════════════════════════════
# STANDALONE TEST
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    detector = AIDetector()
    print()

    test_text = (
        "Furthermore, the existential dimension of truth and reality cannot be "
        "ignored. For thinkers like Søren Kierkegaard, truth is not merely a "
        "matter of factual accuracy but of lived authenticity."
    )
    result = detector.score_detailed(test_text)
    print(f"Test text P(Human): {result['p_human']:.4f}")
    print(f"          P(StdAI): {result['p_standard_ai']:.4f}")
    print(f"          P(Imposter): {result['p_imposter_ai']:.4f}")
    print()

    issues = detector.diagnose(test_text)
    if issues:
        print("Most AI-like features:")
        for feat, val, human_val, direction, severity in issues[:5]:
            print(f"  {feat}: {val:.2f} (human avg: {human_val:.2f}) {direction} severity: {severity:.2f}")
