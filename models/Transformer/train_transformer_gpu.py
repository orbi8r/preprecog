# GPU training script for DeBERTa/DistilBERT with LoRA
import pandas as pd
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, roc_auc_score, accuracy_score
from torch.utils.data import Dataset
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification, Trainer, TrainingArguments
from peft import LoraConfig, get_peft_model, TaskType
import warnings
import pickle
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings('ignore')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.cuda.empty_cache()
print(f'Using device: {device}')

print('\n' + "="*50)
print("LOADING DATA")
print("="*50)
source_path = '../../data_analysis/data.parquet'
df_raw = pd.read_parquet(source_path)

features_expanded = pd.json_normalize(df_raw['feature_cache'])
df = pd.concat([df_raw[['id', 'class', 'topic', 'text']].reset_index(drop=True),
                features_expanded[['author']].reset_index(drop=True)], axis=1)

df['label'] = (df['class'] > 1).astype(int)
print(f'Loaded {len(df)} rows')
print(f'Class distribution: {df["label"].value_counts().to_dict()}')

df['strat_key'] = df['class'].astype(str) + '_' + df['author'].astype(str)
train_val, test = train_test_split(df, test_size=0.15, stratify=df['strat_key'], random_state=42)
train, val = train_test_split(train_val, test_size=0.15/0.85, stratify=train_val['strat_key'], random_state=42)
print(f'Train: {len(train)}, Val: {len(val)}, Test: {len(test)}')

train[['id', 'class', 'topic', 'author']].to_parquet('train.parquet', index=False)
val[['id', 'class', 'topic', 'author']].to_parquet('validate.parquet', index=False)
test[['id', 'class', 'topic', 'author']].to_parquet('test.parquet', index=False)

print("\n" + "="*50)
print("MODEL SETUP")
print("="*50)
model_name = 'distilbert-base-uncased'
tokenizer = DistilBertTokenizer.from_pretrained(model_name)

base_model = DistilBertForSequenceClassification.from_pretrained(
    model_name,
    num_labels=2,
    problem_type='single_label_classification'
)
print(f'Loaded {model_name}')

lora_config = LoraConfig(
    task_type=TaskType.SEQ_CLS,
    r=16,
    lora_alpha=32,
    lora_dropout=0.1,
    target_modules=['q_lin', 'v_lin']
)

model = get_peft_model(base_model, lora_config)
model.print_trainable_parameters()

class TextDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=256):
        self.texts = texts.reset_index(drop=True)
        self.labels = labels.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts.iloc[idx])
        label = int(self.labels.iloc[idx])
        
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

train_dataset = TextDataset(train['text'], train['label'], tokenizer)
val_dataset = TextDataset(val['text'], val['label'], tokenizer)
test_dataset = TextDataset(test['text'], test['label'], tokenizer)
print(f'Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}')

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    probs = torch.softmax(torch.tensor(logits), dim=1).numpy()
    preds = np.argmax(logits, axis=1)
    acc = accuracy_score(labels, preds)
    auc = roc_auc_score(labels, probs[:, 1])
    return {'accuracy': acc, 'auc': auc}

print("\n" + "="*50)
print("TRAINING")
print("="*50)

training_args = TrainingArguments(
    output_dir='./transformer_checkpoints',
    num_train_epochs=10,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=8,
    gradient_accumulation_steps=4,
    warmup_steps=50,
    weight_decay=0.01,
    learning_rate=2e-4,
    logging_dir='./logs',
    logging_steps=25,
    eval_strategy='epoch',
    save_strategy='epoch',
    load_best_model_at_end=True,
    metric_for_best_model='auc',
    greater_is_better=True,
    report_to='none',
    fp16=True,
    dataloader_num_workers=0,
    optim='adamw_torch',
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    compute_metrics=compute_metrics
)

trainer.train()
print('Training complete')

print("\n" + "="*50)
print("EVALUATION")
print("="*50)

val_predictions = trainer.predict(val_dataset)
val_probs = torch.softmax(torch.tensor(val_predictions.predictions), dim=1).numpy()[:, 1]
val_preds = np.argmax(val_predictions.predictions, axis=1)

val_results = val.copy().reset_index(drop=True)
val_results['prob_ai'] = val_probs
val_results['pred'] = val_preds
val_results['actual_binary'] = val_results['label']

auc = roc_auc_score(val_results['actual_binary'], val_results['prob_ai'])
acc = accuracy_score(val_results['actual_binary'], val_results['pred'])
print(f'Validation AUC: {auc:.4f}')
print(f'Validation Accuracy: {acc:.4f}')

results = {
    'val_results': val_results,
    'val_probs': val_probs,
    'val_preds': val_preds,
    'auc': auc,
    'acc': acc,
    'train': train,
    'val': val,
    'test': test
}
with open('transformer_results.pkl', 'wb') as f:
    pickle.dump(results, f)
print('Results saved to transformer_results.pkl')

print("\n" + "="*50)
print("GENERATING PLOTS")
print("="*50)

sns.set_style('whitegrid')
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

fpr, tpr, _ = roc_curve(val_results['actual_binary'], val_results['prob_ai'])

ax1 = axes[0]
ax1.plot(fpr, tpr, 'k-', lw=3, label=f'Overall (AUC={auc:.2f})')

colors = plt.cm.tab10(np.linspace(0, 1, len(val_results['author'].unique())))
for i, author in enumerate(val_results['author'].unique()):
    mask = val_results['author'] == author
    if mask.sum() > 5:
        y_true_sub = val_results[mask]['actual_binary']
        y_prob_sub = val_results[mask]['prob_ai']
        if len(y_true_sub.unique()) > 1:
            fpr_sub, tpr_sub, _ = roc_curve(y_true_sub, y_prob_sub)
            auc_sub = roc_auc_score(y_true_sub, y_prob_sub)
            ax1.plot(fpr_sub, tpr_sub, color=colors[i], lw=1.5, alpha=0.7,
                     label=f'{author} (AUC={auc_sub:.2f})')

ax1.plot([0, 1], [0, 1], 'k--', lw=1)
ax1.set_xlabel('False Positive Rate')
ax1.set_ylabel('True Positive Rate')
ax1.set_title('ROC Curve by Author')
ax1.legend(loc='lower right', fontsize=8)

ax2 = axes[1]
ax2.plot(fpr, tpr, 'k-', lw=3, label=f'Overall (AUC={auc:.2f})')

colors_t = plt.cm.Set2(np.linspace(0, 1, len(val_results['topic'].unique())))
for i, topic in enumerate(val_results['topic'].unique()):
    mask = val_results['topic'] == topic
    if mask.sum() > 5:
        y_true_sub = val_results[mask]['actual_binary']
        y_prob_sub = val_results[mask]['prob_ai']
        if len(y_true_sub.unique()) > 1:
            fpr_sub, tpr_sub, _ = roc_curve(y_true_sub, y_prob_sub)
            auc_sub = roc_auc_score(y_true_sub, y_prob_sub)
            ax2.plot(fpr_sub, tpr_sub, color=colors_t[i], lw=1.5, alpha=0.7,
                     label=f'{topic} (AUC={auc_sub:.2f})')

ax2.plot([0, 1], [0, 1], 'k--', lw=1)
ax2.set_xlabel('False Positive Rate')
ax2.set_ylabel('True Positive Rate')
ax2.set_title('ROC Curve by Topic')
ax2.legend(loc='lower right', fontsize=8)

plt.tight_layout()
plt.savefig('transformer_roc_curves.png', dpi=150)
print('ROC curves saved to transformer_roc_curves.png')

print("\n" + "="*50)
print("TEST SET EVALUATION")
print("="*50)

test_predictions = trainer.predict(test_dataset)
test_probs = torch.softmax(torch.tensor(test_predictions.predictions), dim=1).numpy()[:, 1]
test_preds = np.argmax(test_predictions.predictions, axis=1)

test_results = test.copy().reset_index(drop=True)
test_results['prob_ai'] = test_probs
test_results['pred'] = test_preds
test_results['actual_binary'] = test_results['label']

test_auc = roc_auc_score(test_results['actual_binary'], test_results['prob_ai'])
test_acc = accuracy_score(test_results['actual_binary'], test_results['pred'])
print(f'Test AUC: {test_auc:.4f}')
print(f'Test Accuracy: {test_acc:.4f}')

results['test_results'] = test_results
results['test_probs'] = test_probs
results['test_preds'] = test_preds
results['test_auc'] = test_auc
results['test_acc'] = test_acc

with open('transformer_results.pkl', 'wb') as f:
    pickle.dump(results, f)

trainer.save_model('./transformer_final_model')
print('Model saved to ./transformer_final_model')

print("\n" + "="*50)
print("DONE!")
print("="*50)
