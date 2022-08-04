from transformers import AutoTokenizer, AutoModelForTokenClassification, DataCollatorForTokenClassification, TrainingArguments, Trainer
from datasets import Dataset
import pandas as pd
import pickle
from BratAnnotationReader import BratAnnotationReader


model_name = 'emilyalsentzer/Bio_ClinicalBERT'
save_path = f'../model/{model_name}'
num_labels = 11

labels = BratAnnotationReader.get_entity_lables(BIO=True)

labels_to_ids = {k: v for v, k in enumerate(sorted(labels))}
ids_to_labels = {v: k for v, k in enumerate(sorted(labels))}

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForTokenClassification.from_pretrained(model_name, num_labels=num_labels)

corpus = {}
corpus['train'] = '../data/processed/labeled_snippets_train.pkl'
corpus['test'] = '../data/processed/labeled_snippets_test.pkl'
corpus['dev'] = '../data/processed/labeled_snippets_dev.pkl'


def get_dataset(filename):
    instances = pickle.load(open(filename, 'rb'))
    instances = {
        'tokens': [[word for word in sentence[0]] for sentence in instances],
        'labels': [
            [labels_to_ids[label] for label in sentence[1]] for sentence in instances
        ]
    }
    df = pd.DataFrame(instances)
    return Dataset.from_pandas(df)


def tokenize_and_align_labels(examples):
    tokenized_inputs = tokenizer(examples['tokens'], is_split_into_words=True)
    labels = []
    for i, label in enumerate(examples['labels']):
        word_ids = tokenized_inputs.word_ids(batch_index=i)
        label_ids = [-100 if word_idx is None else label[word_idx] for word_idx in word_ids]
        labels.append(label_ids)
    tokenized_inputs['labels'] = labels
    return tokenized_inputs


dataset = get_dataset(corpus['train'])
tokenized_dataset = dataset.map(tokenize_and_align_labels, batched=True)
data_collator = DataCollatorForTokenClassification(tokenizer)

# eval_dataset = get_dataset(corpus['dev'])
# eval_tokenized_dataset = eval_dataset.map(tokenize_and_align_labels, batched=True)


args = TrainingArguments(
        f'checkpoints/{model_name}'
#         # evaluation_strategy='epoch',
#         # save_strategy='epoch',
#         # load_best_model_at_end=True,
#         # learning_rate=4e-5,
#         # num_train_epochs=epochs
    )

trainer = Trainer(
    model,
    args,
    train_dataset=tokenized_dataset,
    # eval_dataset=eval_tokenized_dataset,
    data_collator=data_collator,
    tokenizer=tokenizer
)

trainer.train()

tokenizer.save_pretrained(save_path)
model.save_pretrained(save_path)