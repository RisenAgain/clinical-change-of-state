from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline,DataCollatorForTokenClassification 
from BratAnnotationReader import BratAnnotationReader
import pickle
import pandas as pd
# from sklearn import metrics
from seqeval.metrics import classification_report

output_dir = '../outputs/'
postprocess = False
predictions_file = f'predictions{"_postprocess" if postprocess else ""}.out'

eval_mode = 'strict'
result_dir = '../results/'
result_file = f'scores_{eval_mode}{"_postprocess" if postprocess else ""}.out'

def flatten(xss):
    return [x for xs in xss for x in xs]

def post_process(predictions, mode='Leading_I_to_B'):
    processed_predictions = []
    for sent_labels in predictions:
        new_sent_labels = sent_labels.copy()
        if mode == 'Leading_I_to_B':
            for i in range(len(sent_labels)):
                if sent_labels[i].startswith('I') and (i == 0 or sent_labels[i-1][2:] != sent_labels[i][2:]):
                    new_sent_labels[i] = 'B-' + sent_labels[i][2:]
        processed_predictions.append(new_sent_labels)
    return processed_predictions

# def get_confusion_matrix(gold, pred, labels, mode='word'):
#     matrix = {}
#     for label1 in labels:
#         matrix[label1] = {label2: 0 for label2 in labels}
    
#     if mode == 'word':
#         for sent_n in range(len(gold)):
#             for word_n in range(len(gold)):
#                 matrix[gold[sent_n][word_n]][pred[sent_n][word_n]] += 1
#     elif mode == 'entity':


# def get_metrics(cf_matrix):
#     scores = {}
#     for label in cf_matrix:
#         scores[label] = {'precision': 0.00, 'recall': 0.00, 'f1-score': 0.00, 'support': 0}

    
#     for label in cf_matrix:
#         scores[label]['precision'] = cf_matrix[label][label]/
        

model_name = 'emilyalsentzer/Bio_ClinicalBERT'
save_path = f'../model/{model_name}'

labels = BratAnnotationReader.get_entity_lables(BIO=True)

labels_to_ids = {k: v for v, k in enumerate(sorted(labels))}
ids_to_labels = {v: k for v, k in enumerate(sorted(labels))}


corpus = {}
corpus['train'] = '../data/processed/labeled_snippets_train.pkl'
corpus['test'] = '../data/processed/labeled_snippets_test.pkl'
corpus['dev'] = '../data/processed/labeled_snippets_dev.pkl'


# def get_dataset(filename):
#     instances = pickle.load(open(filename, 'rb'))
#     examples = []
#     for sentence in instances:
#         example = {}
#         example['tokens'] = [word[0] for word in sentence]
#         example['labels'] = [word[2] for word in sentence]
#         examples.append(example)
#     return examples

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


dataset = get_dataset(corpus['test'])

tokenizer = AutoTokenizer.from_pretrained(save_path)
model = AutoModelForTokenClassification.from_pretrained(save_path)


def tokenize_and_align_labels(examples):
    tokenized_inputs = tokenizer(examples['tokens'], is_split_into_words=True)
    labels = []
    for i, label in enumerate(examples['labels']):
        word_ids = tokenized_inputs.word_ids(batch_index=i)
        label_ids = [-100 if word_idx is None else label[word_idx] for word_idx in word_ids]
        labels.append(label_ids)
    tokenized_inputs['labels'] = labels
    return tokenized_inputs


# dataset = Dataset.from_pandas(pd.DataFrame(examples))
# tokenized_dataset = dataset.map(tokenize_and_align_labels, batched=True, remove_columns=dataset.column_names)
# data_collator = DataCollatorForTokenClassification(tokenizer)
# padded_dataset = data_collator([tokenized_dataset[i] for i in range(len(tokenized_dataset))])
classifier = pipeline("token-classification", model=model, tokenizer=tokenizer)
predictions = [classifier(sequence) for sequence in dataset['tokens']]
predictions = [
    [ max(subwords, key=lambda x: x['score'])['entity'] for subwords in sent ] 
    for sent in predictions
]
predictions = [
    [ids_to_labels[int(label[6:])] for label in sent]
    for sent in predictions
]

gold = [
    [ids_to_labels[label] for label in sent_labels]
    for sent_labels in dataset['labels']
]


if postprocess:
    predictions = post_process(predictions)

with open(f'{output_dir}{predictions_file}', 'w') as f:
    for i in range(len(gold)):
        f.write('\t'.join(dataset['tokens'][i]))
        f.write('\n')
        f.write('\t'.join(gold[i]))
        f.write('\n')
        f.write('\t'.join(predictions[i]))
        f.write('\n')

scores = classification_report(gold, predictions, mode=eval_mode)

with open(f'{result_dir}{result_file}', 'w') as f:
    f.write(scores)





# labels_without_O = list(set(labels) - set(['O']))
# print(metrics.classification_report(flatten(gold), flatten(predictions), labels=labels_without_O))

# print(gold[0], len(gold[0]))
# print(dataset['labels'][0], len(dataset['tokens'][0]))
# print(predictions[0], len(predictions[0]))
# print(dataset['tokens'][1], len(dataset['tokens'][1]))
# print(predictions[1], len(predictions[1]))
# outputs = model(padded_dataset['tokens']).logits
# print(outputs.shape)
