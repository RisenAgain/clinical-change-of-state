from collections import defaultdict
from nltk.tokenize import word_tokenize
from nltk.tokenize.util import align_tokens

class BratAnnotationReader:
    labels = {
        'entities': ['Cos', 'Attr', 'Loc', 'Val', 'Ref'],
        'relations': ['State', 'Location', 'Value', 'Referenced-by', 'Combine', 'Diag', 'Extend']
    }
    
    def __init__(self, textfile, annotationfile) -> None:
        self.parse(annotationfile)
        self.sentence_positions = []
        char_count = 0
        with open(textfile, 'r') as file:
            lines = file.readlines()
            for line in lines:
                self.sentence_positions.append({
                    'start': char_count,
                    'end': char_count + len(line),
                    'sent': line
                })
                char_count += len(line)        
    
    def parse(self, filename) -> None:
        self.annotations = {
            'entities': {},
            'relations': {},
            'notes': {}
        }

        with open(filename, 'r') as file:
            lines = file.read().splitlines()

        for line in lines:
            # parse entities
            if line.startswith('T'):
                tokens = line.split(maxsplit=4)
                if tokens[0] in self.annotations['entities']:
                    print(f"Error: Duplicate ID {tokens[0]} found.")
                else:
                    self.annotations['entities'][tokens[0]] = {
                        'type': tokens[1],
                        'start': int(tokens[2]),
                        'end': int(tokens[3]),
                        'text': tokens[4]
                    }
            # parse relations
            elif line.startswith('R'):
                tokens = line.split()
                if tokens[0] in self.annotations['relations']:
                    print(f"Error: Duplicate ID {tokens[0]} found.")
                else:
                    self.annotations['relations'][tokens[0]] = {
                        'type': tokens[1],
                        'arg1': tokens[2].split(':')[1],
                        'arg2': tokens[3].split(':')[1]
                    }
            elif line.startswith('#'):
                pass
            else:
                print("Error: Unsupported annotation found.")
                print(line)

    def get_sentence_id(self, char_pos) -> int:
        '''
        Get the sentence number given a character position 
        '''
        lo = 0
        hi = len(self.sentence_positions)-1
        while lo < hi:
            mid = lo + (hi-lo+1)//2
            if char_pos >= self.sentence_positions[mid]['start'] and \
                char_pos < self.sentence_positions[mid]['end']:
                return mid
            elif char_pos < self.sentence_positions[mid]['start']:
                hi = mid - 1
            elif char_pos >= self.sentence_positions[mid]['end']:
                lo = mid + 1
        return lo
    
    def get_sentence_annotations(self) -> dict:
        sentence_annotations = defaultdict(lambda: {'entities': {}, 'relations': {}})
        for entity, value in self.annotations['entities'].items():
            sent_id = self.get_sentence_id(value['start'])
            self.annotations['entities'][entity]['sent'] = sent_id
            sentence_annotations[sent_id]['entities'][entity] = value
        
        for relation, value in self.annotations['relations'].items():
            sent_id = self.annotations['entities'][value['arg1']]['sent']
            sentence_annotations[sent_id]['relations'][relation] = value

        return sentence_annotations
    
    def get_sentence_labels(self, sent_pos, sent_ann) -> tuple:
        words = word_tokenize(sent_pos['sent'])
        word_spans = align_tokens(words, sent_pos['sent'])
        word_labels = len(word_spans)*['O']

        for entity in sent_ann['entities'].values():
            start = entity['start'] - sent_pos['start']
            end = entity['end'] - sent_pos['start']
            if entity['type'] in BratAnnotationReader.labels['entities']:
                for i, word_span in enumerate(word_spans):
                    if start < word_span[1] and end > word_span[0]:
                        if word_labels[i] != 'O':
                            print("Overlapping label found.")
                            print(sent_pos, sent_ann)
                            print(words)
                        word_labels[i] = entity['type']
            
        for i in range(len(word_labels)-1, -1, -1):
            if word_labels[i] != 'O':
                if i == 0:
                    word_labels[i] = 'B-' + word_labels[i]
                elif word_labels[i] != word_labels[i-1]:
                    word_labels[i] = 'B-' + word_labels[i]
                else:
                    word_labels[i] = 'I-' + word_labels[i]
        
        return (words, word_labels)
        

    def get_labeled_sentences(self) -> list:
        labeled_sentences = []
        sentence_annotations = self.get_sentence_annotations()
        for sent_id in sorted(sentence_annotations):
            sentence_labels = self.get_sentence_labels(self.sentence_positions[sent_id], sentence_annotations[sent_id])
            labeled_sentences.append(sentence_labels)
        return labeled_sentences
    
    def get_annotations(self) -> dict:
        return self.annotations

    def get_overlapping_entities(self) -> dict:
        overlapping_entities = defaultdict(lambda: defaultdict(int))
        sentence_annotations = self.get_sentence_annotations()

        for sent_annotation in sentence_annotations.values():
            entities = list(sent_annotation['entities'].values())
            for i, val in enumerate(entities):
                for j in range(len(entities)):
                    if i != j:
                        if val['start'] < entities[j]['end'] and val['end'] > entities[j]['start']:
                            overlapping_entities[val['type']][entities[j]['type']] += 1
        return overlapping_entities

    @classmethod
    def get_entity_lables(cls, BIO=False) -> list:
        if BIO:
            labels = [[f'B-{label}', f'I-{label}'] for label in cls.labels['entities']]
            labels = [sublabel for label in labels for sublabel in label]
            return labels + ['O']
        return cls.labels['entities']

    
if __name__ == "__main__":
    brat_reader = BratAnnotationReader('../data/original/snippet501_600.txt', '../data/original/snippet501_600.ann')
    # ann = brat_reader.get_sentence_annotations()
    # print(ann[1]['relations'])
    # print(brat_reader.sentence_positions[1])
    # print(brat_reader.annotations['entities']['T18'])
    # print(brat_reader.get_sentence_id(94))
    ann = brat_reader.get_labeled_sentences()
    # print(len(ann))
    print(ann)