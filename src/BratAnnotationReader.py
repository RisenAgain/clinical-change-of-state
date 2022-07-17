class BratAnnotationReader:
    def __init__(self, filename) -> None:
        self.parse(filename)
    
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

    def getAnnotations(self) -> dict:
        return self.annotations

    def getEntities(self) -> dict:
        return self.annotations['entities']
    
    def getRelations(self) -> dict:
        return self.annotations['relations']
    