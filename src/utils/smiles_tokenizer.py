import json

class SmilesTokenizer:
    def __init__(self, vocab=None):
        if vocab is not None:
            self.vocab = vocab
        else:
            self.vocab = {}

        # Special tokens
        self.pad = "<pad>"
        self.start = "<s>"
        self.end = "</s>"

    def build_vocab(self, smiles_list):
        chars = set()

        for smi in smiles_list:
            for ch in smi:
                chars.add(ch)

        vocab = {self.pad: 0, self.start: 1, self.end: 2}
        
        for i, ch in enumerate(sorted(chars), start=3):
            vocab[ch] = i

        self.vocab = vocab
        self.inv_vocab = {i: ch for ch, i in vocab.items()}

    def encode(self, smiles: str):
        ids = [self.vocab[self.start]]
        for ch in smiles:
            ids.append(self.vocab[ch])
        ids.append(self.vocab[self.end])
        return ids

    def decode(self, ids):
        chars = []
        for i in ids:
            ch = self.inv_vocab.get(i, "")
            if ch in [self.start, self.end, self.pad]:
                continue
            chars.append(ch)
        return "".join(chars)

    def save(self, path):
        with open(path, "w") as f:
            json.dump(self.vocab, f)

    def load(self, path):
        with open(path, "r") as f:
            self.vocab = json.load(f)
            self.inv_vocab = {i: ch for ch, i in self.vocab.items()}