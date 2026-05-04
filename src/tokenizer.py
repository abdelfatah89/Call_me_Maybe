from llm_sdk import Small_LLM_Model
from typing import List, Tuple, Dict
import json
from src.constrained import char_replace, replace_char


class Tokenizer:
    def __init__(self, model: Small_LLM_Model):
        self.merge_rules_path = model.get_path_to_merges_file()
        self.vocab_path = model.get_path_to_vocab_file()

    def get_pairs(self) -> List[Tuple[str, str]]:
        merge_rules: List[Tuple[str, str]] = []
        with open(self.merge_rules_path, 'r') as f:
            for line in f:
                p1, p2 = line.split()
                merge_rules.append((p1, p2))
        return merge_rules

    def get_merges(self) -> Dict[Tuple[str, str], int]:
        pairs = self.get_pairs()
        merges = {pair: i for i, pair in enumerate(pairs)}
        return merges

    def get_vocab(self) -> Tuple[Dict[str, int], Dict[int, str]]:
        with open(self.vocab_path) as f:
            token_to_id = json.load(f)
        id_to_token = {int(v): k for k, v in token_to_id.items()}
        return token_to_id, id_to_token

    def get_word_pairs(self, tokens: List[str]):
        pairs = []
        for i in range(len(tokens) - 1):
            pairs.append((tokens[i], tokens[i+1]))
        return pairs

    def tokenize(self, word: str) -> List[str]:
        tokens = list(char_replace(word))
        merges = self.get_merges()
        while True:
            pairs = self.get_word_pairs(tokens)
            valid_pairs = [p for p in pairs if p in merges]
            if not valid_pairs:
                break

            best_pair = min(valid_pairs, key=lambda p: merges[p])
            new_tokens = []
            i = 0
            while i < len(tokens):
                if (i < len(tokens) - 1
                    and (tokens[i], tokens[i+1]) == best_pair):
                    new_tokens.append(tokens[i] + tokens[i+1])
                    i += 2
                else:
                    new_tokens.append(tokens[i])
                    i += 1
            tokens = new_tokens
        return tokens

    def decode(self, word: str) -> List[int]:
        tokens = self.tokenize(word)
        token_to_id, _ = self.get_vocab()
        ids = [token_to_id[t] for t in tokens]
        return ids

    def encode(self, ids: List[int]) -> List[str]:
        _, id_to_token = self.get_vocab()
        tokens = [id_to_token[i] for i in ids]
        return tokens


if __name__ == "__main__":
    m = Small_LLM_Model()
    t = Tokenizer(m)
    t.tokenize("hi my name is james, and i'm a coder at 1337")
    ids = t.decode("hi my name is james, and i'm a coder at 1337")
    tokens = t.encode(ids)
    print(replace_char(''.join(tokens)))
