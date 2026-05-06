from llm_sdk import Small_LLM_Model
from typing import List, Tuple, Dict
import json
from functools import lru_cache


class Tokenizer:
    def __init__(self, merges_path, vocab_path):
        self.merges_path = merges_path
        self.vocab_path = vocab_path

        self.merges = self._load_merges()
        self.token_to_id, self.id_to_token = self._load_vocab()

    def _load_merges(self) -> Dict[Tuple[str, str], int]:
        merges = {}
        with open(self.merges_path, 'r') as f:
            for i, line in enumerate(f):
                p1, p2 = line.split()
                merges[(p1, p2)] = i
        return merges

    def _load_vocab(self) -> Tuple[Dict[str, int], Dict[int, str]]:
        with open(self.vocab_path) as f:
            token_to_id = json.load(f)
        id_to_token = {int(v): k for k, v in token_to_id.items()}
        return token_to_id, id_to_token

    def tokenize(self,text: str) -> List[str]:
        words = text.split()
        tokens: List[str] = []
        for word in words:
            tokens.extend(self._bpe_cached(word))
        return tokens

    def _bpe(self, word: str) -> List[str]:
        tokens = list(word)

        while True:
            best_pair = None
            best_rank = float('inf')

            for i in range(len(tokens) - 1):
                pair = (tokens[i], tokens[i+1])
                rank = self.merges.get(pair)
                if rank is not None and rank < best_rank:
                    best_rank = rank
                    best_pair = pair

            if best_pair is None:
                break

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

    @lru_cache(10000)
    def _bpe_cached(self, word: str) -> Tuple[str, ...]:
        return tuple(self._bpe(word))

    @lru_cache(maxsize=4096)
    def encode(self, word: str) -> Tuple[int, ...]:
        tokens = self.tokenize(word)
        return tuple(self.token_to_id[t] for t in tokens)

    @lru_cache(maxsize=4096)
    def decode(self, ids: Tuple[int, ...]) -> str:
        return "".join(self.id_to_token[i] for i in ids)


if __name__ == "__main__":
    from src.constrained import replace_char
    m = Small_LLM_Model()
    t = Tokenizer(m.get_path_to_merges_file(),
                  m.get_path_to_vocab_file())
    t.tokenize("hi my name is james, and i'm a coder at 1337")
    ids = t.encode("hi my name is james, and i'm a coder at 1337")
    tokens = t.decode(ids)
    print(replace_char(''.join(tokens)))
