import json
from llm_sdk import Small_LLM_Model
import numpy as np


MAX_TOKENS = 10

model = Small_LLM_Model()
vocab = model.get_path_to_vocab_file()

with open(vocab, 'r') as f:
    token_to_id = json.load(f)

id_to_token = {int(v): k for k, v in token_to_id.items()}

token_ids = model.encode("write a json schema: \n{")
tokens = [id_to_token[i] for i in token_ids.tolist()[0]]
text = "".join(tokens)
print(text)

input_ids = token_ids.tolist()[0]
for _ in range(MAX_TOKENS):
    logits = model.get_logits_from_input_ids(input_ids)
    new_ids = int(np.argmax(logits))
    tokens = [id_to_token[i] for i in [new_ids]]
    text = text + "".join(tokens)
    input_ids.append(new_ids)
    print(text)
