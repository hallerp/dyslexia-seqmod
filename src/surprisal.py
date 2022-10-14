from functools import lru_cache
from typing import List, Tuple

import numpy as np
import torch
import transformers

tokenizer = transformers.BertTokenizerFast.from_pretrained(
    "bert-base-chinese", add_prefix_space=True
)
model = transformers.GPT2LMHeadModel.from_pretrained("ckiplab/gpt2-base-chinese")
model.eval()


@lru_cache(None)
def surprisal(words: Tuple[str]) -> List[torch.Tensor]:
    sentence = "".join(words)
    encodings = tokenizer(sentence, return_offsets_mapping=True)
    # tokens = [tokenizer.decode(i).strip() for i in encodings.input_ids][1:-1]  # includes UNKs
    tokens = [
        sentence[offset[0] : offset[1]] for offset in encodings.offset_mapping[1:-1]
    ]

    with torch.no_grad():
        # inputs: (1, sent_len)
        inputs = torch.tensor([encodings.input_ids])
        # outputs: (1, sent_len, vocab_size)
        outputs = model(inputs, labels=inputs)

        shift_inputs = inputs[:, 1:].squeeze()
        shift_outputs = outputs.logits[:, :-1, :].squeeze()
        # log_probs: (sent_len-1,)
        log_probs = torch.nn.functional.cross_entropy(
            shift_outputs, shift_inputs, reduction="none"
        )
        assert torch.isclose(
            torch.exp(sum(log_probs) / len(log_probs)), torch.exp(outputs.loss)
        )

    token_scores = log_probs[:-1]
    assert len(tokens) == len(token_scores)
    word_scores = []
    current_word = ""
    current_word_scores = []
    for token, score in zip(tokens, token_scores):
        current_word += token
        current_word_scores.append(score)
        expected_word = words[len(word_scores)]
        if current_word == expected_word:
            word_scores.append(np.sum(current_word_scores))
            current_word = ""
            current_word_scores = []
    assert current_word == ""
    assert current_word_scores == []
    assert len(words) == len(word_scores)

    return [
        torch.tensor([score])
        for score in word_scores
    ]
