from functools import lru_cache
from typing import Callable, List, Tuple

import spacy
import torch

_nlp = spacy.load("zh_core_web_trf")
_nlp_chartok = spacy.load("zh_core_web_trf")

_char_freq = {}
with open("../data/SUBTLEX-CH-CHR", encoding="gb18030") as f:
    for _ in range(3):
        f.readline()
    for row in f:
        char, _, _, logfreq, *_ = row.split("\t")
        _char_freq[char] = float(logfreq)

_word_freq = {}
with open("../data/SUBTLEX-CH-WF", encoding="gb18030") as f:
    for _ in range(3):
        f.readline()
    for row in f:
        word, _, _, logfreq, *_ = row.split("\t")
        _word_freq[word] = float(logfreq)


def _char_tokenizer(sentence: str) -> spacy.tokens.Doc:
    return spacy.tokens.Doc(
        _nlp.vocab, words=list(sentence), spaces=[False] * len(sentence)
    )


_nlp_chartok.tokenizer = _char_tokenizer


@lru_cache(None)
def _parse(
    tokens: Tuple[str],
) -> List[Tuple[List[spacy.tokens.Token], List[spacy.tokens.Token]]]:
    sentence = "".join(tokens)
    toklvl_tokens = list(_nlp(sentence))
    charlvl_tokens = list(_nlp_chartok(sentence))
    toklvl_index = 0
    charlvl_index = 0
    token_start = 0
    result = []

    for token in tokens:
        token_end = token_start + len(token)

        ts = []
        del_until = 0
        for i, toklvl_token in enumerate(toklvl_tokens):
            if toklvl_token.idx < token_end:
                ts.append(toklvl_token)
                if toklvl_token.idx + len(toklvl_token) <= token_end:
                    del_until = i + 1
            else:
                break
        toklvl_tokens = toklvl_tokens[del_until:]

        cs = []
        del_until = 0
        for i, charlvl_token in enumerate(charlvl_tokens):
            if charlvl_token.idx < token_end:
                cs.append(charlvl_token)
                if charlvl_token.idx + len(charlvl_token) <= token_end:
                    del_until = i + 1
            else:
                break
        charlvl_tokens = charlvl_tokens[del_until:]

        result.append((ts, cs))
        token_start += len(token)

    assert toklvl_tokens == [], toklvl_tokens
    assert charlvl_tokens == [], charlvl_tokens
    return result


def _retokenize_labels(
    tokens: Tuple[str],
    doc: spacy.tokens.Doc,
    label: Callable[[spacy.tokens.Token], int],
) -> List[int]:
    """
    Given a list of pre-tokenized sentence and a corresponding spaCy Doc, return a list
    of labels (e.g. POS or dependency), such that they match the tokenization of the
    pre-tokenized sentence. This is necessary because our and spaCy's tokenizations
    don't always agree.
    """
    all_labels = []
    current_token_labels = []
    for token, (toklvl_tokens, charlvl_tokens) in zip(tokens, doc):
        # spaCy agrees with our tokenization
        if len(toklvl_tokens) == 1 and len(toklvl_tokens[0]) == len(token):
            p = label(toklvl_tokens[0])

        # spaCy doesn't agree with our tokenization
        else:
            toklvl_labels = [label(t) for t in toklvl_tokens]
            charlvl_labels = [label(t) for t in charlvl_tokens]
            # spaCy assigns the same label to all overlapping tokens
            if len(set(toklvl_labels)) == 1:
                p = toklvl_labels[0]

            # spaCy's character-level labels are consistent
            elif len(set(charlvl_tokens)) == 1:
                p = charlvl_labels[0]

            # spaCy's labels are inconsistent on all levels
            else:
                p = toklvl_labels[0]

        all_labels.append(p)
    return all_labels


# https://universaldependencies.org/u/pos/
POS = {
    "ADJ": "ADJ",
    "ADP": "OTHER",
    "ADV": "ADV",
    "AUX": "OTHER",
    "CCONJ": "OTHER",
    "DET": "OTHER",
    "INTJ": "OTHER",
    "NOUN": "NOUN",
    "NUM": "OTHER",
    "PART": "OTHER",
    "PRON": "OTHER",
    "PROPN": "NOUN",
    "PUNCT": "OTHER",
    "SCONJ": "OTHER",
    "SYM": "OTHER",
    "VERB": "VERB",
    "X": "OTHER",
}
COARSE_POS = list(set(POS.values()))

# https://spacy.io/models/zh
DEP = {
    "ROOT": "ROOT",
    "acl": "mod",
    "advcl:loc": "mod",
    "advmod": "mod",
    "advmod:dvp": "mod",
    "advmod:loc": "mod",
    "advmod:rcomp": "mod",
    "amod": "mod",
    "amod:ordmod": "mod",
    "appos": "other",
    "aux:asp": "other",
    "aux:ba": "other",
    "aux:modal": "other",
    "aux:prtmod": "other",
    "auxpass": "other",
    "case": "other",
    "cc": "other",
    "ccomp": "other",
    "compound:nn": "compound",
    "compound:vc": "compound",
    "conj": "other",
    "cop": "other",
    "dep": "other",
    "det": "other",
    "discourse": "other",
    "dobj": "dobj",
    "etc": "etc",
    "mark": "other",
    "mark:clf": "other",
    "name": "name",
    "neg": "other",
    "nmod": "mod",
    "nmod:assmod": "mod",
    "nmod:poss": "mod",
    "nmod:prep": "mod",
    "nmod:range": "mod",
    "nmod:tmod": "mod",
    "nmod:topic": "mod",
    "nsubj": "nsubj",
    "nsubj:xsubj": "nsubj",
    "nsubjpass": "nsubj",
    "nummod": "mod",
    "parataxis:prnmod": "parataxis",
    "punct": "punct",
    "xcomp": "other",
}
COARSE_DEP = list(set(DEP.values()))


def pos(tokens: Tuple[str]) -> List[torch.Tensor]:
    doc = _parse(tokens)
    labels = _retokenize_labels(tokens, doc, lambda token: token.pos_)
    return [
        torch.nn.functional.one_hot(torch.tensor(COARSE_POS.index(POS[label])), len(COARSE_POS))
        for label in labels
    ]


def dep(tokens: Tuple[str]) -> List[torch.Tensor]:
    doc = _parse(tokens)
    labels = _retokenize_labels(tokens, doc, lambda token: token.dep_)
    return [
        torch.nn.functional.one_hot(torch.tensor(COARSE_DEP.index(DEP[label])), len(COARSE_DEP))
        for label in labels
    ]


def _depth(token: spacy.tokens.Token) -> int:
    if token.dep_ == "ROOT":
        return 0
    return 1 + _depth(token.head)


def depth(tokens: Tuple[str]) -> List[torch.Tensor]:
    doc = _parse(tokens)
    depths = _retokenize_labels(tokens, doc, _depth)
    return [torch.tensor([depth]) for depth in depths]


def character_frequency(tokens: Tuple[str]) -> List[torch.Tensor]:
    freqs = []
    for token in tokens:
        f = [_char_freq[char] for char in token]
        # Use minimum, maximum, and mean character frequency for each token
        freqs.append(
            [
                min(f),
                max(f),
                sum(f) / len(f),  # mean
            ]
        )
    return [torch.tensor(f) for f in freqs]


def word_frequency(tokens: Tuple[str]) -> List[torch.Tensor]:
    freqs = []
    for token in tokens:
        if token in _word_freq:
            freqs.append(_word_freq[token])
        else:
            # TODO: Better way to handle unknown words
            freqs.append(sum(_word_freq.values()) / len(_word_freq))
    return [torch.tensor([freq]) for freq in freqs]
