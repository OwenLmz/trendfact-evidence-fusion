from __future__ import annotations

import re


TOKEN_RE = re.compile(r"[\u3400-\u9fffA-Za-z0-9]+")


def tokenize_character_bigram(text: str) -> list[str]:
    tokens: list[str] = []
    for segment in TOKEN_RE.findall(text):
        if len(segment) == 1:
            tokens.append(segment)
        else:
            tokens.extend(segment[index : index + 2] for index in range(len(segment) - 1))
    return tokens


TOKENIZERS = {"character_bigram": tokenize_character_bigram}
