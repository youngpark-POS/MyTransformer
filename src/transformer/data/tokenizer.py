"""의존성 없는 정규식 기반 토크나이저.

spaCy 같은 외부 라이브러리 대신, 소문자화 후 단어와 구두점을 분리하는
가벼운 토크나이저를 직접 구현한다. 번역 품질을 극대화하기보다 from-scratch
파이프라인을 명확히 보여주는 데 목적이 있다.
"""
from __future__ import annotations

import re
from typing import List

# 단어(유니코드 문자/숫자 연속) 또는 단일 구두점 하나를 토큰으로 잡는다.
_TOKEN_RE = re.compile(r"\w+|[^\w\s]", re.UNICODE)


def tokenize(text: str) -> List[str]:
    """문자열을 토큰 리스트로 변환한다.

    예: "A man, riding!" -> ["a", "man", ",", "riding", "!"]
    """
    return _TOKEN_RE.findall(text.lower().strip())


def detokenize(tokens: List[str]) -> str:
    """토큰 리스트를 사람이 읽을 수 있는 문자열로 되돌린다(근사 복원).

    구두점 앞 공백을 제거하는 정도의 후처리만 수행한다.
    """
    text = " ".join(tokens)
    # 단어와 구두점 사이의 불필요한 공백 정리: "man ," -> "man,"
    text = re.sub(r"\s+([^\w\s])", r"\1", text)
    return text.strip()
