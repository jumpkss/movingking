"""파일명 칸 구조에서 dose와 반복 번호를 추정한다.

**추정이지 해독이 아니다.** 여기서 나온 dose는 사용자에게 반드시 보여주고
바꿀 수 있게 해야 한다 — dose를 잘못 읽으면 dose-gap 곡선 전체가 조용히
틀리고, 그 곡선이 사용자가 dose를 고르는 화면이다.

엔진 코드이므로 Qt를 모른다.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import PurePath
from typing import Sequence

#: 칸 구분자. 연속된 구분자는 한 번으로 본다.
_SEPARATORS = re.compile(r"[_\-\s]+")

#: 칸 전체가 숫자여야 후보가 된다. `300uC`는 숫자 칸이 아니다.
_NUMBER = re.compile(r"\d+(?:\.\d+)?\Z")


@dataclass(frozen=True)
class NamingGuess:
    """폴더 전체의 파일명에서 읽어낸 칸 구조.

    `field_count`가 0이면 추정을 포기한 것이다(칸 수가 어긋나거나 파일이 없다).
    """

    dose_index: int | None
    replicate_index: int | None
    #: 모든 파일에서 같은 값인 숫자 칸 -> 값. 목표 갭 후보다. **참고용일 뿐이고
    #: 측정값과 비교하거나 판정에 쓰지 않는다** — 파일명은 의도이지 계측값이 아니다.
    constant_numeric: dict[int, float]
    #: 파일명 -> dose. `dose_index`가 None이면 비어 있다.
    dose_values: dict[str, float]
    field_count: int


def split_fields(name: str) -> list[str]:
    """확장자를 떼고 구분자로 쪼갠다."""
    return _SEPARATORS.split(PurePath(name).stem)


def _is_number(field: str) -> bool:
    return _NUMBER.match(field) is not None


def _field_table(names: Sequence[str]) -> list[list[str]] | None:
    """칸 수가 모두 같을 때만 칸 표를 돌려준다.

    칸 수가 다른 파일이 섞이면 None이다. 반쯤 맞는 추정이 제일 위험하다 —
    3번 칸이 어떤 파일에서는 dose이고 다른 파일에서는 반복 번호이면, 곡선은
    아무 표시 없이 뒤섞인 x축을 그린다.
    """
    if not names:
        return None
    table = [split_fields(name) for name in names]
    if len({len(fields) for fields in table}) != 1:
        return None
    return table


def _column(table: list[list[str]], index: int) -> list[str]:
    return [fields[index] for fields in table]


def _numeric_columns(table: list[list[str]]) -> list[int]:
    return [index for index in range(len(table[0]))
            if all(_is_number(value) for value in _column(table, index))]


def _looks_like_a_run_from_one(values: set[float]) -> bool:
    """1부터 빠짐없이 이어지는 정수인가. `001, 002`가 그렇다.

    같은 칸 수의 dose 격자에서 dose 칸과 반복 칸은 서로 구별되지 않는다 —
    파일 이름이 서로 다른 이상 어느 칸이든 "나머지 칸으로 묶으면 값이 겹치지
    않는다"가 저절로 성립하기 때문이다. 실제로 남는 신호는 반복 번호가
    1부터 세어 올라간다는 것뿐이다. dose가 140, 160 ... 으로 매겨진 폴더에서
    이 신호는 반복 칸에만 걸린다.
    """
    return values == {float(i) for i in range(1, len(values) + 1)}


def numeric_field_indexes(names: Sequence[str]) -> tuple[int, ...]:
    """모든 파일에서 숫자인 칸 번호. 사용자가 dose 칸으로 고를 수 있는 목록이다."""
    table = _field_table(names)
    if table is None:
        return ()
    return tuple(_numeric_columns(table))


def dose_values_for(names: Sequence[str], index: int) -> dict[str, float]:
    """지정한 칸의 숫자를 dose로 읽는다. 숫자가 아닌 파일은 뺀다.

    사용자가 칸을 직접 고르는 경로다. 추정이 틀렸을 때 바로잡는 유일한 수단이다.
    """
    values: dict[str, float] = {}
    for name in names:
        fields = split_fields(name)
        if 0 <= index < len(fields) and _is_number(fields[index]):
            values[name] = float(fields[index])
    return values


def infer_fields(names: Sequence[str]) -> NamingGuess:
    """파일명들을 구분자로 쪼개 dose 칸과 반복 번호 칸을 추정한다.

    추정이지 해독이 아니다. 반드시 사용자에게 보여주고 확인받아야 한다 —
    dose를 잘못 읽으면 dose-gap 곡선 전체가 조용히 틀린다.
    """
    table = _field_table(names)
    if table is None:
        return NamingGuess(dose_index=None, replicate_index=None,
                           constant_numeric={}, dose_values={}, field_count=0)

    field_count = len(table[0])
    numeric = _numeric_columns(table)
    distinct = {index: {float(v) for v in _column(table, index)}
                for index in numeric}

    constant_numeric = {index: next(iter(values))
                        for index, values in distinct.items()
                        if len(values) == 1}
    candidates = [index for index in numeric if len(distinct[index]) > 1]

    replicate_index: int | None = None
    if len(candidates) > 1:
        # 반복 번호 후보: 1부터 이어지는 번호. 여러 개면 값이 가장 적은 칸,
        # 그래도 같으면 뒤쪽 칸을 고른다(반복 번호는 이름 끝에 붙는다).
        # dose 칸을 남겨야 하므로 후보가 둘 이상일 때만 하나를 뺀다.
        runs = [index for index in candidates
                if _looks_like_a_run_from_one(distinct[index])]
        if runs:
            replicate_index = min(runs, key=lambda i: (len(distinct[i]), -i))

    dose_pool = [index for index in candidates if index != replicate_index]
    if not dose_pool:
        return NamingGuess(dose_index=None, replicate_index=None,
                           constant_numeric=constant_numeric, dose_values={},
                           field_count=field_count)

    # dose 칸: 남은 후보 중 서로 다른 값이 가장 많은 칸.
    dose_index = max(dose_pool, key=lambda i: (len(distinct[i]), -i))
    dose_values = {name: float(fields[dose_index])
                   for name, fields in zip(names, table)}
    return NamingGuess(dose_index=dose_index, replicate_index=replicate_index,
                       constant_numeric=constant_numeric,
                       dose_values=dose_values, field_count=field_count)
