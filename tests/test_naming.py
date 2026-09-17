"""파일명 칸 구조 추정. 추정이지 해독이 아니다."""

import pytest

from ebl_gap.naming import dose_values_for, infer_fields, numeric_field_indexes

#: 사용자의 실제 파일명. dose 5개(140~220)에 반복 촬영이 섞여 있다.
REAL_NAMES = [
    "ARP_70_C_140_001.tif",
    "ARP_70_C_140_002.tif",
    "ARP_70_C_160_001.tif",
    "ARP_70_C_160_002.tif",
    "ARP_70_C_180_001.tif",
    "ARP_70_C_200_001.tif",
    "ARP_70_C_220_001.tif",
]


def test_reads_the_dose_and_replicate_fields_of_the_real_names():
    guess = infer_fields(REAL_NAMES)
    assert guess.dose_index == 3
    assert guess.replicate_index == 4
    assert guess.constant_numeric == {1: 70.0}
    assert guess.field_count == 5


def test_maps_every_name_to_its_dose():
    values = infer_fields(REAL_NAMES).dose_values
    assert values["ARP_70_C_140_001.tif"] == 140.0
    assert values["ARP_70_C_140_002.tif"] == 140.0
    assert values["ARP_70_C_220_001.tif"] == 220.0
    assert len(values) == len(REAL_NAMES)


def test_more_replicates_than_doses_still_reads_the_dose_field():
    """반복이 dose보다 많은 폴더. "서로 다른 값이 가장 많은 칸"만 보면 뒤집힌다.

    dose 2개(140, 160)에 반복 3장씩. 반복 칸이 3개로 더 많으므로, 반복 칸을
    먼저 골라내지 않으면 dose를 1/2/3으로 읽고 곡선의 x축이 통째로 거짓이 된다.
    """
    names = [f"ARP_70_C_{dose}_{rep:03d}.tif"
             for dose in (140, 160) for rep in (1, 2, 3)]
    guess = infer_fields(names)
    assert guess.dose_index == 3
    assert guess.replicate_index == 4
    assert sorted(set(guess.dose_values.values())) == [140.0, 160.0]


def test_gives_up_when_the_field_counts_differ():
    """반쯤 맞는 추정이 제일 위험하다. 칸이 어긋나면 아무 것도 읽지 않는다."""
    guess = infer_fields(["ARP_70_C_140_001.tif", "ARP_70_C_160.tif"])
    assert guess.dose_index is None
    assert guess.replicate_index is None
    assert guess.dose_values == {}
    assert guess.constant_numeric == {}
    assert guess.field_count == 0


def test_a_folder_without_repeats_reads_one_dose_per_file():
    names = [f"ARP_70_C_{dose}.tif" for dose in (140, 160, 180)]
    guess = infer_fields(names)
    assert guess.dose_index == 3
    assert guess.replicate_index is None
    assert guess.dose_values["ARP_70_C_180.tif"] == 180.0


def test_splits_on_dashes_and_spaces_too():
    guess = infer_fields(["ARP-70-C-140.tif", "ARP-70 C-160.tif"])
    assert guess.dose_index == 3
    assert guess.constant_numeric == {1: 70.0}


def test_a_field_that_is_not_numeric_in_every_file_is_not_a_dose():
    guess = infer_fields(["pattern_300uC.tif", "pattern_400uC.tif"])
    assert guess.dose_index is None
    assert guess.dose_values == {}
    assert guess.field_count == 2


def test_no_names_at_all_gives_up():
    guess = infer_fields([])
    assert guess.dose_index is None
    assert guess.field_count == 0


def test_a_single_file_has_nothing_to_compare():
    """칸마다 값이 하나뿐이면 무엇이 변하는 칸인지 알 수 없다."""
    guess = infer_fields(["ARP_70_C_140_001.tif"])
    assert guess.dose_index is None
    assert guess.constant_numeric == {1: 70.0, 3: 140.0, 4: 1.0}


def test_decimal_doses_are_read():
    guess = infer_fields(["a_140.5_001.tif", "a_160.5_001.tif"])
    assert guess.dose_values["a_140.5_001.tif"] == pytest.approx(140.5)


def test_numeric_field_indexes_lists_what_the_user_can_choose():
    """칸 바꾸기 목록. 모든 파일에서 숫자인 칸만 dose가 될 수 있다."""
    assert numeric_field_indexes(REAL_NAMES) == (1, 3, 4)
    assert numeric_field_indexes(["pattern_300uC.tif"]) == ()


def test_dose_values_for_a_chosen_field_overrides_the_guess():
    values = dose_values_for(REAL_NAMES, 1)
    assert set(values.values()) == {70.0}


def test_dose_values_for_a_field_that_is_not_a_number_skips_that_file():
    values = dose_values_for(["a_140.tif", "a_xx.tif"], 1)
    assert values == {"a_140.tif": 140.0}
