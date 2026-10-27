import pandas as pd

from fies_income.occupations import ISCO_MAPPING, ISCO_SKILL_LEVEL, UNEMPLOYED, map_isco


def test_known_titles_map_to_group_and_skill():
    occupation = pd.Series(["Lawyers", "Rice farmers", "Combat soldiers"])

    group, skill = map_isco(occupation)

    assert group.tolist() == [
        "2 - Professionals",
        "6 - Skilled Agricultural, Forestry and Fishery Workers",
        "0 - Armed Forces Occupations",
    ]
    assert skill.tolist() == [8, 4, 0]


def test_unknown_and_missing_titles_are_unemployed():
    occupation = pd.Series(["Unemployed/Not Applicable", None])

    group, skill = map_isco(occupation)

    assert group.tolist() == [UNEMPLOYED, UNEMPLOYED]
    assert skill.tolist() == [-1, -1]
    assert skill.dtype == int


def test_every_group_has_a_skill_level():
    assert set(ISCO_MAPPING) <= set(ISCO_SKILL_LEVEL)


def test_job_titles_are_unique_across_groups():
    titles = [job for jobs in ISCO_MAPPING.values() for job in jobs]

    assert len(titles) == len(set(titles))
