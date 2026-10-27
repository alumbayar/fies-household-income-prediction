import pandas as pd
import pytest

from fies_income.features import (
    EXPENDITURE_COMPONENTS,
    add_household_features,
    add_income_targets,
    engineer_features,
)


@pytest.fixture
def raw_df():
    df = pd.DataFrame(
        {
            "Total Household Income": [120_000, 600_000, 3_000_000],
            "Household Head Occupation": ["Lawyers", None, "Rice farmers"],
            "Household Head Class of Worker": ["Private", None, "Self-employed"],
            "Toilet Facilities": ["Water-sealed", None, "Water-sealed"],
            "Household Head Highest Grade Completed": [
                "High School Graduate",
                "Post Baccalaureate",
                "Not a real level",
            ],
            "House Floor Area": [40, 90, 200],
            "Total Number of Family members": [4, 3, 5],
            "Number of bedrooms": [2, 0, 4],
        }
    )
    for i, col in enumerate(EXPENDITURE_COMPONENTS):
        df[col] = [i, 2 * i, 3 * i]
    return df


def test_add_income_targets(raw_df):
    out = add_income_targets(raw_df)
    expected_annual = sum(range(len(EXPENDITURE_COMPONENTS)))

    assert "Total Household Income" not in out
    assert out["Monthly Income"].tolist() == [10_000, 50_000, 250_000]
    assert out["Annual Expenditure"].tolist() == [
        expected_annual,
        2 * expected_annual,
        3 * expected_annual,
    ]
    assert out["Monthly Expenditure"].tolist() == pytest.approx(
        [expected_annual / 12, expected_annual / 6, expected_annual / 4]
    )
    assert out["Income Bracket"].astype(str).tolist() == [
        "Poor",
        "Middle Income",
        "Rich",
    ]


def test_add_income_targets_does_not_mutate_input(raw_df):
    before = raw_df.copy()

    add_income_targets(raw_df)

    pd.testing.assert_frame_equal(raw_df, before)


def test_add_household_features(raw_df):
    out = add_household_features(raw_df)

    assert out["Household Head Occupation"][1] == "Unemployed/Not Applicable"
    assert out["Household Head Class of Worker"][1] == "Not Applicable"
    assert out["ISCO Major Group"].tolist() == [
        "2 - Professionals",
        "Unemployed",
        "6 - Skilled Agricultural, Forestry and Fishery Workers",
    ]
    assert out["ISCO Skill Level"].tolist() == [8, -1, 4]
    assert out["Toilet Facilities"].tolist() == ["Water-sealed"] * 3
    assert out["Education Years"].tolist() == [10, 16, 0]
    assert out["Area Per Capita"].tolist() == [10, 30, 40]
    assert out["People Per Bedroom"].tolist() == [2, 3, 1.25]


def test_add_household_features_does_not_mutate_input(raw_df):
    before = raw_df.copy()

    add_household_features(raw_df)

    pd.testing.assert_frame_equal(raw_df, before)


def test_engineer_features_combines_both_steps(raw_df):
    expected = add_household_features(add_income_targets(raw_df))

    pd.testing.assert_frame_equal(engineer_features(raw_df), expected)
