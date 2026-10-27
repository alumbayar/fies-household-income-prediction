import pandas as pd

from fies_income.occupations import map_isco

EXPENDITURE_COMPONENTS = [
    "Total Food Expenditure",
    "Alcoholic Beverages Expenditure",
    "Restaurant and hotels Expenditure",
    "Tobacco Expenditure",
    "Clothing, Footwear and Other Wear Expenditure",
    "Housing and water Expenditure",
    "Medical Care Expenditure",
    "Transportation Expenditure",
    "Communication Expenditure",
    "Education Expenditure",
    "Miscellaneous Goods and Services Expenditure",
    "Special Occasions Expenditure",
    "Crop Farming and Gardening expenses",
    "Imputed House Rental Value",
]

# Monthly income thresholds in PHP, rounded from the PIDS income class definitions.
BRACKET_BINS = [0, 11_000, 22_000, 44_000, 77_000, 131_000, 219_000, 9_999_999]
BRACKET_LABELS = [
    "Poor",
    "Low Income",
    "Lower Middle",
    "Middle Income",
    "Upper Middle",
    "Upper Income",
    "Rich",
]

# Approximate years of formal schooling implied by each attainment level.
YEARS_OF_SCHOOLING = {
    "No Grade Completed": 0,
    "Preschool": 0,
    "Grade 1": 1,
    "Grade 2": 2,
    "Grade 3": 3,
    "Grade 4": 4,
    "Grade 5": 5,
    "Grade 6": 6,
    "Elementary Graduate": 6,
    "First Year High School": 7,
    "Second Year High School": 8,
    "Third Year High School": 9,
    "High School Graduate": 10,
    "First Year Post Secondary": 11,
    "Second Year Post Secondary": 12,
    "Other Programs in Education at the Third Level, First Stage, of the Type that Leads to an Award not Equivalent to a First University or Baccalaureate Degree": 12,
    "First Year College": 11,
    "Second Year College": 12,
    "Third Year College": 13,
    "Fourth Year College": 14,
    "Other Programs of Education at the Third Level, First Stage, of the Type that Leads to a Baccalaureate or First University/Professional Degree (HIgher Education Level, First Stage, or Collegiate Education Level)": 14,
    "Agriculture, Forestry, and Fishery Programs": 14,
    "Architecture and Building Programs": 14,
    "Arts Programs": 14,
    "Basic Programs": 14,
    "Business and Administration Programs": 14,
    "Computing/Information Technology Programs": 14,
    "Engineering and Engineering Trades Programs": 15,
    "Engineering and Engineering trades Programs": 15,
    "Environmental Protection Programs": 14,
    "Health Programs": 14,
    "Humanities Programs": 14,
    "Journalism and Information Programs": 14,
    "Law Programs": 14,
    "Life Sciences Programs": 14,
    "Manufacturing and Processing Programs": 14,
    "Mathematics and Statistics Programs": 14,
    "Personal Services Programs": 14,
    "Physical Sciences Programs": 14,
    "Security Services Programs": 14,
    "Social and Behavioral Science Programs": 14,
    "Social Services Programs": 14,
    "Teacher Training and Education Sciences Programs": 14,
    "Transport Services Programs": 14,
    "Veterinary Programs": 14,
    "Post Baccalaureate": 16,
}


def add_income_targets(df: pd.DataFrame) -> pd.DataFrame:
    """Derive monthly income, expenditure totals and the income bracket.

    Args:
        df: Raw FIES records containing ``Total Household Income`` and every
            column listed in ``EXPENDITURE_COMPONENTS``.

    Returns:
        A new DataFrame with ``Total Household Income`` renamed to
        ``Annual Household Income`` and the columns ``Monthly Income``,
        ``Annual Expenditure``, ``Monthly Expenditure`` and ``Income Bracket``
        appended.
    """
    out = df.rename(columns={"Total Household Income": "Annual Household Income"})
    out["Monthly Income"] = out["Annual Household Income"] / 12
    out["Annual Expenditure"] = out[EXPENDITURE_COMPONENTS].sum(axis=1)
    out["Monthly Expenditure"] = out["Annual Expenditure"] / 12
    out["Income Bracket"] = pd.cut(
        out["Monthly Income"], bins=BRACKET_BINS, labels=BRACKET_LABELS
    )
    return out


def add_household_features(df: pd.DataFrame) -> pd.DataFrame:
    """Impute household head fields and add occupation, education and housing features.

    Missing occupations belong to households whose head is not working, so
    they are filled with explicit placeholders rather than a statistical
    imputation.

    Args:
        df: FIES records containing household head, toilet facility, floor
            area, family size and bedroom columns.

    Returns:
        A new DataFrame with missing values filled and the columns
        ``ISCO Major Group``, ``ISCO Skill Level``, ``Education Years``,
        ``Area Per Capita`` and ``People Per Bedroom`` appended.
    """
    out = df.copy()
    out["Household Head Occupation"] = out["Household Head Occupation"].fillna(
        "Unemployed/Not Applicable"
    )
    out["Household Head Class of Worker"] = out[
        "Household Head Class of Worker"
    ].fillna("Not Applicable")
    out["ISCO Major Group"], out["ISCO Skill Level"] = map_isco(
        out["Household Head Occupation"]
    )

    out["Toilet Facilities"] = out["Toilet Facilities"].fillna(
        out["Toilet Facilities"].mode()[0]
    )

    out["Education Years"] = (
        out["Household Head Highest Grade Completed"].map(YEARS_OF_SCHOOLING).fillna(0)
    )

    members = out["Total Number of Family members"]
    out["Area Per Capita"] = out["House Floor Area"] / members
    # Studio-type dwellings report zero bedrooms; count them as one room.
    out["People Per Bedroom"] = members / out["Number of bedrooms"].replace(0, 1)
    return out


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Apply the full feature engineering sequence to raw FIES records.

    Args:
        df: Raw FIES records as loaded from the Kaggle CSV.

    Returns:
        A new DataFrame with targets and derived household features.
    """
    return add_household_features(add_income_targets(df))
