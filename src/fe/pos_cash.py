import numpy as np
import pandas as pd



def get_tolerance_days(
    days_with_tol: pd.Series,
    days_without_tol: pd.Series
)->pd.Series:
    """
    Description:
        A method to compute the tolerance days for the credit
        by taking the day of application for the current credit
        as the reference point.
    Args:
        * days_with_tol     : A Pandas Series containing the 
          number of DPD for the credit under normal circumstances.
        * days_without_tol  : A Pandas Series containing the
          'reduced' number of DPD for the credit due to special
          considerations by the credit authority.
    Returns:
        * tolerance_days    : A Pandas Series containing the 
          number of days reduced as a result of special consideration.
    """
    tolerance_days = days_with_tol - days_without_tol
    return tolerance_days



def is_present(
    data: pd.DataFrame,
    column: str,
    values
)->pd.DataFrame:
    """
    Description:
        A method to return whether the mentioned values are
        present in the series or not.
    Args:
        * data      : A Pandas DataFrame containing The entire data.
        * column    : A String bearing the name of the column to be inspected.
        * values    : An iterable containing the values to be checked 
                      for presence.
    Returns:
        * new_df    : A Pandas DataFrame containing the binary outcome of
                      whether the current value is present in the aggregated
                      data or not.
    """
    # Pre-select the hash-value features:
    new_df = data.iloc[:, :2]

    # Feature Generation:
    for value in values:
        label = value.split()                           # For using join method
        label = "FLAG_POS_" + ("_".join(label).upper()) # Using the join method
        new_df[label] = ( data[column] == value )
    
    # Feature Aggregation:
    new_df = new_df.groupby(by = "SK_ID_PREV").sum()

    return new_df



def get_features(df: pd.DataFrame)->pd.DataFrame:
    """
    Description:
        A method to get the new feature space from the POS_CASH dataframe.
    Args:
        * df    : A Pandas DataFrame containing the POS_CASH data.
    Returns:
        * new_fs: A Pandas DataFrame generated as a result of feature generation
                  and selection process.
    """
    # Calculating the tolerance days:
    df["POS_DAYS_TOLERANCE"] = get_tolerance_days(
        days_with_tol = df["SK_DPD_DEF"],
        days_without_tol = df["SK_DPD"]
    )

    # Getting the feature presence data for NAME_CONTRACT_STATUS:
    new_df = is_present(
        data = df,
        column = "NAME_CONTRACT_STATUS",
        values = [
            "Canceled", "Approved", "Completed",
            "Amortized debt", "Returned to the store"
        ]
    )

    # Dropping the categorical variable after segmenting:
    df = df.drop(labels=["NAME_CONTRACT_STATUS"], axis=1)

    # Feature Aggregation:
    new_fs = df.groupby(
        by = "SK_ID_PREV"
    ).median().join(
        other = new_df,
        on = "SK_ID_PREV",
        how = "left",
        rsuffix = "_R"
    ).drop(
        labels=[
            "SK_ID_CURR_R", "MONTHS_BALANCE",
            "CNT_INSTALMENT", "CNT_INSTALMENT_FUTURE",
            "SK_DPD","SK_DPD_DEF"
        ],
        axis=1
    )

    return new_fs