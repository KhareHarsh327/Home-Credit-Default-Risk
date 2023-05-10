import numpy as np
import pandas as pd



def get_credit_util_ratio(
    total_util: pd.Series,
    credit_limit: pd.Series
)->pd.Series:
    """
    Description:
        A method to get the ratio between the total credit utilized
        to the credit limit.
    Args:
        * total_util        : A Pandas Series containing the total credit utilized.
        * credit_limit      : A Pandas Series containing the monthly credit limit.
    Results:
        * credit_util_ratio : A Pandas Series containing the credit utilization ratio.
    """
    credit_util_ratio = total_util / credit_limit
    return credit_util_ratio



def get_interest_rate(
    principal: pd.Series,
    total: pd.Series
)->pd.Series:
    """
    Description:
        A method to get the monthly interest rate charged.
    Args:
        * principal     : A Pandas Series containing the principal amount.
        * total         : A Pandas Series containing the total payable amount.
    Results:
        * int_rate      : A Pandas Series containing the interest rate for the month.
    """
    int_rate = (total - principal)/principal
    return int_rate



def get_surcharge(
    total_amount: pd.Series,
    total_payable: pd.Series
)->pd.Series:
    """
    Description:
        A method to get the monthly surcharge (other than the interest).
    Args:
        * total_amount  : A Pandas Series containing the amount with principal and interest,
                          but without the surcharge.
        * total_payable : A Pandas Series containing the total payable amount with surcharge.
    Results:
        * surcharge     : A Pandas Series containing the surcharge for the month.
    """
    surcharge = (total_payable - total_amount)/total_amount
    return surcharge



def get_tolerance_days(
    days_with_tol: pd.Series,
    days_without_tol: pd.Series
)->pd.Series:
    """
    Description:
        A method to compute the tolerance days for the credit.
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



def get_payment_ratio(
    paid: pd.Series,
    owed: pd.Series
)->pd.Series:
    """
    Description:
        A method to get the ratio between the total credit utilized
        to the credit limit.
    Args:
        * paid      : A Pandas Series containing the total amount paid during the month.
        * owed      : A Pandas Series containing the credit limit.
    Results:
        * pay_ratio : A Pandas Series containing the _ratio.
    """
    pay_ratio = paid / owed
    return pay_ratio



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
        label = value.split()
        label = "FLAG_CC_" + ("_".join(label).upper())
        new_df[label] = ( data[column] == value )
    
    # Feature Aggregation:
    new_df = new_df.groupby(by = "SK_ID_PREV").sum()

    return new_df



def get_features(df: pd.DataFrame)->pd.DataFrame:
    """
    Description:
        A method to get the new feature space from the credit cards payments dataframe.
    Args:
        * df    : A Pandas DataFrame containing the credit card payments data.
    Returns:
        * new_fs: A Pandas DataFrame generated as a result of feature generation
                  and selection process.
    """
    # Getting the Credit Utilization Ratio:
    df["CREDIT_UTIL_RATIO"] = get_credit_util_ratio(
        total_util = df["AMT_DRAWINGS_CURRENT"],
        credit_limit = df["AMT_CREDIT_LIMIT_ACTUAL"]
    )

    # Getting the Tolerance Days:
    df["CC_DAYS_TOLERANCE"] = get_tolerance_days(
        days_with_tol = df["SK_DPD_DEF"], 
        days_without_tol = df["SK_DPD"]
    )

    # Getting the Payment Ratio:
    df["CC_PAY_RATIO"] = get_payment_ratio(
        paid = df["AMT_PAYMENT_TOTAL_CURRENT"],
        owed = df["AMT_INST_MIN_REGULARITY"]
    )

    # Get the Interest Rate:
    df["CC_INTEREST_RATE"] = get_interest_rate(
        principal = df["AMT_RECEIVABLE_PRINCIPAL"], 
        total = df["AMT_RECIVABLE"]
    )

    # Get the Surcharge:
    df["CC_SURCHARGE"] = get_surcharge(
        total_amount = df["AMT_RECIVABLE"],
        total_payable = df["AMT_TOTAL_RECEIVABLE"]
    )

    # Getting the feature presence data for NAME_CONTRACT_STATUS:
    new_df = is_present(
        data = df,
        column = "NAME_CONTRACT_STATUS",
        values = ["Completed", "Signed", "Refused", "Approved"]
    )

    # Feature Selection:
    df = df[[
        "SK_ID_CURR", "SK_ID_PREV", "CREDIT_UTIL_RATIO",
        "CC_DAYS_TOLERANCE", "CC_PAY_RATIO",
        "CC_INTEREST_RATE", "CC_SURCHARGE"
        ]
    ]

    # Feature Aggregation:
    new_fs = df.groupby(
        by = "SK_ID_PREV",
    ).median().join(
        other = new_df,
        on = "SK_ID_PREV",
        how = "inner",
        rsuffix = "_R"
    ).drop("SK_ID_CURR_R",axis=1)

    return new_fs