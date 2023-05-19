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
    credit_util_ratio: pd.Series = total_util / credit_limit
    return credit_util_ratio



def get_avg_drawn(
    amt_drawn: pd.Series,
    cnt_drawn: pd.Series
)->pd.Series:
    """
    Description:
        A method to get the average withdrawing on a credit per transaction for each month.
    Args:
        * amt_drawn     : A Pandas Series containing the total amount drawn .
        * cnt_drawn     : A Pandas Series containing the total number of 
                          transactions/drawings in the month.
    Results:
        * avg_drawn     : A Pandas Series containing the interest rate for the month.
    """
    avg_drawn: pd.Series = amt_drawn / cnt_drawn
    return avg_drawn



def get_surcharge_ratio(
    amt_without_surcharge: pd.Series,
    amt_with_surcharge: pd.Series
)->pd.Series:
    """
    Description:
        A method to get the monthly surcharge (other than the interest).
        * We assume that the surcharge shall have beeen levied only if 
          the client had not made the payments on time.
    Args:
        * amt_without_surcharge : A Pandas Series containing the total amount 
                                  without surcharge.
        * amt_with_surcharge    : A Pandas Series containing the total amount
                                  with surcharge.
    Results:
        * surcharge     : A Pandas Series containing the surcharge for the month.
    """
    surcharge: pd.Series = (amt_with_surcharge - amt_without_surcharge)/amt_without_surcharge
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
    tolerance_days: pd.Series = days_with_tol - days_without_tol
    return tolerance_days



def get_unpaid_ratio(
    paid: pd.Series,
    owed: pd.Series
)->pd.Series:
    """
    Description:
        A method to get the ratio between the total 'unpaid' credit
        to the minimum payment for the month.
        * Since only the unpaid amounts are to be considered, we set the value of 
          unpaid amount to 0 for the payments where the paid amount > owed amount.
        * This shall differentiate the unpaid amounts from the over-paid amounts.
    Args:
        * paid      : A Pandas Series containing the total amount paid during the month.
        * owed      : A Pandas Series containing the credit limit.
    Results:
        * pay_ratio : A Pandas Series containing the unpaid ratio.
    """
    # Differentiating between unpaid and over-paid amounts:
    diff = owed - paid
    unpaid = pd.Series([max(0, i) for i in diff])

    unpaid_ratio: pd.Series = unpaid / owed
    return unpaid_ratio



def get_cnt_defaults(
    data: pd.DataFrame
)->pd.Series:
    """
    Description:
        A method to get the number of defaults (DPD > 90) on the credit card
        despite considering the tolerance days.
    Args:
        * data          : A Pandas DataFrame containing the entire credit card data.
    Results:
        * cnt_defaults  : A Pandas Series containing the number of defaults.
    """
    # Pre-select the hash-value feature for joining:
    cnt_defaults = data.iloc[:,[0]]

    cnt_defaults["CC_CNT_DEFAULTS"] = (data["SK_DPD_DEF"] > 90)
    cnt_defaults = cnt_defaults.groupby(by="SK_ID_PREV").sum()
    return cnt_defaults



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
    # Pre-select the hash-value feature for joining:
    new_df = data.iloc[:,[0]]

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
    df["CC_CREDIT_UTIL_RATIO"] = get_credit_util_ratio(
        total_util = df["AMT_BALANCE"],
        credit_limit = df["AMT_CREDIT_LIMIT_ACTUAL"]
    )

    # Getting the Tolerance Days:
    df["CC_DAYS_TOLERANCE"] = get_tolerance_days(
        days_with_tol = df["SK_DPD_DEF"], 
        days_without_tol = df["SK_DPD"]
    )

    # Getting the Ratio for Unpaid Amount:
    df["CC_UNPAID_RATIO"] = get_unpaid_ratio(
        paid = df["AMT_PAYMENT_TOTAL_CURRENT"],
        owed = df["AMT_INST_MIN_REGULARITY"]
    )

    # Get the Surcharge:
    df["CC_SURCHARGE_RATIO"] = get_surcharge_ratio(
        amt_without_surcharge = df["AMT_RECIVABLE"],
        amt_with_surcharge = df["AMT_TOTAL_RECEIVABLE"]
    )

    # Get the Average Drawings per Transaction:
    df["CC_AVG_DRAWN"] = get_avg_drawn(
        amt_drawn = df["AMT_DRAWINGS_CURRENT"], 
        cnt_drawn = df["CNT_DRAWINGS_CURRENT"]
    )

    # Getting the feature presence data for NAME_CONTRACT_STATUS:
    flag_df = is_present(
        data = df,
        column = "NAME_CONTRACT_STATUS",
        values = ["Completed", "Signed", "Refused", "Approved"]
    )

    # Getting the number of Defaults:
    def_df = get_cnt_defaults(data=df)

    # Feature Selection:
    df = df[[
        "SK_ID_CURR", "SK_ID_PREV", "CC_CREDIT_UTIL_RATIO",
        "CC_DAYS_TOLERANCE", "CC_UNPAID_RATIO", "CC_SURCHARGE_RATIO"
        ]
    ]

    # Feature Aggregation:
    new_fs = df.groupby(
        by = "SK_ID_PREV",
    ).median().join(
        other = def_df,
        on = "SK_ID_PREV",
        how = "left",
        rsuffix = "_R1"
    ).join(
        other = flag_df,
        on = "SK_ID_PREV",
        how = "left",
        rsuffix = "_R2"
    )

    # Imputing NaN values with 0:
    new_fs = new_fs.fillna(0)

    return new_fs