import numpy as np
import pandas as pd



def get_pay_ratio(
    amt_payable: pd.Series,
    amt_paid: pd.Series
)->pd.Series:
    """
    Description:
        A method to compute the ratio of amount actually paid as installment
        and the amount that was supposed to be paid.
    Args:
        * amt_payable   : A Pandas Series containing the list of payments (as numbers) 
                          'supposed to be made' as installment.
        * amt_paid      : A Pandas Series containing the list of payments (as numbers) 
                          'actually made' as installments.
    Returns:
        * pay_ratio     : A Pandas Series containing the list of ratios of payments made
                          to the payments supposed to be made for an installment.
    """
    pay_ratio: pd.Series = amt_paid/amt_payable
    return pay_ratio



def get_delay_days(
    pay_day: pd.Series,
    due_day: pd.Series
)->pd.Series:
    """
    Description:
        A method to compute the time delay in paying an installment
        in terms of the number of days.
    Args:
        * pay_day   : A Pandas Series containing the list of number of days
                      (relative to the day of application) when the installment 
                      was 'actually paid'.
        * due_day   : A Pandas Series containing the list of number of days
                      (relative to the day of application) when the installment 
                      was 'supposed to be paid'.
    Returns:
        * delay_days: A Pandas Series containing the delay in paying the installments
                      in terms of number of days.
    """
    delay_days: pd.Series = pay_day - due_day
    return delay_days



def get_features(df: pd.DataFrame)->pd.DataFrame:
    """
    Description:
        A method to get the new feature space from the installments dataframe.
    Args:
        * df    : A Pandas DataFrame containing the installments payments data.
    Returns:
        * new_fs: A Pandas DataFrame generated as a result of feature generation
                  and selection process.
    """
    # Computing the payment ratio for each installment:
    df["PAY_RATIO"] = get_pay_ratio(
        amt_payable = df["AMT_INSTALMENT"],
        amt_paid = df["AMT_PAYMENT"]
    )

    # Computing the days delayed for every installment:
    df["DAYS_DELAYED"] = get_delay_days(
        pay_day = df["DAYS_ENTRY_PAYMENT"],
        due_day = df["DAYS_INSTALMENT"]
    )

    # Feature Selection:
    new_fs = df[
        ["SK_ID_PREV", "SK_ID_CURR", "PAY_RATIO", "DAYS_DELAYED"]
    ]

    # Feature Aggregation:
    new_fs = new_fs.groupby(
        by = "SK_ID_PREV"       # Aggregation to obtain data specific to previous credit
    ).median()                  # We choose median over mean due to the high skewness

    # Type conversion:
    new_fs["SK_ID_CURR"] = new_fs["SK_ID_CURR"].astype("int64")

    return new_fs
