import os
import numpy as np
import pandas as pd

ROOT_DIR: str = os.path.join("C:\\Users","KIIT","Desktop","Home Credit Default Risk")



def get_description(df: pd.DataFrame, key: str, value: str)->None:
    """
    Description:
        A method to display the column values for a given condition 
        expressed in terms of a key-value pair.
    Args:
        * df    : a Pandas DataFrame from which the results are to be extracted.
        * key   : a String bearing the name of the attribute to be used for filtering.
        * value : a String bearing the value of the aforesaid key for filtering.
    Returns:
        * None.
    """

    print("ATTRIBUTE NAME: ", value)
    for desc in df[df[key]==value]["Description"].unique():
        print("\n-> ",desc)
    
    return None



def get_missing_values(df: pd.DataFrame)->pd.DataFrame:
    """
    Description:
        A method to display the number and % of values missing from the dataframe.
    Args:
        * df    : A Pandas DataFrame from which the results are to be extracted.
    Returns:
        * table : A Pandas DataFrame containing the missing values expressed as absolute 
          numbers and corresponding percentage.
    """

    # Creating the skeleton for the DataFrame
    table = pd.DataFrame(
        columns=["Column","Missing Values", "Missing Values by %"]
    )

    TOTAL: int = len(df.index)                          # Total number of rows

    for col in df.columns:
        NULL_VAL: int = df[col].isna().sum()            # Number of NaN values
        temp_list: list = [                             # Creating a row to be appended
            col, NULL_VAL,
            round(NULL_VAL/TOTAL,5)*100
        ]
        table.loc[len(table)] = temp_list               # Adding the row to the DataFrame

    return table



def desc_num_var(series: pd.Series):
    """
    Description:
        A method to describe a 'numeric' variable without visualization.
    Args:
        * series: A Pandas Series containing the information of the variable.
    Returns:
        * None
    """
    desc = series.describe()
    desc["IQR"] = abs(desc["75%"] - desc["25%"])
    
    # Deriving the range of the Whiskers in the Box Plot
    max_limit = min(desc["max"], desc["75%"] + 1.5*desc["IQR"])
    min_limit = max(desc["min"], desc["25%"] - 1.5*desc["IQR"])

    # Getting the number and percentage of Outlier Values using IQR method:
    num_outliers = len(series[(series<min_limit)|(series>max_limit)])
    perc_outliers = num_outliers/len(series)*100

    print("\t\tDESCRIPTION OF THE COLUMN")
    temp_df = pd.Series({           
        "Minimum": desc["min"],
        "Q1 (25%)": desc["25%"],
        "Q2 (Median)": desc["50%"],
        "Q3 (75%)": desc["75%"],
        "Maximum": desc["max"],
        "IQR Magnitude": desc["IQR"],
        "Lower Limit of Whisker": min_limit,
        "Upper Limit of Whisker": max_limit,
        "Number of Outliers": num_outliers,
        "Percentage of Outliers": perc_outliers
    })
    print(temp_df)
    
    return None



def desc_cat_var(series: pd.Series):
    """
    Description:
        A method to describe a 'categorical' variable without visualization.
    Args:
        * series: A Pandas Series containing the information of the variable.
    Returns:
        * None
    """
    desc = series.value_counts()

    print("\t\tDISTRIBUTION OF THE COLUMN")
    temp_df = pd.DataFrame({
        "Categories": desc.index,
        "Number of Values": desc.values,
        "Percentage of Values": (desc.values/len(series))*100
    })
    print(temp_df)
    
    print("\nNumber of Categories\t: ", len(desc))
    print("\nMost Frequent Category:")
    print(temp_df.iloc[[0], :])
    print("\nLeast Frequent Category:")
    print(temp_df.iloc[[-1], :])
    
    return None