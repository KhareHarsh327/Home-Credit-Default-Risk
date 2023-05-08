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