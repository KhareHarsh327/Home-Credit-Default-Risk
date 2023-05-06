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