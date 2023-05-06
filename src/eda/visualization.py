import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt



def bar_plot(
    category,
    frequency,
    axes, 
    title: str,
    xlabel: str,
    ylabel: str
)->None:
    """
    Description:
        A method to display the horizontal bar plot for a 
        categorical variable's distribution.
    Args:
        * category  : An iterable containing the name of all the categories.
        * frequency : An iterable containing the frequency of all the categories.
        * axes      : A matplotlib.axes.Axes object for the plotting area.
        * title     : A String bearing the name for the plotting area.
    Returns:
        * None
    """
    # Plot settings:
    axes.set_title(title)
    axes.set_xlabel(xlabel)
    axes.set_ylabel(ylabel)
    
    # Plotting the bar plot:
    plot = sns.barplot(
        x = frequency,
        y = category,
        orient="h",
        ax = axes
    )
    # Annotating the plots of different categories:
    for i in range(len(category)):
        plot.annotate(text=str(frequency[i]),xy=(min(frequency),i))

    return None



def pie_plot(
    category,
    frequency, 
    axes: matplotlib.axes.Axes, 
    title: str
)->None:
    """
    Description:
        A method to display the Pie plot for a categorical variable's distribution.
    Args:
        * category  : An iterable containing the name of all the categories.
        * frequency : An iterable containing the frequency of all the categories.
        * axes      : A matplotlib.axes.Axes object for the plotting area.
        * title     : A String bearing the title for the plotting area.
    Returns:
        * None
    """
    # Plot settings:
    axes.set_title(title)

    # Plotting the pie plot:
    axes.pie(
        labels=category, x=frequency,
        autopct="%.2f%%",textprops=None
    )

    return None



def categorical_distribution(
    title: str,
    series: pd.Series
)->None:
    """
    Description:
        A method to plot the distribution of a categorical variable.
    Args:
        * title     : A String bearing the title for the entire plot.
        * series    : A Pandas Series containing the names of categories.
    Returns:
        * None.
    """
    category = series.groupby(by=series).count()

    fig = plt.figure(
        figsize=(14,10),
        frameon=True,
        edgecolor="black",
        linewidth=2
    )
    fig.suptitle(title)
    
    # Adding the Bar Plot:
    bar_plot(
        category = category.keys(),
        frequency = category.values,
        axes = fig.add_subplot(2,2,1),
        title = "Distribution",
        xlabel = "Number of Instances", 
        ylabel = "Name of the Categories"
    )

    # Adding the Pie Plot:
    pie_plot(
        category = category.keys(),
        frequency = category.values,
        axes = fig.add_subplot(2,2,2),
        title = "Distribution in %"
    )
    
    fig.show()
    return None



def numeric_distribution(
    title: str, 
    data,
    num_col: str, 
    cat_col: str = None
)->None:
    """
    Description:
        A method to plot the distribution of a numeric variable.
    Args:
        * title     : A String bearing the title for the entire plot.
        * data      : A Pandas Series/DataFrame bearing the title 
                      for the entire plot.
        * num_col   : A String bearing the name of the numeric column.
        * cat_col   : A String bearing the name of the categorical column 
                      for plotting the numeric variable category-wise;
                      None by default.
    Returns:
        * None.
    """
    fig = plt.figure(
        figsize=(14,10),
        frameon=True,
        edgecolor="black",
        linewidth=2
    )
    fig.suptitle(title)
    if cat_col is not None:
        data[cat_col] = data[cat_col].apply(lambda x: str(x))
    
    # Adding the Box Plot:
    sns.boxplot(
        data = data,
        x = num_col, y = cat_col,
        ax = fig.add_axes([0, 0.45, 1, 0.45])
    )

    # Adding the Distribution Plot:
    sns.histplot(                
        data = data[num_col],
        ax = fig.add_axes([0, 0, 0.45, 0.35])
    )

    # Adding the KDE Plot:
    sns.kdeplot(
        data = data[num_col],
        ax = fig.add_axes([0.55, 0, 0.45, 0.35])
    )

    fig.show()
    return None



def plot_correlation(
    title: str,
    data: pd.DataFrame,
    target: str = None,
    colormap: str = "RdBu"
)->None:
    """
    Description:
        A method to plot the distribution of a numeric variable.
    Args:
        * title     : A String bearing the title for the entire plot.
        * data      : A Pandas DataFrame bearing the title 
                      for the entire plot.
        * target    : A String bearing the name of the target column;
                      None by default.
        * colormap  : A String bearing the name of the colour map 
                      for plotting the correlations.
    Returns:
        * None.
    """
    data = data.select_dtypes(["int64","float64"])      # Choosing numeric columns
    
    if target is not None:
        
        # Handling optional 'target' argument:
        target_series = data[target]
        data = data.drop(labels=target, axis=1)         # Since target shall be plotted separately
        corr_arr = np.array([])
        
        for col in data.columns.values:                 # Getting the column names
            corr_arr = np.append(
                corr_arr,
                data[col].corr(target_series)           # Correlation between two Series
            )
        corr_arr = corr_arr.reshape(len(corr_arr),1)    # 2-D input for heatmap
    
    fig = plt.figure(
        figsize=(12,10),
        frameon=True,
        edgecolor="black",
        linewidth=2
    )
    fig.suptitle(title)
    
    # Correlation between the non-target columns:
    sns.heatmap(
        data = data.corr(),
        cmap = colormap,
        cbar = True,
        xticklabels = True,
        yticklabels = True,
        ax = fig.add_axes([0.00, 0.00, 0.85, 0.90])
    )
    
    # Correlation between the target column and the non-target columns:
    if target is not None:
        sns.heatmap(
            data = corr_arr,
            cmap = colormap,
            cbar = False,
            axes = fig.add_axes([0.90, 0.00, 0.10, 0.90])
        )
    
    fig.show()
    return None