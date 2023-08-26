#!/usr/bin/env python
# Created by "Thieu" at 17:17, 31/07/2023 ----------%                                                                               
#       Email: nguyenthieu2102@gmail.com            %                                                    
#       Github: https://github.com/thieu1995        %                         
# --------------------------------------------------%

import pandas as pd
import plotly.express as px
import plotly.io as pio

pio.kaleido.scope.mathjax = None


def export_boxplot_figures(df, figure_size=(500, 600), xlabel="Optimizer", ylabel=None, title="Boxplot of comparison models",
                           show_legend=True, show_mean_only=False, exts=(".png", ".pdf"), file_name="boxplot", save_path="history"):
    """
    Parameters
    ----------
    df : pd.DataFrame
        The format of df parameter:
            optimizer	DBI
            FBIO	    1.18145
            FBIO	    1.1815
            GWO	        1.18145
            GWO	        1.18153
            FBIO	    1.18147
            FBIO	    1.18145
            GWO	        1.18137

    figure_size : list, tuple, np.ndarray, None; default=None
        The size for saved figures. `None` means it will automatically set for you.
        Or you can pass (width, height) of figure based on pixel (100px to 1500px)

    xlabel : str; default="Optimizer"
        The label for x coordinate of boxplot figures.

    ylabel : str; default=None
        The label for y coordinate of boxplot figures.

    title : str; default="Boxplot of comparison models"
        The title of figures, it should be the same for all objectives since we have y coordinate already difference.

    show_legend : bool; default=True
        Show the legend or not. For boxplots we can turn on or off this option, but not for convergence chart.

    show_mean_only : bool; default=False
        You can show the mean value only or you can show all mean, std, median of the box by this parameter

    exts : list, tuple, np.ndarray; default=(".png", ".pdf")
        List of extensions of the figures. It is for multiple purposes such as latex (need ".pdf" format), word (need ".png" format).

    file_name : str; default="boxplot"
        The prefix for filenames that will be saved.

    save_path : str; default="history"
        The path to save the figure
    """
    yaxis = dict(nticks=8)
    if abs(df.iloc[0][-1]) > 1e5 or abs(df.iloc[0][-1]) < 1e-5:
        yaxis = dict(tickformat=".2e", exponentformat="power", showexponent="first")
    if xlabel is None:
        xlabel = ""
    if ylabel is None:
        ylabel = ""
    if title is None:
        title = ""
    boxmean = True if show_mean_only else "sd"
    col_name = list(df.columns)[-1]
    if ylabel is None:
        ylabel = col_name
    fig = px.box(df, x="optimizer", y=col_name, color="optimizer",
                 labels={'optimizer': xlabel, col_name: ylabel})
    fig.update_traces(boxmean=boxmean)  # boxmean=True if want to show mean only
    fig.update_layout(width=figure_size[0], height=figure_size[1],
                      margin=dict(l=25, r=20, t=40, b=20), showlegend=show_legend,
                      title={'text': title, 'x': 0.5, 'xanchor': 'center', 'yanchor': 'top'},
                      font=dict(size=15), yaxis=yaxis)
    for idx, ext in enumerate(exts):
        fig.write_image(f"{save_path}/{file_name}{ext}")


def export_convergence_figures(df, figure_size=(500, 600), xlabel="Epoch", ylabel="Fitness value", title="Convergence chart of comparison models",
                               legend_name="Optimizer", exts=(".png", ".pdf"), file_name="convergence", save_path="history"):
    """
    Parameters
    ----------
    df : pd.DataFrame
        The format of df parameter:
            FBIO	    GWO
            62.62501039	62.72457583
            62.62085777	62.71386468
            62.62085777	62.71386468
            62.62085777	62.71386468
            62.62085777	62.66383109
            62.62085777	62.66310589

    figure_size : list, tuple, np.ndarray, None; default=None
            The size for saved figures. `None` means it will automatically set for you.
            Or you can pass (width, height) of figure based on pixel (100px to 1500px)

    xlabel : str; default="Optimizer"
            The label for x coordinate of convergence figures.

    ylabel : str; default=None
        The label for y coordinate of boxplot figures.

    title : str; default="Convergence chart of comparison models"
        The title of figures, it should be the same for all objectives since we have y coordinate already difference.

    legend_name : str; default="Optimizer"
        Set the name for the legend.

    exts : list, tuple, np.ndarray; default=(".png", ".pdf")
        List of extensions of the figures. It is for multiple purposes such as latex (need ".pdf" format), word (need ".png" format).

    file_name : str; default="convergence"
        The prefix for filenames that will be saved.

    save_path : str; default="history"
        The path to save the figure
    """
    yaxis = dict(nticks=8)
    if abs(df.iloc[0][-1]) > 1e5 or abs(df.iloc[0][-1]) < 1e-5:
        yaxis = dict(tickformat=".2e", exponentformat="power", showexponent="first")
    if xlabel is None:
        xlabel = ""
    if ylabel is None:
        ylabel = ""
    if title is None:
        title = ""
    if legend_name is None:
        legend_name = "Optimizer"
    # Melt the DataFrame to convert it from wide to long format
    df = df.reset_index()
    df_long = pd.melt(df, id_vars='index', var_name='Column', value_name='Value')
    # Define the line chart using Plotly Express
    fig = px.line(df_long, x='index', y='Value', color='Column',
                  labels={'index': xlabel, 'Value': ylabel, 'Column': legend_name})
    fig.update_layout(width=figure_size[0], height=figure_size[1],
                      margin=dict(l=25, r=20, t=40, b=20), showlegend=True,
                      title={'text': title, 'x': 0.5, 'xanchor': 'center', 'yanchor': 'top'},
                      font=dict(size=15), yaxis=yaxis)
    for idx, ext in enumerate(exts):
        fig.write_image(f"{save_path}/{file_name}{ext}")
