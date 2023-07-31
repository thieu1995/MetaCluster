#!/usr/bin/env python
# Created by "Thieu" at 17:17, 31/07/2023 ----------%                                                                               
#       Email: nguyenthieu2102@gmail.com            %                                                    
#       Github: https://github.com/thieu1995        %                         
# --------------------------------------------------%

import pandas as pd
import plotly.express as px
import plotly.io as pio

pio.kaleido.scope.mathjax = None


def export_boxplot_figures(df, xlabel="Optimizer", ylabel=None, title="Boxplot of comparison models",
                           show_legend=True, show_mean_only=False,
                           exts=(".png", ".pdf"), file_name="boxplot", save_path="history"):
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
    fig = px.box(df, x="optimizer", y=col_name, color="optimizer", labels={"model": xlabel, col_name: ylabel})
    fig.update_traces(boxmean=boxmean)  # boxmean=True if want to show mean only
    fig.update_layout(margin=dict(l=20, r=20, t=30, b=20), showlegend=show_legend,
                      title={'text': title, 'x': 0.5, 'xanchor': 'center', 'yanchor': 'top'})
    for idx, ext in enumerate(exts):
        fig.write_image(f"{save_path}/{file_name}{ext}")


def export_convergence_figures(df, xlabel="Epoch", ylabel="Fitness value",
                               title="Convergence chart of comparison models",
                               exts=(".png", ".pdf"), file_name="boxplot", save_path="history"):
    if xlabel is None:
        xlabel = ""
    if ylabel is None:
        ylabel = ""
    if title is None:
        title = ""
    # Melt the DataFrame to convert it from wide to long format
    df_long = pd.melt(df, id_vars='index', var_name='Column', value_name='Value')
    # Define the line chart using Plotly Express
    fig = px.line(df_long, x='index', y='Value', color='Column', labels={'index': xlabel, 'Value': ylabel, 'Column': 'Optimizer'})
    fig.update_layout(margin=dict(l=20, r=20, t=30, b=20), showlegend=True,
                      title={'text': title, 'x': 0.5, 'xanchor': 'center', 'yanchor': 'top'})
    for idx, ext in enumerate(exts):
        fig.write_image(f"{save_path}/{file_name}{ext}")
