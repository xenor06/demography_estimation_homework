"""Visualization function examples for the homework project"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def plot_degree_distribution(G):
    """Plot a degree distribution of a graph

    TODO: log-log binning! To understand this better, check out networksciencebook.com
    """
    plot_df = (
        pd.Series(dict(G.degree)).value_counts().sort_index().to_frame().reset_index()
    )
    plot_df.columns = ["k", "count"]
    plot_df["log_k"] = np.log(plot_df["k"])
    plot_df["log_count"] = np.log(plot_df["count"])
    fig, ax = plt.subplots()

    ax.scatter(plot_df["k"], plot_df["count"])
    ax.set_xscale("log")
    ax.set_yscale("log")
    fig.suptitle("Mutual Degree Distribution")
    ax.set_xlabel("k")
    ax.set_ylabel("count_k")


def plot_age_distribution_by_gender(nodes):
    """Plot a histogram where the color represents gender"""
    plot_df = nodes[["AGE", "gender"]].copy(deep=True).astype(float)
    plot_df["gender"] = plot_df["gender"].replace({0.0: "woman", 1.0: "man"})
    sns.histplot(data=plot_df, x="AGE", hue="gender", bins=np.arange(0, 45, 5) + 15)


def plot_node_degree_by_gender(nodes, G):
    """Plot the average of node degree across age and gender"""
    # TODO: this could be generalized for different node level statistics as well!
    nodes_w_degree = nodes.set_index("user_id").merge(
        pd.Series(dict(G.degree)).to_frame(),
        how="left",
        left_index=True,
        right_index=True,
    )
    nodes_w_degree = nodes_w_degree.rename({0: "degree"}, axis=1)
    plot_df = (
        nodes_w_degree.groupby(["AGE", "gender"]).agg({"degree": "mean"}).reset_index()
    )
    sns.lineplot(data=plot_df, x="AGE", y="degree", hue="gender")


def plot_age_relations_heatmap(edges_w_features):
    """Plot a heatmap that represents the distribution of edges"""
    # TODO: check what happpens without logging
    # TODO: instead of logging check what happens if you normalize with the row sum
    #  make sure you figure out an interpretation of that as well!
    # TODO: separate these charts by gender as well
    # TODO: column names could be nicer
    plot_df = edges_w_features.groupby(["gender_x", "gender_y", "AGE_x", "AGE_y"]).agg(
        {"smaller_id": "count"}
    )
    plot_df_w_w = plot_df.loc[(0, 0)].reset_index()
    plot_df_heatmap = plot_df_w_w.pivot_table(
        index="AGE_x", columns="AGE_y", values="smaller_id"
    ).fillna(0)
    plot_df_heatmap_logged = np.log(plot_df_heatmap + 1)
    sns.heatmap(plot_df_heatmap_logged)

    
def plot_neighbor_connectivity_by_gender(nodes, edges, G):
    """Plot the average degree of neighbors of a specific user across age and gender"""
    
    #counts the neighbors of a specific user
    degreeNC=edges.copy()
    degreeNC2=degreeNC.groupby(["smaller_id"]).count()
    degreeNC3=degreeNC.merge(degreeNC2, left_on="greater_id",right_on="smaller_id",how="outer")
    degreeNC3["greater_id_y"]=degreeNC3["greater_id_y"].fillna(0)
    degreeNC3=degreeNC3.drop(columns=("greater_id_x"))
    degreeNC3=degreeNC3.groupby(["smaller_id"]).count()
    degreeNC3=degreeNC3.rename(columns={"greater_id_y":"Neighbor Connectivity"})
    
    #gets the average degree of neighbors of a specific user across age and gender
    nodes_TRAINNC=nodes[nodes["TRAIN_TEST"] == "TRAIN"]
    nodes_TRAINNC=nodes_TRAINNC.drop(columns=["TRAIN_TEST","public","region"])
    degreeNC3=degreeNC3.merge(nodes_TRAINNC,left_on="smaller_id",right_on="user_id",how="inner")
    degreeNC3=degreeNC3.set_index("user_id")
    degreeNC3=degreeNC3.groupby(["AGE", "gender"]).agg({"Neighbor Connectivity": "mean"}).reset_index()
    
    sns.lineplot(data=degreeNC3, x="AGE", y="Neighbor Connectivity", hue="gender")