"""Functions to load and filter the network"""

import pandas as pd
import numpy as np

# notice that  I use all caps for module level strings
# it may be nicer to export this to a csv, but life is sometimes too short to be tidy
COLUMNS_STR = """    user_id
    public
    completion_percentage
    gender
    region
    last_login
    registration
    AGE
    body
    I_am_working_in_field
    spoken_languages
    hobbies
    I_most_enjoy_good_food
    pets
    body_type
    my_eyesight
    eye_color
    hair_color
    hair_type
    completed_level_of_education
    favourite_color
    relation_to_smoking
    relation_to_alcohol
    sign_in_zodiac
    on_pokec_i_am_looking_for
    love_is_for_me
    relation_to_casual_sex
    my_partner_should_be
    marital_status
    children
    relation_to_children
    I_like_movies
    I_like_watching_movie
    I_like_music
    I_mostly_like_listening_to_music
    the_idea_of_good_evening
    I_like_specialties_from_kitchen
    fun
    I_am_going_to_concerts
    my_active_sports
    my_passive_sports
    profession
    I_like_books
    life_style
    music
    cars
    politics
    relationships
    art_culture
    hobbies_interests
    science_technologies
    computers_internet
    education
    sport
    movies
    travelling
    health
    companies_brands
    more"""
COLUMNS_LIST = [col.strip() for col in COLUMNS_STR.split("\n")]

# reproducibility matters!
np.random.seed(42)


def select_relevant_profiles(all_profiles):
    """Select relevant profiles
    criteria:
    * is public
    * region is selected region
    * AGE specified
    * GENDER SPECIFIED
    """
    public_condition = all_profiles["public"] == 1
    age_condition = all_profiles["AGE"] > 14
    gender_condition = all_profiles["gender"].isin([0, 1])
    return all_profiles.loc[public_condition & age_condition & gender_condition]


def select_relevant_edges(all_edges, selected_ids):
    """Select relevant edges for those profiles that are relevant"""
    source_condition = all_edges["source"].isin(selected_ids)
    sink_condition = all_edges["sink"].isin(selected_ids)
    return all_edges.loc[source_condition & sink_condition]


def convert_edges_to_undirected(edges):
    """Convert edges to undirected, and keep only mutual connections"""
    undirected_edges = (
        edges.assign(
            smaller_id=lambda df: df[["source", "sink"]].min(axis=1),
            greater_id=lambda df: df[["source", "sink"]].max(axis=1),
        )
        .groupby(["smaller_id", "greater_id"])
        .agg({"source": "count"})
    )
    print(undirected_edges["source"].value_counts())
    return (
        undirected_edges.loc[undirected_edges["source"] == 2]
        .drop("source", axis=1)
        .reset_index()
    )


def load_and_select_profiles_and_edges():
    """load and select relevant profiles, then filter and undirect edges"""
    print("loading profiles")
    # TODO: Add some functionality to only read a subset of the data!
    profiles = pd.read_csv(
        "data/soc-pokec-profiles.txt",
        sep="\t",
        names=COLUMNS_LIST,
        index_col=False,
        usecols=["user_id", "public", "gender", "region", "AGE"],
    )
    print("loading edges")
    edges = pd.read_csv(
        "data/soc-pokec-relationships.txt", sep="\t", names=["source", "sink"]
    )
    selected_profiles = select_relevant_profiles(profiles)
    selected_ids = selected_profiles["user_id"].unique()
    selected_edges = select_relevant_edges(edges, selected_ids)

    undirected_edges = convert_edges_to_undirected(selected_edges)
    nodes_with_edges = set(undirected_edges["smaller_id"].unique()).union(
        undirected_edges["greater_id"].unique()
    )
    print(f"Selected profiles: {len(selected_profiles)}")
    print(f"Nodes with edges: {len(nodes_with_edges)}")
    selected_profiles = selected_profiles[
        selected_profiles["user_id"].isin(nodes_with_edges)
    ]
    selected_profiles["AGE"] = selected_profiles["AGE"].clip(upper=50)
    selected_profiles = remove_test_set_gender_and_age(selected_profiles)
    return selected_profiles, undirected_edges


def remove_test_set_gender_and_age(nodes):
    """Remove the gender feature from a subset of the nodes for estimation"""
    # todo: the 40k  random can be adjusted if youre working with a subset
    test_profiles = np.random.choice(nodes["user_id"].unique(), 40000, replace=False)
    nodes["TRAIN_TEST"] = "TRAIN"
    test_condition = nodes["user_id"].isin(test_profiles)
    nodes.loc[test_condition, ["AGE", "gender"]] = np.nan
    nodes.loc[test_condition, ["TRAIN_TEST"]] = "TEST"

    return nodes


def test_set_gender_estimation(nodes, edges):
    """Create an estimation of gender on the test set! Count the number of male and female friends of each node,
    and predict accordingly"""
    
    #filter the nodes for the TRAIN data
    nodes_TRAIN=nodes[nodes["TRAIN_TEST"] == "TRAIN"]
    nodes_TRAIN=nodes_TRAIN[["user_id","gender"]]
    nodes_TRAIN=nodes_TRAIN.set_index("user_id")
    
    #for each node count the number of male and female friends, and their percentage
    degree=edges.copy()
    degree=degree.rename(columns={"greater_id": "user_id"})
    degree=pd.merge(degree, nodes_TRAIN, on="user_id",how="inner")
    degree=degree.groupby(["smaller_id","gender"]).count()
    degree=degree.rename(columns={"user_id": "Degree"})
    degree.index.names=['user_id','gender']
    degree=degree.groupby(["user_id","gender"]).sum()/degree.groupby(["user_id"]).sum()
    degree=degree.reset_index()
    degree=degree.drop_duplicates(subset=['user_id'])
    degree=degree.set_index("user_id")
    
    #filter the nodes for the TEST data
    nodes_TESTDATA=nodes[nodes["TRAIN_TEST"] == "TEST"]
    nodes_TESTDATA=nodes_TESTDATA[["user_id","gender"]]
    nodes_TESTDATA=nodes_TESTDATA.set_index("user_id")
    
    #predict the gender on the test set - for each node if the percentage of male friends is over 0.5,
    #so the node has more male friends than female, then predict that the unknown gender friend is male as well
    #if the percentage equals 0.5, then we estimate that the unknown gender friend is male
    #those who have no friends were left out from the prediction, as in the article
    degree_TEST=edges.copy()
    nodes_TEST=nodes_TESTDATA.merge(degree_TEST, left_on="user_id",right_on="smaller_id", how="inner")
    nodes_TEST=nodes_TEST.groupby(["smaller_id","greater_id"]).count()
    nodes_TEST=nodes_TEST.reset_index()
    nodes_TEST=nodes_TEST.merge(degree, left_on="smaller_id",right_on="user_id", how="inner")
    nodes_TEST=nodes_TEST.drop_duplicates(subset=["smaller_id"])
    nodes_TEST=nodes_TEST.drop(columns=("greater_id"))
    nodes_TEST["gender_x"]=nodes_TEST["Degree"].apply(lambda x: 0.0 if x>0.5 else 1.0)
    nodes_TEST["gender_x"]=nodes_TEST["gender_x"]+nodes_TEST["gender_y"]
    nodes_TEST=nodes_TEST.drop(columns=["gender_y","Degree"])
    nodes_TEST=nodes_TEST.rename(columns={"smaller_id": "user_id","gender_x": "gender"})
    
    return nodes_TEST.set_index("user_id")
    