import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
import seaborn as sns
music_health_df = pd.read_csv("Music & Mental Health.csv")
st.title("Music & Mental Health Project")
st.write("The selected Dataset contains data about...")
st.header("Exploration of the Dataset")

music_health_df.columns = [col.replace(" ", "_").replace("Frequency", "Freq").replace("[","").replace("]","") for col in music_health_df.columns]

delate_col = ["Timestamp", "Permissions", "BPM", "Foreign_languages", "Primary_streaming_service"]
for col in delate_col:
    music_health_df.drop([col], axis = 1, inplace = True)

music_health_df.drop(music_health_df[music_health_df["Music_effects"].isnull()].index, inplace = True) #I delete the rows that don't have a value in the last column, because they are irrelevant: I won't be able to use them for my study.
null_col = ["While_working", "Instrumentalist"]
for col in null_col:
    music_health_df.drop(music_health_df[music_health_df[col].isnull()].index, inplace = True)
music_health_df["Age"].fillna(round(music_health_df["Age"].mean()), inplace = True) #I replace the remaining missing value in the column "Age" with the mean of this column(rounded)

for i in music_health_df.columns[7:23]:
  music_health_df.loc[music_health_df[i] == music_health_df[i].unique()[0], i] = 0
  music_health_df.loc[music_health_df[i] == music_health_df[i].unique()[1], i] = 1
  music_health_df.loc[music_health_df[i] == music_health_df[i].unique()[2], i] = 2
  music_health_df.loc[music_health_df[i] == music_health_df[i].unique()[3], i] = 3
for i in music_health_df.columns[2:5]:
  music_health_df.loc[music_health_df[i] == "Yes", i] = 0
  music_health_df.loc[music_health_df[i] == "No", i] = 1
music_health_df.loc[music_health_df["Exploratory"] == "Yes", "Exploratory"] = 0
music_health_df.loc[music_health_df["Exploratory"] == "No", "Exploratory"] = 1
music_health_df.loc[music_health_df["Music_effects"] == music_health_df.Music_effects.unique()[0], "Music_effects"] = 0
music_health_df.loc[music_health_df["Music_effects"] == music_health_df.Music_effects.unique()[1], "Music_effects"] = 1
music_health_df.loc[music_health_df["Music_effects"] == music_health_df.Music_effects.unique()[2], "Music_effects"] = 2

music_health_df[music_health_df.columns[7:23]] = music_health_df[music_health_df.columns[7:23]].astype(int)
music_health_df[music_health_df.columns[2:5]] = music_health_df[music_health_df.columns[2:5]].astype(int)
music_health_df["Exploratory"] = music_health_df["Exploratory"].astype(int)
music_health_df["Music_effects"] = music_health_df["Music_effects"].astype(int)
music_health_df[music_health_df.columns[0:2]] = music_health_df[music_health_df.columns[0:2]].astype(int)
music_health_df[music_health_df.columns[23:27]] = music_health_df[music_health_df.columns[23:27]].astype(int)

st.write("After having cleaned the dataset, I started exploring it, by printing the first rows, and using info() and describe() functions")
st.write(music_health_df.head())
st.text(music_health_df.info())
st.write(music_health_df.describe())

st.write("""Then, I studied the correlation between the pairs of columns of the dataset, except for the non-numeric columns. 
The plot of a heatmap allowed me to understand that the correlation of the columns was generally low. This result proves that the pairs
have not a strong linear correlation. However, I decided to search for the maximum negaive and positive values, in order to find the 
most correlated columns.""")
corr_matrix = music_health_df[music_health_df.columns[music_health_df.columns != "Fav_genre"]].corr()
fig, ax = plt.subplots()
sns.heatmap(corr_matrix)
st.write(fig)

corr_sequence = corr_matrix.unstack()
sorted_corr_sequence = corr_sequence.sort_values(kind = "quicksort")
st.write(sorted_corr_sequence[-29:-28])
st.write("""The strongest positive correlation is between the columns Anxiety and Depression, registering a value around 0.5209.
This reflects the idea that anxiety and depression are in a sense linearly correlated, with an increase in the second if the first increase, and 
viceversa.""")
st.write(sorted_corr_sequence[0:1])
st.write("""On the other hand, the strongest negative correlation can be detected between the columns Hours_per_day and While_working, 
assuming a value around -0.2879. This value is too small to say that these two columns are somehow negatively correlated.""")