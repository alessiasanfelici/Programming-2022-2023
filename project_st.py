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

backup_dataset2 = music_health_df.copy()

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

st.header("Plots")

st.write("""The firt interesting type of plot is the histogram. I used it to represent the frequence of listening to the different 
genres. In the x ax we can identify the frequence of listening, while in the y ax the number of people corresponding to each label 
are represented.""")
y = st.selectbox("Select the plot that you want to see:", music_health_df.columns[7:23], key = 1)
for col in music_health_df.columns[7:23]:
  if y == col:
    fig,ax = plt.subplots()
    ax.hist(backup_dataset2[col], color = "lightgreen")
    ax.set_title("Frequence of " + str(col)[5:].replace("_", " ").replace("music", "") + " music")
    ax.set_xlabel("Frequence")
    ax.set_ylabel(str(col)[5:].replace("_", " ").replace("music", "") + " music")
    st.pyplot(fig)

st.write("The second step correspond to a comparison between the various different genres of music. The idea is to use 4 pie charts.")
frequences = []
for col in music_health_df.columns[7:23]:
    freq = dict(backup_dataset2[col].value_counts())
    frequences.append(freq)
rarely_list = [frequences[i]["Rarely"] for i in range(len(frequences))]
never_list = [frequences[i]["Never"] for i in range(len(frequences))]
sometimes_list = [frequences[i]["Sometimes"] for i in range(len(frequences))]
very_frequently_list = [frequences[i]["Very frequently"] for i in range(len(frequences))]

colors = ["Aqua", "Red", "Blue", "Green", "DarkMagenta", "Linen", "Gray", "HotPink", "Brown", "LightSeaGreen", "Olive", 
          "Orchid", "Thistle", "SpringGreen", "Peru", "Yellow"]

fig,axs = plt.subplots(2, 2, figsize = (18,14))
axs[0,0].pie(never_list, startangle = 90, labels = music_health_df.columns[7:23], colors = colors)
axs[0,0].title.set_text("Never")
axs[0,1].pie(rarely_list, startangle = 90,labels = music_health_df.columns[7:23], colors = colors)
axs[0,1].title.set_text("Rarely")
axs[1,0].pie(sometimes_list, startangle = 90, labels = music_health_df.columns[7:23], colors = colors)
axs[1,0].title.set_text("Sometimes")
axs[1,1].pie(very_frequently_list, startangle = 90, labels = music_health_df.columns[7:23], colors = colors)
axs[1,1].title.set_text("Very frequently")
st.pyplot(fig)

st.write("""From the charts, it is easy to see that the genre that has the highest amount of people listening to it very frequently is Rock,
followed by Pop. On the contrary, the genres that have the highest amount of people never listening to it are Gospel, K-Pop and Latin.""")
st.write("""The charts corresponding to the labels Rarely and Sometimes are almost balanced between the various genres, in particular the first one.""")

