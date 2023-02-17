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

music_health_df.index = [x for x in range(len(music_health_df.index))]

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
sns.heatmap(corr_matrix, cmap = sns.color_palette("mako", as_cmap=True))
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

st.write("""The following graph is an histogram, that represents the relation between age and music effects. The columns represent the frequence
of each effect, with a different column for each age interval.""")
mask_1 = (music_health_df["Age"] < 20) | (music_health_df["Age"] == 20)
mask_2 = (music_health_df["Age"] > 20) & (music_health_df["Age"] < 40) | (music_health_df["Age"] == 40)
mask_3 = (music_health_df["Age"] > 40) & (music_health_df["Age"] < 60) | (music_health_df["Age"] == 60)
mask_4 = (music_health_df["Age"] > 60)
music_health_df.loc[mask_1,"Age_group"] = "under 20"
music_health_df.loc[mask_2, "Age_group"] = "21-40"
music_health_df.loc[mask_3, "Age_group"] = "41-60"
music_health_df.loc[mask_4, "Age_group"] = "over 60"

fig,ax = plt.subplots(figsize = (10,8))
ax.hist([music_health_df.loc[music_health_df["Age_group"] == "under 20","Music_effects"], music_health_df.loc[music_health_df["Age_group"] == "21-40","Music_effects"],
music_health_df.loc[music_health_df["Age_group"] == "41-60","Music_effects"], music_health_df.loc[music_health_df["Age_group"] == "over 60","Music_effects"]], 
bins = 3, label = ["under 20", "21-40", "41-60", "over 60"], width = 0.15, color = sns.color_palette("rocket")[-1:-5:-1])
plt.xticks([0.3,1,1.7], ["No effect", "Improve", "Worsen"])
plt.xlabel("Music effects")
plt.ylabel("Frequence for age")
plt.title("Music effects frequence related to age")
plt.legend()
st.pyplot(fig)
st.write("""The chart shows that in the majority of the cases, listening to music improve the situation of the patients. No negative effect 
was registered for patients over 40, the majority of which had positive effects, that improved their conditions. In all the possible
effects, the highest columns are the ones related to the youngest people: under 20 and 21-40. This result is due to the fact that these two 
categories are the most represented in our data, as we can see from the following table:""")
st.write(music_health_df["Age_group"].value_counts())

st.write("""The following graph is a set of histograms, showing the frequence of various levels of Anxiety, Depression, Insomnia and OCD, 
with a differentiation according to the effects of music on the patients.""")

sns.set(style="darkgrid")
fig, axs = plt.subplots(2, 2, figsize = (16,16))
for i in range(0,2):
    for j in range(2,4):
        if i == 0:
            axs[i,j-2].hist([music_health_df.loc[music_health_df[music_health_df.columns[-2]] == 0,music_health_df.columns[23:27][i+j-2]], music_health_df.loc[music_health_df[music_health_df.columns[-2]] == 1,music_health_df.columns[23:27][i+j-2]], 
            music_health_df.loc[music_health_df[music_health_df.columns[-2]] == 2,music_health_df.columns[23:27][i+j-2]]], color = ["mediumturquoise", "cornflowerblue", "blue"], label = ["No effect", "Improve", "Worsen"])
            axs[i,j-2].title.set_text(music_health_df.columns[23:27][i+j-2])
            axs[i,j-2].set_xlabel(music_health_df.columns[23:27][i+j-2])
            axs[i,j-2].set_ylabel("Frequence")
            axs[i,j-2].legend()
        else:
            axs[i,j-2].hist([music_health_df.loc[music_health_df[music_health_df.columns[-2]] == 0,music_health_df.columns[23:27][i+j-1]], music_health_df.loc[music_health_df[music_health_df.columns[-2]] == 1,music_health_df.columns[23:27][i+j-1]], 
            music_health_df.loc[music_health_df[music_health_df.columns[-2]] == 2,music_health_df.columns[23:27][i+j-1]]], color = ["mediumturquoise", "cornflowerblue", "blue"], label = ["No effect", "Improve", "Worsen"])
            axs[i,j-2].title.set_text(music_health_df.columns[23:27][i+j-1])
            axs[i,j-2].set_xlabel(music_health_df.columns[23:27][i+j-1])
            axs[i,j-2].set_ylabel("Frequence")
            axs[i,j-2].legend()
st.pyplot(fig)

st.write("""From the graph, we can understand that the majority of people who registered the positive effect of music have high levels of anxiety and depression,
but low levels of insomnia and OCD. Focusing on the worsen effect, it can be said that the graphs don't help in any further analysis because the
bars of the histograms are very low. That is due to the fact that we had data about only 17 patients with a value "worsen" in the music effects column
The only not irrelevant bar is the one related to insomnia equal to 10, that reaches a frquence around 8 for the people who had a worsen effect. Finally, 
we can shift to patients who had no effect due to listening to music. They generally tend to have low levels of anxiety and even more of OCD. Their level of 
anxiety seems to be independent, because the Aqua colored bars in the first histogram are more or less equally high. Their level of depression fluctuates a 
lot, but the highest frequences are registered at both very low and very high levels.""")

st.write("""With the following plot, I wanted to analyze the relationship between Anxiety and Depression, since I found that they are the couple 
with the highest level of correlation. I represented these two columns in 4 scatter plots, each one for an age group.""")
fig, ax = plt.subplots(2, 2)
fig = sns.relplot(data = music_health_df, x="Anxiety", y="Depression", hue = "Age_group", col = "Age_group", kind="scatter", 
col_wrap = 2, palette = ["g", "r", "orange", "blue"], marker = "X")
st.pyplot(fig)
st.write("""As you can see, the plots corresponding to the youngest groups (under 20 and 21-40) are the ones that are responsible for a reduction
in the correlation between the two attributes. That is due to the fact that their graphs are very confused, and points are represented without
an explicable pattern. On the contrary, the other two groups seem to express a higher level of correlation between anxiety and depression, 
in particular for the over 60 patients (it seems that we can approximate it with a linear relationship).""")

st.write("""The next idea was to investigate and further analyze the relationship between Anxiety and Depression fo the over 60 group. 
I applied a linear model in order to understand if there was a linear correlation among these two variables. Starting from m = 1 and q = 0
(that seemed to be reasonable values for the slope and the intercept of the straight line), I created and used some functions, of which the
most important are the fitting function and the squared error function. After applying the fitting to the model, the error decreased - as we can
see below - and the result obtained is the following:""")

x_train = music_health_df.loc[music_health_df["Age_group"] == "over 60","Anxiety"].loc[::2]
y_train = music_health_df.loc[music_health_df["Age_group"] == "over 60","Depression"].loc[::2]
x_test = music_health_df.loc[music_health_df["Age_group"] == "over 60","Anxiety"].loc[2::2]
y_test = music_health_df.loc[music_health_df["Age_group"] == "over 60","Depression"].loc[2::2]

def straight_line(x, m, q): 
  return (m*x)+q

def show_plot(y_train, model): 
  fig, ax = plt.subplots()
  plt.scatter(music_health_df.loc[music_health_df["Age_group"] == "over 60","Anxiety"], music_health_df.loc[music_health_df["Age_group"] == "over 60","Depression"])
  plt.plot(x_train, model, "red")
  st.pyplot(fig)

def squared_error(y, model):
  e = y - model
  sq_e = e**2
  return sum(sq_e)

def fit(y_train, m, q, steps = 200, epsilon = 0.01):  
  model = straight_line(x_train, m, q)
  sq_e = squared_error(y_train, model)
  st.write("Initial error:", sq_e)
  for i in range(steps):
    m_ = m + (np.random.choice([1,-1], size = 1)*epsilon) 
    q_ = q + (np.random.choice([1,-1], size = 1)*epsilon) 
    model_ = straight_line(x_train, m_, q)
    sq_e_ = squared_error(y_train, model_)
    if sq_e_ < sq_e: 
      m = m_
      q = q_
      sq_e = sq_e_
  st.write("Final error:", sq_e)
  return m, q

m, q = fit(y_train, 1 , 0, steps = 2000, epsilon = 0.001)
# model = straight_line(x_train, m, q)
# show_plot(y_train, model)

fig, ax = plt.subplots()
plt.scatter(x_train, y_train, c = "green", label = "Train points")
plt.scatter(x_test, y_test, c='red', label = "Test points")
x = np.arange(len(y_train)+1)
model = straight_line(x, m, q)
plt.plot(model, label = "Model")
plt.legend()
st.pyplot(fig)

st.write("Where the slope is", m[0], "and the intercept is", q[0], ".")

st.write("""Since the squared error is not so bad, we can conclude this analysis by saying that anxiety and depression in over 60 
patients are somehow related, having almost a linear relationship. For this reason these two attributes are generally registered
simultaneously and with correlated levels in older people.""")