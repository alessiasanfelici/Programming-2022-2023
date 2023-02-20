import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import io
import streamlit as st
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import sklearn.metrics as metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

music_health_df = pd.read_csv("Music & Mental Health.csv")
st.title("Music & Mental Health Project, Alessia Sanfelici")

st.header("Introduction and first study of the dataset")

#Explanation and introduction of the dataset 
st.write("""The selected Dataset collects information about the effect of Music Therapy on people. The data contained in the dataset refers to
answers of selected individuals about their personal background, their music habits, their ranks about how often they listen to 16 music genres (Never, Rarely, 
Sometimes or Very Frequently), and their ranks about experienced levels of Anxiety, Depression, Insomnia and OCD (from 0 to 10). The last column 
represents the effect of listening to music on each person.\\
After a general presentation of the dataset, this report will focus on the study of the behaviour of the selected individuals,
in order to understand music tastes and trends in levels of anxiety, depression, insomnia and OCD, and study the relationship between age and hours of listening.
Then, the last focus will be on the attribute Music effects: in the last part of the analysis I focused on the study of this column, both in relationship with 
other attributes and alone. This part also aims at using models to predict and better understand the music effects on the selected individuals.\\
The scope of this analysis is to find if there is any correlation between music tastes and mental health, focusing on the effect that 
music has on mental health (does it improve the condition of an individual, or has it negative consequences?). The results could be a guide for 
better applying music therapy, according to age and habits. Moreover, we can use the available data to gain a deeper knowledge about human mind, 
as it is always difficult to understand and explain how it works and what is able to influence it.""")

#cleaning of the dataset
music_health_df.columns = [col.replace(" ", "_").replace("Frequency", "Freq").replace("[","").replace("]","") for col in music_health_df.columns]

backup_dataset2 = music_health_df.copy() #backup copy of the dataset

delate_col = ["Timestamp", "Permissions", "BPM", "Foreign_languages", "Primary_streaming_service"] #list of columns to delate
for col in delate_col:
    music_health_df.drop([col], axis = 1, inplace = True)

music_health_df.drop(music_health_df[music_health_df["Music_effects"].isnull()].index, inplace = True) 
#I delete the rows that don't have a value in the last column, because they are irrelevant: I won't be able to use them for my study.
null_col = ["While_working", "Instrumentalist"] #I delate the null values in these columns
for col in null_col:
    music_health_df.drop(music_health_df[music_health_df[col].isnull()].index, inplace = True)
music_health_df["Age"].fillna(round(music_health_df["Age"].mean()), inplace = True) 
#I replace the remaining missing value in the column "Age" with the mean of this column(rounded)

for i in music_health_df.columns[7:23]:
  #each label correspond to a number
  music_health_df.loc[music_health_df[i] == "Never", i] = 0
  music_health_df.loc[music_health_df[i] == "Sometimes", i] = 2
  music_health_df.loc[music_health_df[i] == "Rarely", i] = 1
  music_health_df.loc[music_health_df[i] == "Very frequently", i] = 3
#repeat the same process for the columns with values "Yes" and "No" and for the column Music Effects
for i in music_health_df.columns[2:5]:
  music_health_df.loc[music_health_df[i] == "Yes", i] = 1
  music_health_df.loc[music_health_df[i] == "No", i] = 0
replace_dict = {"No": 0, "Yes": 1}
music_health_df.Exploratory.replace(replace_dict, inplace = True)
replace_dict2 = {"No effect": 0, "Improve": 1, "Worsen": 2}
music_health_df.Music_effects.replace(replace_dict2, inplace = True)

#transforming the columns into integer columns
music_health_df[music_health_df.columns[7:23]] = music_health_df[music_health_df.columns[7:23]].astype(int)
music_health_df[music_health_df.columns[2:5]] = music_health_df[music_health_df.columns[2:5]].astype(int)
music_health_df["Exploratory"] = music_health_df["Exploratory"].astype(int)
music_health_df["Music_effects"] = music_health_df["Music_effects"].astype(int)
#transforming to integers also the float columns (because they have integer values inside)
music_health_df[music_health_df.columns[0:2]] = music_health_df[music_health_df.columns[0:2]].astype(int)
music_health_df[music_health_df.columns[23:27]] = music_health_df[music_health_df.columns[23:27]].astype(int)
#replacing the indexes of the rows of the dataset, starting from 0
music_health_df.index = [x for x in range(len(music_health_df.index))]

#masks to create groups according to the age of the people
mask_1 = (music_health_df["Age"] < 20) | (music_health_df["Age"] == 20)
mask_2 = (music_health_df["Age"] > 20) & (music_health_df["Age"] < 40) | (music_health_df["Age"] == 40)
mask_3 = (music_health_df["Age"] > 40) & (music_health_df["Age"] < 60) | (music_health_df["Age"] == 60)
mask_4 = (music_health_df["Age"] > 60)
music_health_df.loc[mask_1,"Age_group"] = "under 20" #a new column that represents the age group of each row
music_health_df.loc[mask_2, "Age_group"] = "21-40"
music_health_df.loc[mask_3, "Age_group"] = "41-60"
music_health_df.loc[mask_4, "Age_group"] = "over 60"

st.write("""After having cleaned the dataset (by removing the irrelevant columns, properly changing the values in some columns from categorical
to numerical, and dealing with the missing values), I started exploring it to better understand what to do in the following steps. Printing the 
first rows, and using info() and describe() functions, allowed me to have a general overview of the available data.""")
st.write(music_health_df.head())
buffer = io.StringIO()
music_health_df.info(buf=buffer)
st.text(buffer.getvalue())
#explanation of the columns in the dataset
st.write("""The columns of the cleaned dataset are:\\
        - Age: age of the individual in the row\\
        - Hours_per_day: number of hours the person listens to music per day\\
        - While_working: does the person listens to music while studying/working?\\
        - Instrumentalist: does the person plays an instrument regularly?\\
        - Composer: does the person compose music?\\
        - Fav_genre: favourite genre\\
        - Exploratory: does the person actively explore new artists/genres?\\
        - Freq_Classical: how frequently the interviewee listens to classical music\\
        - Freq_Country: how frequently the interviewee listens to country music\\
        - Freq_EDM: how frequently the interviewee listens to EDM music\\
        - Freq_Folk: how frequently the interviewee listens to folks music\\
        - Freq_Gospel: how frequently the interviewee listens to gospel music\\
        - Freq_Hip_hop: how frequently the interviewee listens to hip hop music\\
        - Freq_Jazz: how frequently the interviewee listens to jazz music\\
        - Freq_K_pop: how frequently the interviewee listens to k-pop music\\
        - Freq_Latin: how frequently the interviewee listens to latin music\\
        - Freq_Lofi: how frequently the interviewee listens to lofi music\\
        - Freq_Metal: how frequently the interviewee listens to metal music\\
        - Freq_Pop: how frequently the interviewee listens to pop music\\
        - Freq_R&B: how frequently the interviewee listens to R&B music\\
        - Freq_Rap: how frequently the interviewee listens to rap music\\
        - Freq_Rock: how frequently the interviewee listens to rock music\\
        - Freq_Video_game_music: how frequently the interviewee listens to video game music\\
        - Anxiety: self-reported anxiety, on a scale of 0-10\\
        - Depression: self-reported depression, on a scale of 0-10\\
        - Insomnia: self-reported insomnia, on a scale of 0-10\\
        - OCD: self-reported OCD, on a scale of 0-10\\
        - Music_effects: does music improve/worsen the person's mental health conditions? (0 = No effect, 1 = Improve, 2 = Worsen)\\
        - Age_group: the age group the individual belongs to, among under 20, 21-40, 41-60 and over 60""")
st.write(music_health_df.describe())

#correlation section
st.subheader("Correlation")
st.write("""I studied the correlation between the pairs of columns of the dataset, except for the non-numeric columns. 
The plot of a heatmap allowed me to understand that the correlation of the columns was generally low. This result proves that the pairs
have not a strong linear correlation. However, I decided to search for the maximum negative and positive values, in order to find the 
most correlated columns.""")
corr_matrix = music_health_df.drop(["Fav_genre", "Age_group"], axis = 1).corr()
fig, ax = plt.subplots()
sns.heatmap(corr_matrix, cmap = sns.color_palette("mako", as_cmap=True))
st.write(fig)

#which are the most correlated attributes?
corr_sequence = corr_matrix.unstack()
sorted_corr_sequence = corr_sequence.sort_values(kind = "quicksort") #sorting the values, I can find the minimum and the maximum
col1, col2 = st.columns(2)
#the maximum value is the maximum value different from 1 (because 1 is given by the elements in the diagonal)
col1.caption("Strongest positive correlation")
col1.write(sorted_corr_sequence[-29:-28])
col2.caption("Strongest negative correlation")
col2.write(sorted_corr_sequence[0:1])
st.write("""The strongest positive correlation is between the columns Freq_Hip_hop and Freq_Rap, registering a value around 0.7823.
This reflects the idea that the frequences of listening to Hip hop and Rap are in a sense positively and linearly correlated: people that 
listen to hip hop tend to listen also to rap, and viceversa. On the other hand, the strongest negative correlation can be detected between the columns 
Age and Freq_Video_game_music, assuming a value around -0.2676. This value is too small to say that these two columns have a similar behaviour.""")

#section with plots and models
st.header("Deeper analysis with Plots and Models")

st.subheader("Music genres listening frequence")
#music tastes and genres
st.write("""Firstly, I wanted to focus my analysis on the study of music tastes of the selected individuals. For this scope, I used an histogram 
to represent the frequence of listening to the different genres. In the x axis we can see the frequence of listening (identified with the 
labels Never, Sometimes, Rarely and Very Frequently), while in the y axis the number of people corresponding to each label is represented.""")
y = st.selectbox("Select the plot that you want to see:", music_health_df.columns[7:23], key = 1)
#an histogram for the columns corresponding to the listening frequence of a music genre
for col in music_health_df.columns[7:23]:
  if y == col:
    fig,ax = plt.subplots()
    backup_dataset2[col].value_counts().plot.bar(color = "lightgreen")
    plt.xticks(rotation = 30)
    ax.set_title("Frequence of " + str(col)[5:].replace("_", " ").replace("music", "") + " music")
    plt.ylabel("Counts")
    st.pyplot(fig)

st.write("""The following step correspond to a comparison between the different genres of music. The idea is to use 4 pie charts, to understand the
the trend in listening to music. What are the most and the least listened genres?""")
frequences = [] #a list with all the values that correspond to the number of people for each listening frequence
for col in music_health_df.columns[7:23]:
    freq = dict(backup_dataset2[col].value_counts())
    #a dictionary for every music genre
    frequences.append(freq) #the list is a list of dictionaries
rarely_list = [frequences[i]["Rarely"] for i in range(len(frequences))] #list of all the values corresponding to the key "Rarely"
never_list = [frequences[i]["Never"] for i in range(len(frequences))]
sometimes_list = [frequences[i]["Sometimes"] for i in range(len(frequences))]
very_frequently_list = [frequences[i]["Very frequently"] for i in range(len(frequences))]

colors = ["plum", "lemonchiffon", "c", "tan", "lightblue", "Linen", "Gray", "lightskyblue", "lightgrey", "LightSeaGreen", "skyblue", 
          "pink", "Thistle", "y", "lightsalmon", "lightcyan"]

#pie charts
fig,axs = plt.subplots(2, 2, figsize = (18,14))
axs[0,0].pie(never_list, startangle = 90, labels = music_health_df.columns[7:23], wedgeprops = { "linewidth" : 1, "edgecolor" : "black" }, 
colors = ["#B7C3F3", "#B7C3F3", "#B7C3F3", "#B7C3F3", "hotpink", "#B7C3F3", "#B7C3F3", "#DD7596", "orchid", "#B7C3F3", "#B7C3F3", "#B7C3F3", "#B7C3F3", "#B7C3F3", "#B7C3F3", "#B7C3F3"], 
explode = [0, 0, 0, 0, 0.2, 0, 0, 0.2, 0.2, 0, 0, 0, 0, 0, 0, 0])
axs[0,0].title.set_text("Never")
axs[0,1].pie(rarely_list, startangle = 90, labels = music_health_df.columns[7:23], wedgeprops = { "linewidth" : 1, "edgecolor" : "black" }, colors = colors)
axs[0,1].title.set_text("Rarely")
axs[1,0].pie(sometimes_list, startangle = 90, labels = music_health_df.columns[7:23], wedgeprops = { "linewidth" : 1, "edgecolor" : "black" }, colors = colors)
axs[1,0].title.set_text("Sometimes")
axs[1,1].pie(very_frequently_list, startangle = 90, labels = music_health_df.columns[7:23], colors = ["#B7C3F3", "#B7C3F3", "#B7C3F3", "#B7C3F3", "#B7C3F3", 
"#B7C3F3", "#B7C3F3", "#B7C3F3", "#B7C3F3", "#B7C3F3", "#B7C3F3", "orchid", "#B7C3F3", "#B7C3F3", "hotpink", "#B7C3F3"], 
wedgeprops = { "linewidth" : 1, "edgecolor" : "black" }, explode = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.2, 0, 0, 0.2, 0])
axs[1,1].title.set_text("Very frequently")
st.pyplot(fig)

st.write("""From the charts, it is easy to see that the genre that has the highest amount of people listening to it very frequently is Rock,
followed by Pop. On the contrary, the genres that have the highest amount of people never listening to it are Gospel, K-Pop and Latin.\\
The charts corresponding to the labels Rarely and Sometimes are more or less balanced between the various genres, in particular the first one.""")

st.subheader("Anxiety, Depression, Insomnia and OCD")
st.write("""I thought it could be interesting to analyze the attributes relative to Anxiety, Depression, Insomnia and OCD. My idea was to
show the frequence of the levels of these attributes, with a differentiation according to the effects of music on the selected individuals. For this 
purpose, the histogram was the better solution. So, I created 4 plots, one for each characteristic of mental health.""")

sns.set(style="darkgrid")
fig, axs = plt.subplots(2, 2, figsize = (14,14))
for i in range(0,2):
    for j in range(2,4):
        if i == 0: #first row
            axs[i,j-2].hist([music_health_df.loc[music_health_df[music_health_df.columns[-2]] == 0,music_health_df.columns[23:27][i+j-2]], music_health_df.loc[music_health_df[music_health_df.columns[-2]] == 1,music_health_df.columns[23:27][i+j-2]], 
            music_health_df.loc[music_health_df[music_health_df.columns[-2]] == 2,music_health_df.columns[23:27][i+j-2]]], color = ["mediumturquoise", "cornflowerblue", "blue"], label = ["No effect", "Improve", "Worsen"], align = "right")
            axs[i,j-2].title.set_text(music_health_df.columns[23:27][i+j-2])
            axs[i,j-2].set_xlabel(music_health_df.columns[23:27][i+j-2])
            axs[i,j-2].set_ylabel("Frequence")
            axs[i,j-2].legend()
        else:#second row
            axs[i,j-2].hist([music_health_df.loc[music_health_df[music_health_df.columns[-2]] == 0,music_health_df.columns[23:27][i+j-1]], music_health_df.loc[music_health_df[music_health_df.columns[-2]] == 1,music_health_df.columns[23:27][i+j-1]], 
            music_health_df.loc[music_health_df[music_health_df.columns[-2]] == 2,music_health_df.columns[23:27][i+j-1]]], color = ["mediumturquoise", "cornflowerblue", "blue"], label = ["No effect", "Improve", "Worsen"], align = "right")
            axs[i,j-2].title.set_text(music_health_df.columns[23:27][i+j-1])
            axs[i,j-2].set_xlabel(music_health_df.columns[23:27][i+j-1])
            axs[i,j-2].set_ylabel("Frequence")
            axs[i,j-2].legend()
st.pyplot(fig)

st.write("""From the graphs, we can understand that the majority of people who registered a positive effect due to music have high levels of anxiety and depression,
but low levels of insomnia and OCD.""")
st.write("""Focusing on the worsen effect, it can be said that the graphs don't help in any further analysis because the
bars of the relative histograms are very low. That is due to the fact that we have information about only 17 individuals with a value "worsen" in the music effects column.
For this category, the only not irrelevant bar is the one related to a level of depression equal to 10, that reaches a frequence around 8 for the people who had a worsen effect.""")
st.write("""Finally, we can focus on the individuals who had no effect due to listening to music. They generally tend to have low levels of insomnia and even more of OCD. Their level of 
anxiety seems to have no particular influence on the music effect results, because the Aqua colored bars in the first histogram are more or less equally high. Their level of 
depression fluctuates a lot, but the highest frequences are registered at both very low and very high levels.""")

st.write("""The above histograms show that anxiety and depression seem to have a similar behaviour, in particular around the highest levels. For this reason, I wanted to further the relationship 
between these two attributes, whose level of correlation is: """, corr_matrix.loc["Anxiety"]["Depression"], ".")
st.write("""I represented them through 4 scatter plots, one for each age group.""")
fig, ax = plt.subplots(2, 2)
fig = sns.relplot(data = music_health_df, x="Anxiety", y="Depression", hue = "Age_group", col = "Age_group", kind="scatter", 
col_wrap = 2, palette = ["g", "r", "orange", "blue"], marker = "X")
st.pyplot(fig)
st.write("""As you can see, the plots corresponding to the youngest groups (under 20 and 21-40) are very confused, and points are represented without
an explicable pattern. Therefore, these groups are the ones that are responsible for a reduction in the correlation between the two attributes. 
On the contrary, the other two groups seem to express a higher level of correlation between anxiety and depression, in particular for the over 
60 patients (it seems that we can approximate the points with a straight line).""")

#further study of the relationship beween anxiety and depression
st.write("""This is the reason why I investigated and further analyzed the relationship between Anxiety and Depression for the over 60 group. 
I applied a linear regression model in order to understand if there was a linear correlation among these two variables. The obtained 
result is the following:""")
#linear regression
x = music_health_df.loc[music_health_df["Age_group"] == "over 60","Anxiety"]
y = music_health_df.loc[music_health_df["Age_group"] == "over 60","Depression"]
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2) #creation of test and train subsets of the data

def straight_line(x, m, q): #function that creates a straight line
  return (m*x)+q

x_train = np.array(x_train).reshape(-1,1) #reshape for applying linear regression
y_train = np.array(y_train).reshape(-1,1)
x_test = np.array(x_test).reshape(-1,1)
model = LinearRegression()
model.fit(x_train, y_train)
y_pred = model.predict(x_test)
#representation of the linear regression (with train and test points in different colors)
fig, ax = plt.subplots()
plt.scatter(x_train, y_train, c = "blue", label = "Train points")
plt.scatter(x_test, y_test, c = "LightSeaGreen", label = "Test points")
x = np.arange(11)
line = straight_line(x, model.intercept_[0], model.coef_[0])
plt.plot(line, c = "Magenta", label = "Model")
plt.legend()
st.pyplot(fig)

st.write("Where the slope of the straight line is", model.coef_[0][0], "and the intercept is", model.intercept_[0], ".")
st.write("The mean squared error is: ", metrics.mean_squared_error(y_pred, y_test))

st.write("""Since the mean squared error is relatively low, we can conclude this analysis by saying that anxiety and depression in over 60 
individuals are somehow correlated, having almost a linear relationship. For this reason these two attributes are generally registered
simultaneously and with correlated levels in older people.""")

st.subheader("Age and Hours per day - Clustering")
st.write("""I wanted to study the relationship between Age and Hours per day of music listened, in order to understand if there is 
a correlation between these two attributes. I expected to find some result, because I believe that the hours of music listened per day 
somehow depends on the age on the individual. The following graph represents the behaviour of these two attributes in the available data.""")
#plot of age and hours per day of music listened
fig, axs = plt.subplots()
plt.scatter(music_health_df["Age"], music_health_df["Hours_per_day"], c = "darkmagenta", marker = "*")
plt.xlabel("Age")
plt.ylabel("Hours per day")
plt.title("Age and Hours of music per day")
st.pyplot(fig)

#clustering with KMeans method
st.write("""I thought that I could apply the KMeans method, in order to find if these two attributes were characterized by some clusters.
For understanding the appropriate number of clusters, I used the Elbow method. The idea of this technique is to compare the value of the sum 
of squared distances of samples to their closest cluster center (the so called inertia).""")
square_distances = []
x = music_health_df[["Age", "Hours_per_day"]] #select only the two columns I am interested in
for i in range(1,11):
  km = KMeans(n_clusters = i, n_init = "auto", random_state = 42) #apply KMeans method for a number of clusters that varies from 1 to 10
  km.fit(x)
  square_distances.append(km.inertia_) #create a list of the values of inertia for each number of cluster

fig, axs = plt.subplots() #plot the relationship between the number of clusters and the correspondent value of inertia
plt.plot(range(1,11), square_distances, "rx-")
plt.xlabel("Number of clusters")
plt.ylabel("Inertia")
plt.title("Inertia per number of clusters")
plt.xticks(list(range(1,11)))
st.pyplot(fig)

st.write("""The appropriate number of clusters is given by the value that correspont to the last elbow: in my case this value was 3. So,
I needed to search for 3 clusters.""")
km = KMeans(n_clusters = 3, n_init = "auto", random_state = 42)
y_pred = km.fit_predict(x)

labels = ["Cluster 1", "Cluster 2", "Cluster 3"]
colors = ["mediumorchid", "violet", "rebeccapurple"]
markers = ["+", "d", "."]
fig, axs = plt.subplots() #plot with division in 3 clusters
for i in range(3):
  plt.scatter(x.loc[y_pred == i, "Age"], x.loc[y_pred == i, "Hours_per_day"], label = labels[i], c = colors[i], marker = markers[i])
plt.xlabel("Age")
plt.ylabel("Hours per day")
plt.title("Age and Hours of music per day")
plt.legend()
st.pyplot(fig)

st.write("""The above plot represents the clusters that I found with the KMeans method, each one pictured with a different color and a 
different marker. I noticed that the second cluster is more separated from the others, while the first and the third ones are very close
to each other. This holds particularly for the values around an age of about 28 years old. So, the first and the third cluster are not so separated: 
if we changed the label of the points around the age of 28, then the result would be pretty much the same. The squared distance will change, 
but its value will be almost equivalent (the change is very smooth).""")


st.subheader("Age and music effects")
st.write("""Then I used another histogram, in order to represent the relationship between age and music effects. The columns represent the frequence
of each effect, with a different column for each age interval.""")

fig, ax = plt.subplots(figsize = (10,8))
ax.hist([music_health_df.loc[music_health_df["Age_group"] == "under 20","Music_effects"], music_health_df.loc[music_health_df["Age_group"] == "21-40","Music_effects"],
music_health_df.loc[music_health_df["Age_group"] == "41-60","Music_effects"], music_health_df.loc[music_health_df["Age_group"] == "over 60","Music_effects"]], 
bins = 3, label = ["under 20", "21-40", "41-60", "over 60"], width = 0.15, color = sns.color_palette("rocket")[-1:-5:-1])
plt.xticks([0.3,1,1.7], ["No effect", "Improve", "Worsen"])
plt.xlabel("Music effects")
plt.ylabel("Counts")
plt.title("Music effects related to age")
plt.legend()
st.pyplot(fig)
st.write("""The chart shows that in the majority of the cases, listening to music improve the situation of the patients. No negative effect 
was registered for patients over 40, the majority of which had positive effects, improving their conditions. In correspondence of all the 
three labels, the highest columns are the ones related to the youngest people: under 20 and 21-40. This result is due to the fact that these 
two categories are the most represented in the available data, as we can see from the following table:""")
st.write(music_health_df["Age_group"].value_counts()) #evaluation of the frequence of each value of age group

st.subheader("Random Forest Classifier")
st.write("""I applied a Random Forest Classifier model, in order to predict the music effects on patients. I found the best model by changing 
the random state value and comparing the correspondent accurancy obtained:""")

y_data = music_health_df["Music_effects"]
x_data = music_health_df.drop(["Music_effects", "Fav_genre", "Age_group"], axis = 1)

accuracies = [] #list of accuracies tested with differend random states
model = RandomForestClassifier()
for random_state in [1, 23, 42, 15, 56]:
    x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.2, random_state=random_state)
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)
    accuracies.append(accuracy_score(y_pred, y_test))
st.table(accuracies)
st.write("""It is evident that the one indicated by the value 2 has the highest accuracy. For this reason, this is the one I selected and used for the prediction 
(the correspondent random state was 42). This value of accuracy tells me that the prediction the model is exact in around 80% of the cases.\\
Here is the prediction of the music effects (where 0 = no effect, 1 = improve, 2 = worsen):""")

x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.2, random_state=42)
model.fit(x_train, y_train)
y_pred = model.predict(x_test)
if st.button("Click here to see the predicted values"):
  st.table(y_pred)
st.write("""As we can see from the predicted results, no patient was predicted to have negative effects (this is probably due to the fact
that we have only a few cases of negative effects in our data). The prediction assigned to the majority of people is the label 1 - Improve, 
while only a few of them have a predicted no effect, as we can see from the following table:""")
st.table(np.unique(y_pred, return_counts = True)) #how many values are predicted for each label?

st.subheader("Principal Component Analysis")
st.write("""The aim of this section is to reduce the dimensions of my dataset, in order to discover if it is possible to better understand and 
explain the behaviour of the Music effects columns, that is the target of this analysis. \\
First of all, I normalized the dataset (considering only the numerical columns):""")
x = music_health_df.drop(["Music_effects", "Fav_genre", "Age_group"], axis=1) #drop the not numerical columns and the target column
x = StandardScaler().fit_transform(x) #normalization of the dataset
col = music_health_df.columns
col = col.drop(["Music_effects", "Fav_genre", "Age_group"])
normalized_music_health_df = pd.DataFrame(x, columns=[i + " normalized" for i in col])
st.write(normalized_music_health_df.head()) #normalized dataset

st.write("I decided to start from 20 Principal Components and I evaluated the cumulative explained variance:")
#application of PCA
n_components = 20
pca = PCA(n_components = n_components)
principal_components = pca.fit_transform(x)
principal_music_health_df = pd.DataFrame(principal_components, columns = ['PC_' + str(x + 1) for x in range(n_components)])
st.write(principal_music_health_df.head())
sum_variance = []
for i in range(1, 21):
    sum_variance.append(sum(pca.explained_variance_ratio_[:i])) #cumulative variance

st.write("Cumulative explained variance: ")
st.write(sum_variance)
#graph that represents the cumulative variance explained by the different numbers of components
fig, axs = plt.subplots()
plt.plot(range(1, 21), sum_variance, "rx-")
plt.xlabel("Number of PC")
plt.ylabel("Cumulative Explained Variance")
plt.xticks(list(range(1, 21)))
st.pyplot(fig)
st.write("""It is evident that each component gives a very low contribution to the cumulative explained variance. So, it is necessary 
to select a relative high number of principal component to describe the data. For example, by selecting 12 components, we can obtain 
a good level of cumulative explained variance: just above 70%.""")
st.write("""I wanted to analyze the situation obtained by selecting only 2 components, in order to make it possible to represents the data in a two dimensional plot:""")
#what happens with only two components
n_components = 2
pca = PCA(n_components = n_components)
principal_components = pca.fit_transform(x)
principal_music_health_df = pd.DataFrame(principal_components, columns = ["PC_" + str(x+1) for x in range(n_components)])

labels = ["No effects", "Improve", "Worsen"]
colors = ["b", "lightgreen", "deeppink"]
plt.figure(figsize = (10,6))
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
targets = [0, 1, 2]
#scatter plot
fig, ax = plt.subplots() 
for t in targets:
  plt.scatter(principal_music_health_df.loc[music_health_df["Music_effects"] == t, "PC_1"], 
  principal_music_health_df.loc[music_health_df["Music_effects"] == t, "PC_2"], label = labels[t], color = colors[t])
plt.legend()
st.pyplot(fig)

st.write("""We can immediatly notice that the three clusters, corresponding to the labels of music effects, overlap in the plot, making
the graph very confused. As I found before, two components explain only about 23% of the variance. This is a very low level, that do not 
allow to create separated clusters according to the music effect. For this reason, it is necessary to increase the number of Principal 
Components needed to represent the data. The problem with an increase in dimension is that then it is not possible to visually represent the
data in a plot. However, selecting for example 12 principal components is a good solution, allowing to reduce the dimensions of the dataset 
of 14 units (as the dataset with only numerical columns has 26 columns).""")

st.subheader("Groupby")
st.write("""In this step, I created a subset of my dataset, keeping only the columns I was interested in (the ones that I've not analyzed yet). 
Then, I decided to use the groupby function in order to group the dataset according to music effects, and apply the mean. The result is the following dataset:""")
new_dataset = music_health_df.copy()
#creation of a subset of the dataset, with only the columns I'm interested in
new_dataset.drop(music_health_df.columns[7:27], axis = 1, inplace = True)
new_dataset.drop(music_health_df.columns[-1], axis = 1, inplace = True)
new_dataset.drop(music_health_df.columns[5], axis = 1, inplace = True)
data_groupby_effects_mean  = new_dataset.groupby(["Music_effects"]).mean() #group the new dataset by Music effects and make the mean
data_groupby_effects_mean["Music_effects"] = [0, 1, 2]
st.write(data_groupby_effects_mean)

st.write("""The values in the columns While_working, Instrumentalist, Composer, Exploratory represent the percentage of individuals that 
answered positively to the question relative to the corresponsing action (that is because I set the values of the columns equal to 1 
if the answer was Yes and 0 if the answer was no). \\
My idea was to create a set of bar charts to represent the relationship between music effects and the mean of the selected columns. The 
scope was to understand if a particular behaviour could be detected in these data.""")
#creation of bar chart for every column from 1 to 6
tab1, tab2, tab3, tab4, tab5 = st.tabs([x for x in data_groupby_effects_mean.columns[1:6]])
tabs = [tab1, tab2, tab3, tab4, tab5]
n = 0
for i in data_groupby_effects_mean.columns[1:6]:
    fig, ax = plt.subplots()
    plt.bar(data_groupby_effects_mean["Music_effects"].index, data_groupby_effects_mean[i], width = 0.4, color = "c", edgecolor = "black")
    plt.xlabel("Music effect")
    plt.ylabel("Mean")
    tabs[n].markdown(i)
    plt.xticks([0, 1, 2], ["No effect", "Improve", "Worsen"])
    tabs[n].pyplot(fig)
    n = n+1

st.write("""By analyzing the charts, I understood that, in all the graphs, the central column is always the highest. This gives the idea
that people with positive and improved effects are associated with a higher mean. \\
Individuals with improved effects due to music tend to listen daily more music than the others. They are followed by people without any effect, that 
seem to have a similar value. The label worsen is characterized by a decrease in the hours of listening, of around 1 hour per day. \\
All the other graphs have a similar behaviour: the highest column is the one corresponding to an improve in the condition of the individuals, while
the other two column are shortest, with very similar values.""")
st.write("""In conclusion, the more a person listen to music, even while working, and the more he/she is in an instrumentalist, a composer 
or an exploratory person, then the higher is the likelihood that listening to music will have a positive effect on this individual, improving
his/her condition in terms of levels of Anxiety, Depression, Insomnia and OCD.""")
