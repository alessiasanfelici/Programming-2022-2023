import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

music_health_df = pd.read_csv("Music & Mental Health.csv")
music_health_df.head()
music_health_df.columns = [col.replace(" ", "_").replace("Frequency", "Freq").replace("[","").replace("]","") for col in music_health_df.columns]
music_health_df.drop(["Timestamp"], axis = 1, inplace = True)
music_health_df.Permissions.unique()
music_health_df.drop(["Permissions"], axis = 1, inplace = True)
music_health_df.drop(["BPM"], axis = 1, inplace = True)
music_health_df.drop(["Foreign_languages"], axis = 1, inplace = True)
music_health_df.drop(["Primary_streaming_service"], axis = 1, inplace = True)
music_health_df.info()
backup_dataset = music_health_df.copy() #I create a backup copy of the dataset
#I drop the rows with a null value in the column "Music_effect", because this is the column that I will predict in the correlation model. As these data are useless for computation, I delete the corresponding rows.
music_health_df.drop(music_health_df[music_health_df["Music_effects"].isnull()].index, inplace = True) 
#I drop also the rows with a null value in the following columns (because values are only yes or no)
music_health_df.drop(music_health_df[music_health_df["While_working"].isnull()].index, inplace = True)
music_health_df.drop(music_health_df[music_health_df["Instrumentalist"].isnull()].index, inplace = True)
#I replace the remaining missing value in the column "Age" with the mean of this column(rounded)
music_health_df["Age"].fillna(int(music_health_df["Age"].mean()), inplace = True)
music_health_df.info()
music_health_df.describe()
