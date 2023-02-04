import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
music_health_df = pd.read_csv("Music & Mental Health.csv")
st.title("Music & Mental Health Project")
st.write("The selected Dataset contains data about...")
st.header("Exploration of the Dataset")
st.write("I started the exploration of the Dataset, simply by printing the first rows and using the info() function. First of all I decided to rename the columns in order to delate spaces between words and to have a better explanation of each attribute. I noticed that the first column was irrelevant for the purpose of the project, so I dropped it. I inspected the last column, finding out that it contained a unique value for all the rows. For this reason, I found it meaningless, so I dropped it.")
st.write(music_health_df.head())