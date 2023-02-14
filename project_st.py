import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
music_health_df = pd.read_csv("Music & Mental Health.csv")
st.title("Music & Mental Health Project")
st.write("The selected Dataset contains data about...")
st.header("Exploration of the Dataset")
st.write("After having cleaned the dataset, I started exploring it, by printing the first rows, and using info() and describe() functions")
st.write(music_health_df.head())
st.write(music_health_df.info())
st.write(music_health_df.describe())