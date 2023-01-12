import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
music_health_df = pd.read_csv("Music & Mental Health.csv")
st.title("Music & Mental Health Project")
st.header("Exploration of the Dataset")
st.write(music_health_df.head())