import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
import streamlit_tags as tags
music_health_df = pd.read_csv("Music & Mental Health.csv")
music_health_df.head()