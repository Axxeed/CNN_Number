
import pandas as pd
import streamlit as st
from model import *

st.set_page_config(layout="wide")

st.header("Prediction MNIST")
sample = sample()
first_pred(sample)
