import streamlit as st 
import pandas as pd 
import matplotlib.pylab as plt
from tensorflow.keras.models import load_model

model = load_model('saved_email_classification')

st.write(model.predict("""Subject: Congratulations! You’ve Won a Gift 🎁

Body:
Dear User,

You have been selected as the lucky winner of our $500 gift card!
Click the link below to claim your reward:

👉 Claim Your Gift Now

Hurry! This offer is valid for 24 hours only.

Best regards,
Rewards Team"""))

