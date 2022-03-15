#######
import streamlit as st
import pandas as pd
import numpy as np
import subprocess
import sys
import os
import shutil
import run_xgb
from streamlit_autorefresh import st_autorefresh

if os.path.exists('crypto.csv'):
     subprocess.run([f"{sys.executable}", "src/run_xgb.py"]) 

else:
    title = st.title("ToTheMoon")
    list = []
    options=['1h', '2h', '4h', '6h', '8h', '12h', '1d', '3d', '1w', '1M']
    inters = st.select_slider(
        'Select time interval',
         options=options)
    crypto = st.text_input("Input Ticker")

    if options[:5]:
        start_str = "90 days ago UTC"
    else:
        start_str = "11 hours ago UTC"
        
    btn = st.button("Run Bot")

    if btn and (crypto != ""):
        list.append((crypto, inters, start_str))
        df = pd.DataFrame(list, columns = ['crypto', 'inter', 'start'])
        df.to_csv('crypto.csv')
        subprocess.run([f"{sys.executable}", "src/run_xgb.py"]) 
        st_autorefresh(interval=15000, limit=5000)
        