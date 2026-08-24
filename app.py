import streamlit as st
import pickle
import pandas as pd
import numpy as np

st.set_page_config(page_title="IPL Win Predictor", page_icon="🏏", layout="centered")

pipe = pickle.load(open('pipe.pkl','rb'))

teams = ['Sunrisers Hyderabad','Mumbai Indians','Royal Challengers Bangalore',
         'Kolkata Knight Riders','Kings XI Punjab','Chennai Super Kings',
         'Rajasthan Royals','Delhi Capitals']

cities = ['Hyderabad', 'Bangalore', 'Mumbai', 'Indore', 'Kolkata', 'Delhi',
       'Chandigarh', 'Jaipur', 'Chennai', 'Cape Town', 'Port Elizabeth',
       'Durban', 'Centurion', 'East London', 'Johannesburg', 'Kimberley',
       'Bloemfontein', 'Ahmedabad', 'Cuttack', 'Nagpur', 'Dharamsala',
       'Visakhapatnam', 'Pune', 'Raipur', 'Ranchi', 'Abu Dhabi',
       'Sharjah', 'Mohali', 'Bengaluru']

#css
st.markdown("""
<style>
   .main { background-color: #0E1117; }
    div[data-testid="stMetric"] { background-color: #1F2937; border: 1px solid #374151; padding: 15px; border-radius: 12px; }
   .stButton>button { background: linear-gradient(90deg, #FF4B2B, #FF416C); color: white; font-weight: bold; height: 3em; border-radius: 10px; width: 100%; border: none; font-size: 16px; }
   .stButton>button:hover { transform: scale(1.02); }
</style>
""", unsafe_allow_html=True)

st.title('🏏 IPL Win Predictor 2026')
st.caption('Live win probability using ML model trained on 1000+ matches')

#input section
col1, col2 = st.columns(2)
with col1:
    batting_team = st.selectbox('Batting Team', sorted(teams))
with col2:
    bowling_team = st.selectbox('Bowling Team', sorted(teams))

if batting_team == bowling_team:
    st.warning("Batting and Bowling team cannot be same!")

selected_city = st.selectbox('Host City', sorted(cities))
target = st.number_input('Target', min_value=1, max_value=300, value=180, step=1)

col3, col4, col5 = st.columns(3)
with col3:
    score = st.number_input('Current Score', min_value=0, max_value=300, value=85)
with col4:
    overs = st.number_input('Overs Completed', min_value=0.0, max_value=20.0, value=10.0, step=0.1)
with col5:
    wickets_out = st.number_input('Wickets Out', min_value=0, max_value=9, value=3)

#prediction
if st.button('Predict Probability '):
    try:
        runs_left = target - score
        balls_left = 120 - int(overs*6)
        wickets_left = 10 - wickets_out

        #calculations
        crr = score/overs if overs > 0 else 0
        rrr = (runs_left*6)/balls_left if balls_left > 0 else 0

        input_df = pd.DataFrame({
            'batting_team':[batting_team], 'bowling_team':[bowling_team], 'city':[selected_city],
            'runs_left':[runs_left], 'balls_left':[balls_left], 'wickets_left':[wickets_left],
            'total_runs_x':[target], 'crr':[crr], 'rrr':[rrr]
        })

        input_df = input_df.replace([np.inf, -np.inf], 0).fillna(0)

        result = pipe.predict_proba(input_df)
        loss = result[0][0]
        win = result[0][1]

        st.divider()
        m1, m2 = st.columns(2)
        m1.metric(batting_team, f"{round(win*100)}%", f"{int(runs_left)} runs needed")
        m2.metric(bowling_team, f"{round(loss*100)}%", f"{balls_left} balls left")

        st.progress(int(win*100), text=f"{batting_team} Winning Chance")

        if win > 0.6:
            st.success(f" 🏆 {batting_team} is FAVOURITE to win!")
        elif loss > 0.6:
            st.error(f"  🏆{bowling_team} is FAVOURITE to win!")
        else:
            st.info("⚖️ Match is Balanced - Thriller Loading!")

    except Exception as e:
        st.error(f"Enter valid Overs (e.g., 10.2 means 10 overs 2 balls). Error: {e}")