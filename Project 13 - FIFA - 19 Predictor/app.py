import streamlit as st
import pandas as pd
import numpy as np
import pickle
import warnings

warnings.simplefilter("ignore")

with open("Finalized-Model.pickle", "rb") as f:
    model = pickle.load(f)

with open("scaler_X.pickle", "rb") as f:
    scaler_X = pickle.load(f)

with open("scaler_y.pickle", "rb") as f:
    scaler_y = pickle.load(f)

X_columns = ['Age', 'Overall', 'Potential', 'International Reputation', 'Weak Foot', 
             'Skill Moves', 'Jersey Number', 'Contract Valid Until', 'Weight', 
             'Crossing', 'Finishing', 'HeadingAccuracy', 'ShortPassing', 'Dribbling', 
             'LongPassing', 'BallControl', 'Acceleration', 'SprintSpeed', 'Agility', 
             'Reactions', 'Balance', 'ShotPower', 'Stamina', 'Strength', 'Interceptions', 
             'Positioning', 'Vision', 'Composure', 'Marking', 'StandingTackle', 
             'SlidingTackle', 'GKDiving', 'GKHandling', 'GKKicking', 'GKPositioning', 
             'GKReflexes']

clubs = ['Atlético Madrid','Chelsea','FC Barcelona','FC Bayern München','Juventus',
         'Manchester City','Manchester United','Paris Saint-Germain','Real Madrid','Tottenham Hotspur']

nationalities = ['Argentina','Belgium','Brazil','Croatia','England','France','Germany',
                 'Italy','Poland','Portugal','Slovenia','Spain','Uruguay']

preferred_foot = ['Left','Right']

st.title("Football Player Release Clause Predictor")

st.header("Enter Player Details:")

with st.form("player_form"):
    Age = st.slider("Age", min_value=15, max_value=50, value=25)
    Overall = st.slider("Overall", min_value=0, max_value=100, value=75)
    Potential = st.slider("Potential", min_value=0, max_value=100, value=80)
    IntRep = st.slider("International Reputation", min_value=0, max_value=5, value=3)
    WeakFoot = st.slider("Weak Foot", min_value=1, max_value=5, value=3)
    SkillMoves = st.slider("Skill Moves", min_value=1, max_value=5, value=3)
    JerseyNumber = st.slider("Jersey Number", min_value=1, max_value=99, value=10)
    ContractValid = st.slider("Contract Valid Until (Year)", min_value=2023, max_value=2035, value=2028)
    Weight = st.slider("Weight (kg)", min_value=50, max_value=120, value=75)
    Crossing = st.slider("Crossing", min_value=0, max_value=100, value=70)
    Finishing = st.slider("Finishing", min_value=0, max_value=100, value=70)
    HeadingAccuracy = st.slider("Heading Accuracy", min_value=0, max_value=100, value=65)
    ShortPassing = st.slider("Short Passing", min_value=0, max_value=100, value=75)
    Dribbling = st.slider("Dribbling", min_value=0, max_value=100, value=80)
    LongPassing = st.slider("Long Passing", min_value=0, max_value=100, value=70)
    BallControl = st.slider("Ball Control", min_value=0, max_value=100, value=80)
    Acceleration = st.slider("Acceleration", min_value=0, max_value=100, value=75)
    SprintSpeed = st.slider("Sprint Speed", min_value=0, max_value=100, value=75)
    Agility = st.slider("Agility", min_value=0, max_value=100, value=75)
    Reactions = st.slider("Reactions", min_value=0, max_value=100, value=75)
    Balance = st.slider("Balance", min_value=0, max_value=100, value=70)
    ShotPower = st.slider("Shot Power", min_value=0, max_value=100, value=75)
    Stamina = st.slider("Stamina", min_value=0, max_value=100, value=80)
    Strength = st.slider("Strength", min_value=0, max_value=100, value=75)
    Interceptions = st.slider("Interceptions", min_value=0, max_value=100, value=65)
    Positioning = st.slider("Positioning", min_value=0, max_value=100, value=70)
    Vision = st.slider("Vision", min_value=0, max_value=100, value=75)
    Composure = st.slider("Composure", min_value=0, max_value=100, value=70)
    Marking = st.slider("Marking", min_value=0, max_value=100, value=60)
    StandingTackle = st.slider("Standing Tackle", min_value=0, max_value=100, value=65)
    SlidingTackle = st.slider("Sliding Tackle", min_value=0, max_value=100, value=60)
    GKDiving = st.slider("GK Diving", min_value=0, max_value=100, value=50)
    GKHandling = st.slider("GK Handling", min_value=0, max_value=100, value=50)
    GKKicking = st.slider("GK Kicking", min_value=0, max_value=100, value=50)
    GKPositioning = st.slider("GK Positioning", min_value=0, max_value=100, value=50)
    GKReflexes = st.slider("GK Reflexes", min_value=0, max_value=100, value=50)
    
    Club = st.selectbox("Club", clubs)
    Nationality = st.selectbox("Nationality", nationalities)
    PreferredFoot = st.selectbox("Preferred Foot", preferred_foot)
    
    submit = st.form_submit_button("Predict Release Clause")

if submit:
    input_df = pd.DataFrame([{
        'Age': Age, 'Overall': Overall, 'Potential': Potential, 'International Reputation': IntRep,
        'Weak Foot': WeakFoot, 'Skill Moves': SkillMoves, 'Jersey Number': JerseyNumber,
        'Contract Valid Until': ContractValid, 'Weight': Weight, 'Crossing': Crossing,
        'Finishing': Finishing, 'HeadingAccuracy': HeadingAccuracy, 'ShortPassing': ShortPassing,
        'Dribbling': Dribbling, 'LongPassing': LongPassing, 'BallControl': BallControl,
        'Acceleration': Acceleration, 'SprintSpeed': SprintSpeed, 'Agility': Agility,
        'Reactions': Reactions, 'Balance': Balance, 'ShotPower': ShotPower, 'Stamina': Stamina,
        'Strength': Strength, 'Interceptions': Interceptions, 'Positioning': Positioning,
        'Vision': Vision, 'Composure': Composure, 'Marking': Marking, 'StandingTackle': StandingTackle,
        'SlidingTackle': SlidingTackle, 'GKDiving': GKDiving, 'GKHandling': GKHandling,
        'GKKicking': GKKicking, 'GKPositioning': GKPositioning, 'GKReflexes': GKReflexes,
        'Club': Club, 'Nationality': Nationality, 'Preferred Foot': PreferredFoot
    }])
    
    input_df = pd.get_dummies(input_df)
    
    for col in model.feature_names_in_:
        if col not in input_df.columns:
            input_df[col] = 0
    input_df = input_df[model.feature_names_in_]
    
    input_scaled = scaler_X.transform(input_df)
    pred_scaled = model.predict(input_scaled)
    pred_y = scaler_y.inverse_transform(pred_scaled.reshape(-1,1))[0,0]
    pred_str = f"{pred_y:.2f}"
    st.success(f"Estimated Release Clause Is: **€{pred_str}M**")

