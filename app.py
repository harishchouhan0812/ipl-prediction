import streamlit as st
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import os
import zipfile

# Define the path for the extracted data
data_path = "ipl_data"
zip_path = "archive.zip"

# Extract data if not already extracted
if not os.path.exists(data_path):
    os.makedirs(data_path)
    if os.path.exists(zip_path):
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(data_path)
    else:
        st.error("archive.zip not found. Please upload the data.")
        st.stop()


# Load data
@st.cache_data
def load_data():
    matches = pd.read_csv(os.path.join(data_path, "matches.csv"))
    deliveries = pd.read_csv(os.path.join(data_path, "deliveries.csv"))
    return matches, deliveries

matches, deliveries = load_data()

# --- Win Prediction Model ---
# Drop matches with no result
matches_clean = matches.dropna(subset=['winner']).copy()

# Encode team1, team2, and winner
le_teams = LabelEncoder()
matches_clean['team1_enc'] = le_teams.fit_transform(matches_clean['team1'])
matches_clean['team2_enc'] = le_teams.transform(matches_clean['team2'])
matches_clean['winner_enc'] = le_teams.transform(matches_clean['winner'])

# Features and Target
X_win = matches_clean[['team1_enc', 'team2_enc']]
y_win = matches_clean['winner_enc']

# Train Win Prediction Model
win_model = RandomForestClassifier()
win_model.fit(X_win, y_win)


# --- Man of the Match Prediction Model ---
# Drop null values
mom_data = matches.dropna(subset=['player_of_match']).copy()

le_mom = LabelEncoder()
mom_data['mom_enc'] = le_mom.fit_transform(mom_data['player_of_match'])

# Features (basic): team1, team2 - use the same team encoder as win prediction
mom_data['team1_enc'] = le_teams.transform(mom_data['team1'])
mom_data['team2_enc'] = le_teams.transform(mom_data['team2'])


X_mom = mom_data[['team1_enc', 'team2_enc']]
y_mom = mom_data['mom_enc']

# Train MOM Prediction Model
mom_model = RandomForestClassifier()
mom_model.fit(X_mom, y_mom)


# --- Streamlit App ---
st.title("IPL Match and MOM Predictor")

# Get unique teams for dropdowns
teams = sorted(matches['team1'].unique())

# User input for teams
team1_name = st.selectbox("Select Team 1", teams)
team2_name = st.selectbox("Select Team 2", teams)

if st.button("Predict"):
    # Encode selected teams
    try:
        team1_enc = le_teams.transform([team1_name])[0]
        team2_enc = le_teams.transform([team2_name])[0]

        # Predict Winner
        win_prediction_enc = win_model.predict([[team1_enc, team2_enc]])
        predicted_winner = le_teams.inverse_transform(win_prediction_enc)[0]

        # Predict Man of the Match
        mom_prediction_enc = mom_model.predict([[team1_enc, team2_enc]])
        predicted_mom = le_mom.inverse_transform(mom_prediction_enc)[0]


        st.subheader("Prediction Results:")
        st.write(f"🏆 Predicted Winner: {predicted_winner}")
        st.write(f"👑 Predicted Man of the Match: {predicted_mom}")

    except ValueError as e:
        st.error(f"Error encoding teams: {e}. Please select valid teams.")
