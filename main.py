import streamlit as st
import math
import pandas as pd
import numpy as np
import pickle

# Load model
with open("model_pl.pkl", "rb") as model_file:
    model = pickle.load(model_file)

# Load and clean data
datos = pd.read_csv("England CSV.csv")
datos["HomeTeam"] = datos["HomeTeam"].replace(
    {"Brighton": "Brighton & Hove Albion", "Ipswich": "Ipswich Town"}
)
datos["AwayTeam"] = datos["AwayTeam"].replace(
    {"Brighton": "Brighton & Hove Albion", "Ipswich": "Ipswich Town"}
)

ppm_2023 = {
    "Man City": 2.34,
    "Arsenal": 2.21,
    "Man United": 1.97,
    "Newcastle": 1.87,
    "Liverpool": 1.76,
    "Brighton & Hove Albion": 1.63,
    "Aston Villa": 1.61,
    "Tottenham": 1.58,
    "Brentford": 1.55,
    "Fulham": 1.37,
    "Crystal Palace": 1.18,
    "Chelsea": 1.16,
    "Wolves": 1.08,
    "West Ham": 1.05,
    "Bournemouth": 1.03,
    "Nott'm Forest": 1.00,
    "Everton": 0.95,
    "Leicester": 0.89,
    "Leeds": 0.82,
    "Southampton": 0.66,
}


mask_2023 = datos["Season"] == "2022/23"
datos.loc[mask_2023, "HomeTeamPPM"] = datos.loc[mask_2023, "HomeTeam"].map(ppm_2023)
datos.loc[mask_2023, "AwayTeamPPM"] = datos.loc[mask_2023, "AwayTeam"].map(ppm_2023)

# Points per game 2023/24
ppm_2024 = {
    "Man City": 2.39,
    "Arsenal": 2.34,
    "Liverpool": 2.16,
    "Aston Villa": 1.79,
    "Tottenham": 1.74,
    "Chelsea": 1.66,
    "Newcastle": 1.58,
    "Man United": 1.58,
    "West Ham": 1.37,
    "Crystal Palace": 1.29,
    "Brighton & Hove Albion": 1.26,
    "Bournemouth": 1.26,
    "Fulham": 1.24,
    "Wolves": 1.21,
    "Everton": 1.05,
    "Brentford": 1.03,
    "Nott'm Forest": 0.84,
    "Luton": 0.68,
    "Burnley": 0.63,
    "Sheffield United": 0.42,
}


mask_2024 = datos["Season"] == "2023/24"


datos.loc[mask_2024, "HomeTeamPPM"] = datos.loc[mask_2024, "HomeTeam"].map(ppm_2024)
datos.loc[mask_2024, "AwayTeamPPM"] = datos.loc[mask_2024, "AwayTeam"].map(ppm_2024)

# Points per game 2024/25
ppm_2025 = {
    "Liverpool": 2.32,
    "Arsenal": 1.97,
    "Nott'm Forest": 1.85,
    "Newcastle": 1.82,
    "Man City": 1.79,
    "Chelsea": 1.68,
    "Aston Villa": 1.65,
    "Brighton & Hove Albion": 1.62,
    "Tottenham": 1.59,
    "Man United": 1.53,
    "Fulham": 1.47,
    "Brentford": 1.41,
    "Bournemouth": 1.38,
    "West Ham": 1.32,
    "Crystal Palace": 1.24,
    "Everton": 1.15,
    "Wolves": 1.06,
    "Leicester": 0.97,
    "Ipswich Town": 0.88,
    "Southampton": 0.71,
}

# Filter for 2024/25 season
mask_2025 = datos["Season"] == "2024/25"

# Map PPM values to new columns
datos.loc[mask_2025, "HomeTeamPPM"] = datos.loc[mask_2025, "HomeTeam"].map(ppm_2025)
datos.loc[mask_2025, "AwayTeamPPM"] = datos.loc[mask_2025, "AwayTeam"].map(ppm_2025)


drop_cols = [
    "Referee",
    "H Fouls",
    "FT Result",
    "A Fouls",
    "League",
    "Display_Order",
    "HTH Goals",
    "HTA Goals",
    "HT Result",
]
sim = datos.drop(columns=drop_cols)
test = datos.drop(columns=drop_cols)

# Filter by seasons
seasons_to_keep = ["2022/23", "2023/24", "2024/25"]
seasons_to_test = ["2024/25"]
sim = sim[sim["Season"].isin(seasons_to_keep)]
test = test[test["Season"].isin(seasons_to_test)]

# Sidebar selections
st.sidebar.title("Match Selection")
home_team = st.sidebar.selectbox("Select Home Team", sorted(test["HomeTeam"].unique()))
away_team = st.sidebar.selectbox("Select Away Team", sorted(test["AwayTeam"].unique()))


# Feature generation based on historical averages
def media_datos(Home_Team, Away_Team, ppm_2025=ppm_2025):
    teams_data = sim[(sim["HomeTeam"] == Home_Team) & (sim["AwayTeam"] == Away_Team)]
    mean_H_shots = teams_data["H Shots"].mean()
    mean_A_shots = teams_data["A Shots"].mean()
    mean_H_SOT = teams_data["H SOT"].mean()
    mean_A_SOT = teams_data["A SOT"].mean()
    mean_H_corners = teams_data["H Corners"].mean()
    mean_A_corners = teams_data["A Corners"].mean()
    mean_H_yellow = teams_data["H Yellow"].mean()
    mean_A_yellow = teams_data["A Yellow"].mean()
    mean_H_red = teams_data["H Red"].mean()
    mean_A_red = teams_data["A Red"].mean()
    if Home_Team in ppm_2025:
        HPPM = ppm_2025[Home_Team]
    else:
        HPPM = 0
    if Away_Team in ppm_2025:
        APPM = ppm_2025[Away_Team]
    else:
        APPM = 0

    return pd.DataFrame(
        {
            "H Shots": [mean_H_shots],
            "A Shots": [mean_A_shots],
            "H SOT": [mean_H_SOT],
            "A SOT": [mean_A_SOT],
            "H Corners": [mean_H_corners],
            "A Corners": [mean_A_corners],
            "H Yellow": [mean_H_yellow],
            "A Yellow": [mean_A_yellow],
            "H Red": [mean_H_red],
            "A Red": [mean_A_red],
            "HomeTeamPPM": [HPPM],
            "AwayTeamPPM": [APPM],
        }
    )


# Prediction and classification
def clasificar_goles(goals):
    if goals < 0.5:
        return "UNDER 0.5"
    elif goals < 1.5:
        return "UNDER 1.5"
    elif goals < 2.5:
        return "UNDER 2.5"
    elif goals < 3.5:
        return "OVER 2.5"
    else:
        return "OVER 3.5"


def prediccion_partido(home_team, away_team):
    Simulacion = media_datos(home_team, away_team)

    if Simulacion.isnull().values.any():
        st.error("Insufficient data for this match combination.")
        return

    y_predict = model.predict(np.log1p(Simulacion))
    home_goals = math.expm1(y_predict[0][0])
    away_goals = math.expm1(y_predict[0][1])
    total_goals = home_goals + away_goals

    st.subheader("Predicted Goals")
    st.markdown(
        f"**Home team ({home_team})**: {home_goals:.2f} — *{clasificar_goles(home_goals)}*"
    )
    st.markdown(
        f"**Away team ({away_team})**: {away_goals:.2f} — *{clasificar_goles(away_goals)}*"
    )
    st.markdown(
        f"**Total goals**: {total_goals:.2f} — *{clasificar_goles(total_goals)}*"
    )

    st.subheader("Average Match Data")
    st.dataframe(Simulacion)

    if "Ipswich Town" in [home_team, away_team]:
        st.warning("DISCLAIMER: Not full data available for Ipswich Town.")


# Run prediction
prediccion_partido(home_team, away_team)
