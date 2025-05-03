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
def media_datos(Home_Team, Away_Team):
    teams_data = sim[(sim["HomeTeam"] == Home_Team) & (sim["AwayTeam"] == Away_Team)]
    return pd.DataFrame(
        {
            "H Shots": [teams_data["H Shots"].mean()],
            "A Shots": [teams_data["A Shots"].mean()],
            "H SOT": [teams_data["H SOT"].mean()],
            "A SOT": [teams_data["A SOT"].mean()],
            "H Corners": [teams_data["H Corners"].mean()],
            "A Corners": [teams_data["A Corners"].mean()],
            "H Yellow": [teams_data["H Yellow"].mean()],
            "A Yellow": [teams_data["A Yellow"].mean()],
            "H Red": [teams_data["H Red"].mean()],
            "A Red": [teams_data["A Red"].mean()],
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
