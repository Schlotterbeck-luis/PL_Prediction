import streamlit as st
import math
import pandas as pd
import numpy as np
import pickle

model_file = open("model_pl.pkl", "rb")
model = pickle.loads(model_file.read())
model_file.close()

datos = pd.read_csv("England CSV.csv")
datos["HomeTeam"] = datos["HomeTeam"].apply(
    lambda x: "Brighton & Hove Albion" if x == "Brighton" else x
)
datos["AwayTeam"] = datos["AwayTeam"].apply(
    lambda x: "Brighton & Hove Albion" if x == "Brighton" else x
)

datos["HomeTeam"] = datos["HomeTeam"].apply(
    lambda x: "Ipswich Town" if x == "Ipswich" else x
)
datos["AwayTeam"] = datos["AwayTeam"].apply(
    lambda x: "Ipswich Town" if x == "Ipswich" else x
)
sim = datos.drop(
    columns=[
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
)
seasons_to_keep = ["2022/23", "2023/24", "2024/25"]
seasons_to_test = ["2024/25"]
sim = sim[sim["Season"].isin(seasons_to_keep)]
test = datos.drop(
    columns=[
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
)
test = test[test["Season"].isin(seasons_to_test)]


def media_datos(Home_Team, Away_Team):
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
        }
    )


home_team = st.selectbox("Select your home team:", test["HomeTeam"].unique())
away_team = st.selectbox("Select your away team:", test["AwayTeam"].unique())

Simulacion = media_datos(home_team, away_team)
y_predict = model.predict(np.log1p(Simulacion))


def prediccion_partido(y_predict):
    home_goals = math.expm1(y_predict[0][0])
    away_goals = math.expm1(y_predict[0][1])

    if home_goals > 4.5:
        home_goals = 4.5
    elif home_goals > 3.5:
        home_goals = 3.5
    elif home_goals > 2.5:
        home_goals = 2.5
    elif home_goals > 1.5:
        home_goals = 1.5
    elif home_goals > 0.5:
        home_goals = 0.5
    elif home_goals < 0.5:
        home_goals = 0

    if away_goals > 4.5:
        away_goals = 4.5
    elif away_goals > 3.5:
        away_goals = 3.5
    elif away_goals > 2.5:
        away_goals = 2.5
    elif away_goals > 1.5:
        away_goals = 1.5
    elif away_goals > 0.5:
        away_goals = 0.5
    elif away_goals < 0.5:
        away_goals = 0

    total_goals = home_goals + away_goals
    if home_goals >= 0.5:
        st.write(f"**Predicted goals Home team : OVER** {home_goals:.2f}")
    else:
        st.write(f"**Predicted goals Home team : UNDER** {0.50}")
    if away_goals >= 0.5:

        st.write(f"**Predicted goals Away team: OVER** {away_goals:.2f}")
    else:
        st.write(f"**Predicted goals Away team: UNDER** {0.50}")

    if total_goals == 4:
        total_goals = 4.5
    elif total_goals == 3:
        total_goals = 3.5
    elif total_goals == 2:
        total_goals = 2.5
    elif total_goals == 1:
        total_goals = 1.5
    elif home_goals < 0.5 and away_goals < 0.5:
        total_goals = 0
    if total_goals == 0:
        st.write(f"**Predicted Total goals: UNDER** {0.50}")
    else:
        st.write(f"**Predicted Total goals: OVER** {total_goals:.2f}")
    st.write("### Average Match Data:")
    st.dataframe(Simulacion)

    st.write("DISCLAIMER: Not full data for team Ipswich Town.")


prediccion_partido(y_predict)
