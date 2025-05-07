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

# Goles por partido en general

# 2023

gpp_2023 = {
    "Man City": 2.5,
    "Arsenal": 2.3,
    "Liverpool": 2.0,
    "Brighton & Hove Albion": 1.9,
    "Newcastle": 1.7,
    "Man United": 1.6,
    "Tottenham": 1.9,
    "Aston Villa": 1.5,
    "Brentford": 1.6,
    "Fulham": 1.5,
    "Chelsea": 1.0,
    "Crystal Palace": 1.1,
    "Wolves": 0.9,
    "West Ham": 1.1,
    "Bournemouth": 1.0,
    "Nott'm Forest": 0.9,
    "Everton": 0.9,
    "Leicester": 1.3,
    "Leeds": 1.1,
    "Southampton": 0.9,
}

mask_2023 = datos["Season"] == "2022/23"
datos.loc[mask_2023, "HomeTeamGPM"] = datos.loc[mask_2023, "HomeTeam"].map(gpp_2023)
datos.loc[mask_2023, "AwayTeamGPM"] = datos.loc[mask_2023, "AwayTeam"].map(gpp_2023)

# 2024

gpp_2024 = {
    "Man City": 2.53,
    "Arsenal": 2.39,
    "Liverpool": 2.26,
    "Newcastle": 2.24,
    "Chelsea": 2.03,
    "Aston Villa": 2.00,
    "Tottenham": 1.95,
    "West Ham": 1.58,
    "Crystal Palace": 1.50,
    "Man United": 1.50,
    "Brentford": 1.47,
    "Brighton & Hove Albion": 1.45,
    "Fulham": 1.45,
    "Bournemouth": 1.42,
    "Luton": 1.37,
    "Wolves": 1.32,
    "Nott'm Forest": 1.29,
    "Burnley": 1.08,
    "Everton": 1.05,
    "Sheffield United": 0.92,
}

mask_2024 = datos["Season"] == "2023/24"
datos.loc[mask_2024, "HomeTeamGPM"] = datos.loc[mask_2024, "HomeTeam"].map(gpp_2024)
datos.loc[mask_2024, "AwayTeamGPM"] = datos.loc[mask_2024, "AwayTeam"].map(gpp_2024)

# 2025
gpp_2025 = {
    "Liverpool": 2.31,  # 81 goles en 35 partidos
    "Man City": 1.91,  # 67 goles en 35 partidos
    "Newcastle": 1.89,  # 66 goles en 35 partidos
    "Arsenal": 1.83,  # 64 goles en 35 partidos
    "Tottenham": 1.80,  # 63 goles en 35 partidos
    "Brentford": 1.77,  # 62 goles en 35 partidos
    "Chelsea": 1.77,  # 62 goles en 35 partidos
    "Brighton & Hove Albion": 1.63,  # 57 goles en 35 partidos
    "Aston Villa": 1.57,  # 55 goles en 35 partidos
    "Bournemouth": 1.57,  # 55 goles en 35 partidos
    "West Ham": 1.51,  # 53 goles en 35 partidos
    "Crystal Palace": 1.43,  # 50 goles en 35 partidos
    "Man United": 1.40,  # 49 goles en 35 partidos
    "Fulham": 1.34,  # 47 goles en 35 partidos
    "Wolves": 1.31,  # 46 goles en 35 partidos
    "Nott'm Forest": 1.29,  # 45 goles en 35 partidos
    "Everton": 1.14,  # 40 goles en 35 partidos
    "Ipswich Town": 1.00,  # 35 goles en 35 partidos
    "Leicester": 0.83,  # 29 goles en 35 partidos
    "Southampton": 0.71,  # 25 goles en 35 partidos
}


mask_2025 = datos["Season"] == "2024/25"
datos.loc[mask_2025, "HomeTeamGPM"] = datos.loc[mask_2025, "HomeTeam"].map(gpp_2025)
datos.loc[mask_2025, "AwayTeamGPM"] = datos.loc[mask_2025, "AwayTeam"].map(gpp_2025)

# Goles encajados por partido
# 2023

gep_2023 = {
    "Man City": 0.87,
    "Newcastle": 0.87,
    "Arsenal": 1.13,
    "Man United": 1.13,
    "Brentford": 1.21,
    "Aston Villa": 1.21,
    "Liverpool": 1.24,
    "Chelsea": 1.24,
    "Crystal Palace": 1.29,
    "Brighton & Hove Albion": 1.39,
    "Fulham": 1.39,
    "West Ham": 1.45,
    "Everton": 1.50,
    "Wolves": 1.53,
    "Tottenham": 1.66,
    "Nott'm Forest": 1.79,
    "Leicester": 1.79,
    "Bournemouth": 1.87,
    "Southampton": 1.92,
    "Leeds": 2.05,
}


mask_2023 = datos["Season"] == "2022/23"
datos.loc[mask_2023, "HomeTeamGEP"] = datos.loc[mask_2023, "HomeTeam"].map(gep_2023)
datos.loc[mask_2023, "AwayTeamGEP"] = datos.loc[mask_2023, "AwayTeam"].map(gep_2023)


gep_2024 = {
    "Arsenal": 0.76,
    "Man City": 0.89,
    "Liverpool": 1.08,
    "Everton": 1.34,
    "Crystal Palace": 1.53,
    "Man United": 1.53,
    "Chelsea": 1.66,
    "Aston Villa": 1.61,
    "Tottenham": 1.61,
    "Fulham": 1.61,
    "Brighton & Hove Albion": 1.63,
    "Newcastle": 1.63,
    "Wolves": 1.71,
    "Brentford": 1.71,
    "Nott'm Forest": 1.76,
    "Bournemouth": 1.76,
    "West Ham": 1.95,
    "Burnley": 2.05,
    "Luton": 2.24,
    "Sheffield United": 2.74,
}

mask_2024 = datos["Season"] == "2023/24"
datos.loc[mask_2024, "HomeTeamGEP"] = datos.loc[mask_2024, "HomeTeam"].map(gep_2024)
datos.loc[mask_2024, "AwayTeamGEP"] = datos.loc[mask_2024, "AwayTeam"].map(gep_2024)

gep_2025 = {
    "Arsenal": 0.89,  # 31 goles en 35 partidos
    "Liverpool": 0.94,  # 32 goles en 34 partidos
    "Chelsea": 1.18,  # 40 goles en 34 partidos
    "Man City": 1.23,  # 43 goles en 35 partidos
    "Aston Villa": 1.40,  # 49 goles en 35 partidos
    "Bournemouth": 1.20,  # 42 goles en 35 partidos
    "Brentford": 1.47,  # 50 goles en 34 partidos
    "Brighton & Hove Albion": 1.62,  # 55 goles en 34 partidos
    "Tottenham": 1.65,  # 56 goles en 34 partidos
    "Newcastle": 1.29,  # 44 goles en 34 partidos
    "Man United": 1.38,  # 47 goles en 34 partidos
    "Fulham": 1.34,  # 47 goles en 35 partidos
    "Crystal Palace": 1.38,  # 47 goles en 34 partidos
    "Wolves": 1.77,  # 62 goles en 35 partidos
    "West Ham": 1.71,  # 58 goles en 34 partidos
    "Everton": 1.23,  # 43 goles en 35 partidos
    "Nott'm Forest": 1.21,  # 41 goles en 34 partidos
    "Southampton": 2.34,  # 82 goles en 35 partidos
    "Leicester": 2.17,  # 76 goles en 35 partidos
    "Ipswich Town": 2.17,  # 76 goles en 35 partidos
}

mask_2025 = datos["Season"] == "2024/25"
datos.loc[mask_2025, "HomeTeamGEP"] = datos.loc[mask_2025, "HomeTeam"].map(gep_2025)
datos.loc[mask_2025, "AwayTeamGEP"] = datos.loc[mask_2025, "AwayTeam"].map(gep_2025)

# Valores De plantillas

# 2025

valores_mercado_2025 = {
    "Liverpool": 993.5,
    "Arsenal": 1130,
    "Man City": 1310,
    "Newcastle": 635,
    "Chelsea": 922,
    "Nott'm Forest": 447.18,
    "Aston Villa": 627.5,
    "Fulham": 362.6,
    "Brighton & Hove Albion": 555.6,
    "Bournemouth": 435.35,
    "Brentford": 417.1,
    "Crystal Palace": 441,
    "Wolves": 408.8,
    "Man United": 699.25,
    "Everton": 365.1,
    "Tottenham": 836.1,
    "West Ham": 454.3,
    "Ipswich Town": 279.6,
    "Leicester": 273.3,
    "Southampton": 273.6,
}

mask_2025 = datos["Season"] == "2024/25"
datos.loc[mask_2025, "HomeTeamMV"] = datos.loc[mask_2025, "HomeTeam"].map(
    valores_mercado_2025
)
datos.loc[mask_2025, "AwayTeamMV"] = datos.loc[mask_2025, "AwayTeam"].map(
    valores_mercado_2025
)

# 2024

valores_mercado_2024 = {
    "Man City": 1460.0,
    "Arsenal": 1200.0,
    "Chelsea": 1010.0,
    "Man United": 808.35,
    "Liverpool": 955.65,
    "Tottenham": 838.50,
    "Aston Villa": 709.65,
    "Newcastle": 652.25,
    "West Ham": 487.50,
    "Nott'm Forest": 501.35,
    "Brighton & Hove Albion": 524.40,
    "Brentford": 428.78,
    "Everton": 391.75,
    "Bournemouth": 401.75,
    "Crystal Palace": 446.58,
    "Wolves": 442.73,
    "Fulham": 384.35,
    "Burnley": 273.48,
    "Sheffield United": 152.35,
    "Luton": 140.90,
}


mask_2024 = datos["Season"] == "2023/24"
datos.loc[mask_2024, "HomeTeamMV"] = datos.loc[mask_2024, "HomeTeam"].map(
    valores_mercado_2024
)
datos.loc[mask_2024, "AwayTeamMV"] = datos.loc[mask_2024, "AwayTeam"].map(
    valores_mercado_2024
)

valores_mercado_2023 = {
    "Man City": 1500.0,
    "Arsenal": 1000.0,
    "Chelsea": 995.0,
    "Man United": 847.75,
    "Liverpool": 811.85,
    "Tottenham": 649.10,
    "Aston Villa": 509.55,
    "Newcastle": 541.60,
    "West Ham": 465.60,
    "Nott'm Forest": 376.25,
    "Brighton & Hove Albion": 529.80,
    "Brentford": 371.20,
    "Everton": 413.15,
    "Bournemouth": 287.20,
    "Crystal Palace": 323.05,
    "Wolves": 497.65,
    "Fulham": 295.10,
    "Leicester": 490.70,
    "Southampton": 419.95,
    "Leeds": 345.15,
}

mask_2023 = datos["Season"] == "2022/23"
datos.loc[mask_2023, "HomeTeamMV"] = datos.loc[mask_2023, "HomeTeam"].map(
    valores_mercado_2023
)
datos.loc[mask_2023, "AwayTeamMV"] = datos.loc[mask_2023, "AwayTeam"].map(
    valores_mercado_2023
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
def media_datos(
    Home_Team,
    Away_Team,
    ppm_2025=ppm_2025,
    gpp_2025=gpp_2025,
    valores=valores_mercado_2025,
    gep_2025=gep_2025,
):
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
    if Home_Team in gpp_2025:
        HGPM = gpp_2025[Home_Team]
    else:
        HGPM = 0
    if Away_Team in gpp_2025:
        AGPM = gpp_2025[Away_Team]
    else:
        AGPM = 0
    if Home_Team in valores_mercado_2025:
        HMV = valores_mercado_2025[Home_Team]
    else:
        HMV = 0
    if Away_Team in valores_mercado_2025:
        AMV = valores_mercado_2025[Away_Team]
    else:
        AMV = 0
    if Home_Team in gep_2025:
        HGEP = gep_2025[Home_Team]
    else:
        HGEP = 0
    if Away_Team in gep_2025:
        AGEP = gep_2025[Away_Team]
    else:
        AGEP = 0

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
            "HomeTeamGPM": [HGPM],
            "AwayTeamGPM": [AGPM],
            "HomeTeamMV": [HMV],
            "AwayTeamMV": [AMV],
            "HomeTeamGEP": [HGEP],
            "AwayTeamGEP": [AGEP],
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
    Simulacion_modelo = Simulacion.drop(
        columns=["H Corners", "A Corners", "H Yellow", "A Yellow", "H Red", "A Red"]
    )

    if Simulacion_modelo.isnull().values.any():
        st.error("Insufficient data for this match combination.")
        return

    y_predict = model.predict(np.log1p(Simulacion_modelo))
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
    st.dataframe(
        Simulacion.drop(
            columns=[
                "HomeTeamPPM",
                "AwayTeamPPM",
                "HomeTeamGPM",
                "AwayTeamGPM",
                "HomeTeamMV",
                "AwayTeamMV",
                "HomeTeamGEP",
                "AwayTeamGEP",
            ]
        )
    )

    if "Ipswich Town" in [home_team, away_team]:
        st.warning("DISCLAIMER: Not full data available for Ipswich Town.")


# Run prediction
prediccion_partido(home_team, away_team)
