"""
Streamlit frontend for Insurance Fraud Detection.
Connects to the FastAPI backend at localhost:8000.
Includes a single-claim prediction form, a batch CSV prediction page, and an EDA dashboard.
"""

import streamlit as st
import requests
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import io

API_URL = "http://127.0.0.1:8000"

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Insurance Fraud Detection",
    page_icon="🔍",
    layout="wide",
)

# ── Load dataset (cached) ────────────────────────────────────────────────────
@st.cache_data
def load_data():
    df = pd.read_csv("../insurance_claims.csv")
    if "_c39" in df.columns:
        df.drop("_c39", axis=1, inplace=True)
    return df

df = load_data()

def unique_sorted(column):
    """Return sorted unique values from a column, excluding '?' and NaN."""
    vals = df[column].dropna().unique()
    vals = [v for v in vals if str(v).strip() != "?"]
    return sorted(vals, key=str)

# ── Sidebar navigation ───────────────────────────────────────────────────────
st.sidebar.title("Navigation")
page = st.sidebar.radio("Page", ["Prediction", "Prediction CSV", "Dashboard"], label_visibility="collapsed")

# ══════════════════════════════════════════════════════════════════════════════
#  PREDICTION PAGE (single claim)
# ══════════════════════════════════════════════════════════════════════════════
if page == "Prediction":
    st.title("Detection de Fraude - Reclamation Individuelle")
    st.markdown("Remplissez les informations de la reclamation puis cliquez sur **Predire**.")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.subheader("Assure")
        age = st.slider("Age", 18, 80, 39)
        insured_sex = st.selectbox("Sexe", ["MALE", "FEMALE"])
        insured_education_level = st.selectbox(
            "Niveau d'education",
            ["JD", "High School", "College", "Masters", "Associate", "MD", "PhD"]
        )
        insured_occupation = st.selectbox(
            "Profession",
            unique_sorted("insured_occupation")
        )
        insured_hobbies = st.selectbox(
            "Hobbies",
            unique_sorted("insured_hobbies")
        )
        insured_relationship = st.selectbox(
            "Relation",
            unique_sorted("insured_relationship")
        )
        months_as_customer = st.number_input("Anciennete (mois)", 0, 500, 200)

    with col2:
        st.subheader("Police d'assurance")
        policy_state = st.selectbox("Etat de la police", unique_sorted("policy_state"))
        policy_csl = st.selectbox("CSL", ["100/300", "250/500", "500/1000"])
        policy_deductable = st.selectbox("Franchise", [500, 1000, 2000], index=1)
        policy_annual_premium = st.number_input("Prime annuelle", 0.0, 3000.0, 1200.0, step=50.0)
        umbrella_limit = st.selectbox(
            "Umbrella Limit",
            unique_sorted("umbrella_limit")
        )
        capital_gains = st.number_input("Capital gains", 0, 200000, 0, step=1000)
        capital_loss = st.number_input("Capital loss", -200000, 0, 0, step=1000)

    with col3:
        st.subheader("Incident & Reclamation")
        incident_type = st.selectbox(
            "Type d'incident",
            unique_sorted("incident_type")
        )
        collision_type = st.selectbox(
            "Type de collision",
            unique_sorted("collision_type")
        )
        incident_severity = st.selectbox(
            "Severite",
            unique_sorted("incident_severity")
        )
        authorities_contacted = st.selectbox(
            "Autorites contactees",
            unique_sorted("authorities_contacted")
        )
        incident_state = st.selectbox("Etat de l'incident", unique_sorted("incident_state"))
        incident_hour = st.slider("Heure de l'incident", 0, 23, 12)
        nb_vehicles = st.selectbox("Nb vehicules impliques", [1, 2, 3, 4], index=0)
        property_damage = st.selectbox("Dommages materiels", ["YES", "NO"])
        bodily_injuries = st.slider("Blessures corporelles", 0, 3, 1)
        witnesses = st.slider("Temoins", 0, 5, 1)
        police_report = st.selectbox("Rapport de police", ["YES", "NO"])
        total_claim = st.number_input("Montant total", 0.0, 200000.0, 52000.0, step=1000.0)
        injury_claim = st.number_input("Reclamation blessure", 0.0, 100000.0, 6500.0, step=500.0)
        property_claim = st.number_input("Reclamation propriete", 0.0, 100000.0, 7500.0, step=500.0)
        vehicle_claim = st.number_input("Reclamation vehicule", 0.0, 150000.0, 40000.0, step=1000.0)
        auto_make = st.selectbox("Marque auto", unique_sorted("auto_make"))
        auto_year = st.number_input("Annee du vehicule", 1990, 2025, 2005)

    st.markdown("---")

    if st.button("Predire", type="primary", use_container_width=True):
        payload = {
            "months_as_customer": months_as_customer,
            "age": age,
            "policy_state": policy_state,
            "policy_csl": policy_csl,
            "policy_deductable": policy_deductable,
            "policy_annual_premium": policy_annual_premium,
            "umbrella_limit": int(umbrella_limit),
            "insured_sex": insured_sex,
            "insured_education_level": insured_education_level,
            "insured_occupation": insured_occupation,
            "insured_hobbies": insured_hobbies,
            "insured_relationship": insured_relationship,
            "capital_gains": capital_gains,
            "capital_loss": capital_loss,
            "incident_type": incident_type,
            "collision_type": collision_type,
            "incident_severity": incident_severity,
            "authorities_contacted": authorities_contacted,
            "incident_state": incident_state,
            "incident_hour_of_the_day": incident_hour,
            "number_of_vehicles_involved": nb_vehicles,
            "property_damage": property_damage,
            "bodily_injuries": bodily_injuries,
            "witnesses": witnesses,
            "police_report_available": police_report,
            "total_claim_amount": total_claim,
            "injury_claim": injury_claim,
            "property_claim": property_claim,
            "vehicle_claim": vehicle_claim,
            "auto_make": auto_make,
            "auto_year": auto_year,
        }

        try:
            resp = requests.post(f"{API_URL}/predict", json=payload, timeout=10)
            if resp.status_code == 422:
                st.error(f"Erreur de validation: {resp.json()}")
                st.stop()
            resp.raise_for_status()
            result = resp.json()

            prob = result["fraud_probability"]
            label = result["label"]
            risk = result["risk_level"]

            # Result cards
            r1, r2, r3 = st.columns(3)
            r1.metric("Prediction", label)
            r2.metric("Probabilite de Fraude", f"{prob:.1%}")
            r3.metric("Niveau de Risque", risk)

            # Gauge chart
            fig_gauge = go.Figure(go.Indicator(
                mode="gauge+number",
                value=prob * 100,
                title={"text": "Risque de Fraude (%)"},
                gauge={
                    "axis": {"range": [0, 100]},
                    "bar": {"color": "#e74c3c" if prob > 0.5 else "#2ecc71"},
                    "steps": [
                        {"range": [0, 30], "color": "#d5f5e3"},
                        {"range": [30, 60], "color": "#fdebd0"},
                        {"range": [60, 100], "color": "#fadbd8"},
                    ],
                    "threshold": {"line": {"color": "black", "width": 3}, "thickness": 0.8, "value": 50},
                },
            ))
            fig_gauge.update_layout(height=300)
            st.plotly_chart(fig_gauge, use_container_width=True)

        except requests.exceptions.ConnectionError:
            st.error("Impossible de se connecter au backend. Verifiez que le serveur tourne sur localhost:8000.")
        except Exception as e:
            st.error(f"Erreur: {e}")

# ══════════════════════════════════════════════════════════════════════════════
#  PREDICTION CSV PAGE (batch)
# ══════════════════════════════════════════════════════════════════════════════
elif page == "Prediction CSV":
    st.title("Detection de Fraude - Prediction par Lot (CSV)")
    st.markdown(
        "Uploadez un fichier CSV au meme format que `insurance_claims.csv` "
        "(sans la colonne `fraud_reported`). L'API retournera les predictions pour chaque ligne."
    )

    uploaded_file = st.file_uploader("Choisir un fichier CSV", type=["csv"])

    if uploaded_file is not None:
        # Preview
        df_preview = pd.read_csv(uploaded_file)
        st.subheader(f"Apercu du fichier ({len(df_preview)} lignes)")
        st.dataframe(df_preview.head(10), use_container_width=True)

        if st.button("Lancer les predictions", type="primary", use_container_width=True):
            with st.spinner("Prediction en cours..."):
                try:
                    uploaded_file.seek(0)
                    files = {"file": (uploaded_file.name, uploaded_file.getvalue(), "text/csv")}
                    resp = requests.post(f"{API_URL}/predictCSV", files=files, timeout=60)
                    resp.raise_for_status()

                    df_result = pd.read_csv(io.StringIO(resp.text))

                    st.success(f"Predictions terminees pour {len(df_result)} reclamations !")

                    # KPIs
                    nb_fraud = (df_result["prediction"] == 1).sum()
                    nb_legit = (df_result["prediction"] == 0).sum()
                    avg_prob = df_result["fraud_probability"].mean()

                    k1, k2, k3, k4 = st.columns(4)
                    k1.metric("Total reclamations", len(df_result))
                    k2.metric("Legitimes", nb_legit)
                    k3.metric("Fraudes detectees", nb_fraud)
                    k4.metric("Probabilite moyenne", f"{avg_prob:.1%}")

                    # Result table
                    st.subheader("Resultats")
                    highlight_cols = ["prediction", "fraud_probability", "label", "risk_level"]
                    st.dataframe(
                        df_result[highlight_cols + [c for c in df_result.columns if c not in highlight_cols]],
                        use_container_width=True,
                    )

                    # Distribution of fraud probability
                    fig_hist = px.histogram(
                        df_result, x="fraud_probability", color="label",
                        color_discrete_map={"Legitimate": "#2ecc71", "Fraud": "#e74c3c"},
                        nbins=20, title="Distribution des Probabilites de Fraude",
                        barmode="overlay",
                    )
                    st.plotly_chart(fig_hist, use_container_width=True)

                    # Download button
                    csv_data = df_result.to_csv(index=False)
                    st.download_button(
                        label="Telecharger les resultats (CSV)",
                        data=csv_data,
                        file_name="predictions.csv",
                        mime="text/csv",
                        use_container_width=True,
                    )

                except requests.exceptions.ConnectionError:
                    st.error("Impossible de se connecter au backend. Verifiez que le serveur tourne sur localhost:8000.")
                except Exception as e:
                    st.error(f"Erreur: {e}")

# ══════════════════════════════════════════════════════════════════════════════
#  DASHBOARD PAGE
# ══════════════════════════════════════════════════════════════════════════════
else:
    st.title("Dashboard - Analyse des Reclamations d'Assurance")

    # ── KPIs ──────────────────────────────────────────────────────────────────
    total = len(df)
    nb_fraud = (df["fraud_reported"] == "Y").sum()
    nb_legit = total - nb_fraud
    fraud_rate = nb_fraud / total
    avg_claim = df["total_claim_amount"].mean()

    k1, k2, k3, k4, k5 = st.columns(5)
    k1.metric("Total Reclamations", f"{total:,}")
    k2.metric("Legitimes", f"{nb_legit:,}")
    k3.metric("Frauduleuses", f"{nb_fraud:,}")
    k4.metric("Taux de Fraude", f"{fraud_rate:.1%}")
    k5.metric("Montant Moyen", f"${avg_claim:,.0f}")

    st.markdown("---")

    # ── Row 1: Fraud distribution + Incident Type ────────────────────────────
    c1, c2 = st.columns(2)

    with c1:
        fraud_counts = df["fraud_reported"].value_counts()
        fig_pie = px.pie(
            values=fraud_counts.values,
            names=fraud_counts.index.map({"N": "Legitime", "Y": "Fraude"}),
            color_discrete_sequence=["#2ecc71", "#e74c3c"],
            title="Repartition Fraude / Legitime",
            hole=0.4,
        )
        st.plotly_chart(fig_pie, use_container_width=True)

    with c2:
        inc_fraud = df.groupby("incident_type")["fraud_reported"].apply(
            lambda x: (x == "Y").mean()
        ).reset_index()
        inc_fraud.columns = ["Type d'incident", "Taux de Fraude"]
        fig_inc = px.bar(
            inc_fraud, x="Type d'incident", y="Taux de Fraude",
            color="Taux de Fraude",
            color_continuous_scale="RdYlGn_r",
            title="Taux de Fraude par Type d'Incident",
            text_auto=".1%",
        )
        fig_inc.update_layout(showlegend=False)
        st.plotly_chart(fig_inc, use_container_width=True)

    # ── Row 2: Severity + Authorities ────────────────────────────────────────
    c3, c4 = st.columns(2)

    with c3:
        sev_fraud = df.groupby("incident_severity")["fraud_reported"].apply(
            lambda x: (x == "Y").mean()
        ).reset_index()
        sev_fraud.columns = ["Severite", "Taux de Fraude"]
        fig_sev = px.bar(
            sev_fraud, x="Severite", y="Taux de Fraude",
            color="Severite",
            color_discrete_sequence=["#f39c12", "#e74c3c", "#9b59b6", "#3498db"],
            title="Taux de Fraude par Severite",
            text_auto=".1%",
        )
        fig_sev.update_layout(showlegend=False)
        st.plotly_chart(fig_sev, use_container_width=True)

    with c4:
        auth_fraud = df.groupby("authorities_contacted")["fraud_reported"].apply(
            lambda x: (x == "Y").mean()
        ).reset_index()
        auth_fraud.columns = ["Autorite", "Taux de Fraude"]
        fig_auth = px.bar(
            auth_fraud, x="Autorite", y="Taux de Fraude",
            color="Autorite",
            title="Taux de Fraude par Autorite Contactee",
            text_auto=".1%",
        )
        fig_auth.update_layout(showlegend=False)
        st.plotly_chart(fig_auth, use_container_width=True)

    # ── Row 3: Age + Total Claim Amount ──────────────────────────────────────
    c5, c6 = st.columns(2)

    with c5:
        fig_age = px.histogram(
            df, x="age", color="fraud_reported",
            color_discrete_map={"N": "#2ecc71", "Y": "#e74c3c"},
            barmode="overlay", nbins=30,
            title="Distribution de l'Age par Statut",
            labels={"fraud_reported": "Fraude"},
        )
        fig_age.update_layout(bargap=0.1)
        st.plotly_chart(fig_age, use_container_width=True)

    with c6:
        fig_claim = px.box(
            df, x="fraud_reported", y="total_claim_amount",
            color="fraud_reported",
            color_discrete_map={"N": "#2ecc71", "Y": "#e74c3c"},
            title="Montant Total par Statut de Fraude",
            labels={"fraud_reported": "Fraude", "total_claim_amount": "Montant Total"},
        )
        fig_claim.update_layout(showlegend=False)
        st.plotly_chart(fig_claim, use_container_width=True)

    # ── Row 4: Hobbies + Education ───────────────────────────────────────────
    c7, c8 = st.columns(2)

    with c7:
        hobby_fraud = df.groupby("insured_hobbies")["fraud_reported"].apply(
            lambda x: (x == "Y").mean()
        ).sort_values(ascending=False).reset_index()
        hobby_fraud.columns = ["Hobby", "Taux de Fraude"]
        fig_hobby = px.bar(
            hobby_fraud, x="Taux de Fraude", y="Hobby",
            orientation="h",
            color="Taux de Fraude",
            color_continuous_scale="RdYlGn_r",
            title="Taux de Fraude par Hobby",
            text_auto=".1%",
        )
        fig_hobby.update_layout(height=500, showlegend=False)
        st.plotly_chart(fig_hobby, use_container_width=True)

    with c8:
        edu_fraud = df.groupby("insured_education_level")["fraud_reported"].apply(
            lambda x: (x == "Y").mean()
        ).sort_values(ascending=False).reset_index()
        edu_fraud.columns = ["Education", "Taux de Fraude"]
        fig_edu = px.bar(
            edu_fraud, x="Taux de Fraude", y="Education",
            orientation="h",
            color="Taux de Fraude",
            color_continuous_scale="RdYlGn_r",
            title="Taux de Fraude par Niveau d'Education",
            text_auto=".1%",
        )
        fig_edu.update_layout(height=500, showlegend=False)
        st.plotly_chart(fig_edu, use_container_width=True)

    # ── Row 5: Claim breakdown + Police report ───────────────────────────────
    c9, c10 = st.columns(2)

    with c9:
        # Stacked bar: injury, property, vehicle claims by fraud status
        claim_cols = ["injury_claim", "property_claim", "vehicle_claim"]
        claim_means = df.groupby("fraud_reported")[claim_cols].mean().T
        claim_means.columns = ["Legitime", "Fraude"]
        fig_claims = go.Figure()
        fig_claims.add_trace(go.Bar(name="Legitime", x=claim_means.index, y=claim_means["Legitime"],
                                     marker_color="#2ecc71"))
        fig_claims.add_trace(go.Bar(name="Fraude", x=claim_means.index, y=claim_means["Fraude"],
                                     marker_color="#e74c3c"))
        fig_claims.update_layout(
            title="Montant Moyen par Type de Reclamation",
            barmode="group", yaxis_title="Montant moyen ($)"
        )
        st.plotly_chart(fig_claims, use_container_width=True)

    with c10:
        police_fraud = df.groupby("police_report_available")["fraud_reported"].apply(
            lambda x: (x == "Y").mean()
        ).reset_index()
        police_fraud.columns = ["Rapport Police", "Taux de Fraude"]
        fig_police = px.bar(
            police_fraud, x="Rapport Police", y="Taux de Fraude",
            color="Rapport Police",
            color_discrete_sequence=["#e74c3c", "#3498db", "#f39c12"],
            title="Taux de Fraude selon Rapport de Police",
            text_auto=".1%",
        )
        fig_police.update_layout(showlegend=False)
        st.plotly_chart(fig_police, use_container_width=True)

    # ── Row 6: Correlation heatmap + Monthly premium vs claim ────────────────
    c11, c12 = st.columns(2)

    with c11:
        df_temp = df.copy()
        df_temp["fraud_num"] = (df_temp["fraud_reported"] == "Y").astype(int)
        numeric_cols = ["months_as_customer", "age", "policy_deductable",
                        "policy_annual_premium", "umbrella_limit", "capital-gains",
                        "capital-loss", "incident_hour_of_the_day",
                        "number_of_vehicles_involved", "bodily_injuries", "witnesses",
                        "total_claim_amount", "injury_claim", "property_claim",
                        "vehicle_claim", "auto_year", "fraud_num"]
        corr = df_temp[numeric_cols].corr()
        fig_corr = px.imshow(
            corr, text_auto=".2f", color_continuous_scale="RdBu_r",
            title="Matrice de Correlation",
            aspect="auto",
        )
        fig_corr.update_layout(height=600)
        st.plotly_chart(fig_corr, use_container_width=True)

    with c12:
        fig_scatter = px.scatter(
            df, x="policy_annual_premium", y="total_claim_amount",
            color="fraud_reported",
            color_discrete_map={"N": "#2ecc71", "Y": "#e74c3c"},
            opacity=0.5,
            title="Prime Annuelle vs Montant Total de Reclamation",
            labels={
                "policy_annual_premium": "Prime Annuelle ($)",
                "total_claim_amount": "Montant Total ($)",
                "fraud_reported": "Fraude",
            },
        )
        fig_scatter.update_layout(height=600)
        st.plotly_chart(fig_scatter, use_container_width=True)

    # ── Row 7: Policy state + Auto make ──────────────────────────────────────
    c13, c14 = st.columns(2)

    with c13:
        state_fraud = df.groupby("policy_state")["fraud_reported"].apply(
            lambda x: (x == "Y").mean()
        ).reset_index()
        state_fraud.columns = ["Etat", "Taux de Fraude"]
        fig_state = px.bar(
            state_fraud, x="Etat", y="Taux de Fraude",
            color="Etat",
            color_discrete_sequence=["#3498db", "#e67e22", "#9b59b6"],
            title="Taux de Fraude par Etat de Police",
            text_auto=".1%",
        )
        fig_state.update_layout(showlegend=False)
        st.plotly_chart(fig_state, use_container_width=True)

    with c14:
        make_fraud = df.groupby("auto_make")["fraud_reported"].apply(
            lambda x: (x == "Y").mean()
        ).sort_values(ascending=False).reset_index()
        make_fraud.columns = ["Marque", "Taux de Fraude"]
        fig_make = px.bar(
            make_fraud, x="Taux de Fraude", y="Marque",
            orientation="h",
            color="Taux de Fraude",
            color_continuous_scale="RdYlGn_r",
            title="Taux de Fraude par Marque Automobile",
            text_auto=".1%",
        )
        fig_make.update_layout(height=500, showlegend=False)
        st.plotly_chart(fig_make, use_container_width=True)
