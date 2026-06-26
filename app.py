import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go
import plotly.express as px

# ── Configuración de página ──────────────────────────────────
st.set_page_config(
    page_title="HR Attrition Predictor",
    page_icon="👥",
    layout="wide"
)

# ── Estilos ──────────────────────────────────────────────────
st.markdown("""
<style>
    .metric-card {
        background: #f8f9fa;
        border-radius: 10px;
        padding: 1rem 1.2rem;
        border-left: 4px solid #534AB7;
    }
    .risk-high   { border-left-color: #E24B4A; background: #fff5f5; }
    .risk-medium { border-left-color: #EF9F27; background: #fffbf0; }
    .risk-low    { border-left-color: #1D9E75; background: #f0faf6; }
</style>
""", unsafe_allow_html=True)

# ── Carga del modelo ─────────────────────────────────────────
@st.cache_resource
def load_model():
    return joblib.load('attrition_pipeline.pkl')

model = load_model()

# ── Sidebar — perfil del empleado ───────────────────────────
st.sidebar.title("Perfil del empleado")
st.sidebar.markdown("Ingresa los datos para predecir el riesgo de renuncia.")

with st.sidebar:
    st.markdown("#### Información personal")
    age             = st.slider("Edad", 18, 60, 35)
    gender          = st.selectbox("Género", ["Male", "Female"])
    marital_status  = st.selectbox("Estado civil", ["Single", "Married", "Divorced"])
    education       = st.selectbox("Nivel educativo", [1, 2, 3, 4, 5],
                                   format_func=lambda x: {
                                       1: "Secundaria", 2: "Técnico",
                                       3: "Universitario", 4: "Maestría", 5: "Doctorado"
                                   }[x])
    education_field = st.selectbox("Campo educativo",
                                   ["Life Sciences", "Medical", "Marketing",
                                    "Technical Degree", "Human Resources", "Other"])

    st.markdown("#### Puesto y trayectoria")
    department          = st.selectbox("Departamento",
                                       ["Sales", "Research & Development", "Human Resources"])
    job_role            = st.selectbox("Rol", [
                              "Sales Executive", "Research Scientist", "Laboratory Technician",
                              "Manufacturing Director", "Healthcare Representative",
                              "Manager", "Sales Representative", "Research Director",
                              "Human Resources"])
    job_level           = st.slider("Nivel de puesto", 1, 5, 2)
    total_working_years = st.slider("Años de experiencia total", 0, 40, 8)
    years_at_company    = st.slider("Años en la empresa", 0, 40, 4)
    years_in_role       = st.slider("Años en el rol actual", 0, 18, 2)
    years_since_promo   = st.slider("Años desde última promoción", 0, 15, 1)
    num_companies       = st.slider("Empresas anteriores", 0, 9, 2)
    training_last_year  = st.slider("Capacitaciones el último año", 0, 6, 2)

    st.markdown("#### Condiciones laborales")
    overtime            = st.selectbox("Horas extra", ["Yes", "No"])
    business_travel     = st.selectbox("Viajes de trabajo",
                                       ["Non-Travel", "Travel_Rarely", "Travel_Frequently"])
    distance_home       = st.slider("Distancia al trabajo (km)", 1, 29, 5)
    monthly_income      = st.slider("Ingreso mensual (USD)", 1000, 20000, 5000, step=500)
    percent_salary_hike = st.slider("Último aumento salarial (%)", 11, 25, 14)
    stock_option_level  = st.slider("Nivel de stock options", 0, 3, 1)

    st.markdown("#### Satisfacción")
    job_satisfaction    = st.slider("Satisfacción con el trabajo", 1, 4, 3)
    environment_satisf  = st.slider("Satisfacción con el ambiente", 1, 4, 3)
    relationship_satisf = st.slider("Satisfacción con relaciones", 1, 4, 3)
    work_life_balance   = st.slider("Balance vida-trabajo", 1, 4, 3)
    job_involvement     = st.slider("Involucramiento en el trabajo", 1, 4, 3)
    performance_rating  = st.selectbox("Rating de desempeño", [3, 4],
                                       format_func=lambda x: {
                                           3: "Excellent", 4: "Outstanding"
                                       }[x])

# ── Construcción del input ───────────────────────────────────
input_data = pd.DataFrame([{
    'Age': age,
    'BusinessTravel': business_travel,
    'DailyRate': 800,
    'Department': department,
    'DistanceFromHome': distance_home,
    'Education': education,
    'EducationField': education_field,
    'EnvironmentSatisfaction': environment_satisf,
    'Gender': gender,
    'HourlyRate': 66,
    'JobInvolvement': job_involvement,
    'JobLevel': job_level,
    'JobRole': job_role,
    'JobSatisfaction': job_satisfaction,
    'MaritalStatus': marital_status,
    'MonthlyIncome': monthly_income,
    'MonthlyRate': 14000,
    'NumCompaniesWorked': num_companies,
    'OverTime': overtime,
    'PercentSalaryHike': percent_salary_hike,
    'PerformanceRating': performance_rating,
    'RelationshipSatisfaction': relationship_satisf,
    'StockOptionLevel': stock_option_level,
    'TotalWorkingYears': total_working_years,
    'TrainingTimesLastYear': training_last_year,
    'WorkLifeBalance': work_life_balance,
    'YearsAtCompany': years_at_company,
    'YearsInCurrentRole': years_in_role,
    'YearsSinceLastPromotion': years_since_promo,
    'YearsWithCurrManager': 2,
}])

# ── Predicción ───────────────────────────────────────────────
prob     = model.predict_proba(input_data)[0][1]
prob_pct = round(prob * 100, 1)

if prob_pct >= 60:
    risk_level = "Alto"
    risk_color = "#E24B4A"
    risk_css   = "risk-high"
    risk_icon  = "🔴"
elif prob_pct >= 35:
    risk_level = "Medio"
    risk_color = "#EF9F27"
    risk_css   = "risk-medium"
    risk_icon  = "🟡"
else:
    risk_level = "Bajo"
    risk_color = "#1D9E75"
    risk_css   = "risk-low"
    risk_icon  = "🟢"

# ── Layout principal ─────────────────────────────────────────
st.title("HR Attrition Predictor")
st.caption("Modelo de clasificación — IBM HR Analytics Dataset")

tab1, tab2, tab3 = st.tabs(["Predicción", "Factores de riesgo", "Exploración del dataset"])

# ── Tab 1: Predicción ────────────────────────────────────────
with tab1:
    col1, col2, col3 = st.columns([1.2, 1, 1])

    with col1:
        st.markdown(f"""
        <div class="metric-card {risk_css}">
            <div style="font-size:0.8rem;color:#888;margin-bottom:4px">Riesgo de renuncia</div>
            <div style="font-size:2.4rem;font-weight:700;color:{risk_color}">{prob_pct}%</div>
            <div style="font-size:1rem;color:{risk_color}">{risk_icon} Riesgo {risk_level}</div>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=prob_pct,
            number={'suffix': '%', 'font': {'size': 28}},
            gauge={
                'axis': {'range': [0, 100]},
                'bar': {'color': risk_color},
                'steps': [
                    {'range': [0, 35],   'color': '#f0faf6'},
                    {'range': [35, 60],  'color': '#fffbf0'},
                    {'range': [60, 100], 'color': '#fff5f5'},
                ],
                'threshold': {
                    'line': {'color': risk_color, 'width': 3},
                    'thickness': 0.75,
                    'value': prob_pct
                }
            }
        ))
        fig_gauge.update_layout(height=220, margin=dict(t=20, b=0, l=20, r=20))
        st.plotly_chart(fig_gauge, use_container_width=True)

    with col3:
        st.markdown("#### Resumen del perfil")
        st.markdown(f"""
        - **Edad:** {age} años
        - **Rol:** {job_role}
        - **Depto:** {department}
        - **Años en empresa:** {years_at_company}
        - **Ingreso:** ${monthly_income:,}
        - **Horas extra:** {overtime}
        - **Satisfacción trabajo:** {job_satisfaction}/4
        - **Balance vida-trabajo:** {work_life_balance}/4
        """)

    st.divider()
    st.markdown("#### Interpretación")
    if risk_level == "Alto":
        st.error(f"Este empleado tiene un **{prob_pct}% de probabilidad** de renunciar. "
                 "Se recomienda intervención inmediata: revisión salarial, plan de carrera "
                 "o mejora de condiciones laborales.")
    elif risk_level == "Medio":
        st.warning(f"Riesgo moderado ({prob_pct}%). Monitorear satisfacción y evaluar "
                   "oportunidades de desarrollo en los próximos 3 meses.")
    else:
        st.success(f"Riesgo bajo ({prob_pct}%). El empleado muestra señales de estabilidad "
                   "y compromiso con la organización.")

# ── Tab 2: Factores de riesgo ────────────────────────────────
with tab2:
    st.markdown("#### Importancia de variables (modelo global)")
    st.caption("Qué variables influyen más en las predicciones del modelo en general.")

    feat_names  = (model.named_steps['preprocessor']
                   .transformers_[0][2] +
                   model.named_steps['preprocessor']
                   .transformers_[1][2])
    importances = model.named_steps['classifier'].feature_importances_
    feat_df     = (pd.DataFrame({'Variable': feat_names, 'Importancia': importances})
                   .sort_values('Importancia', ascending=True)
                   .tail(15))

    fig_imp = px.bar(
        feat_df, x='Importancia', y='Variable',
        orientation='h',
        color='Importancia',
        color_continuous_scale=['#E1F5EE', '#534AB7'],
    )
    fig_imp.update_layout(
        height=420,
        showlegend=False,
        coloraxis_showscale=False,
        margin=dict(t=10, b=10)
    )
    st.plotly_chart(fig_imp, use_container_width=True)

# ── Tab 3: Exploración del dataset ───────────────────────────
with tab3:
    st.markdown("#### Dataset IBM HR Analytics")
    st.caption("1,470 empleados — distribución de variables clave")

    @st.cache_data
    def load_data():
        url = ("https://raw.githubusercontent.com/IBM/employee-attrition-aif360"
               "/master/data/emp_attrition.csv")
        df = pd.read_csv(url)
        df['Attrition_bin'] = (df['Attrition'] == 'Yes').astype(int)
        return df

    df = load_data()

    col1, col2 = st.columns(2)

    with col1:
        dept_attr = (df.groupby('Department')['Attrition_bin']
                     .mean().reset_index()
                     .rename(columns={'Attrition_bin': 'Tasa de renuncia'}))
        dept_attr['Tasa de renuncia'] = (dept_attr['Tasa de renuncia'] * 100).round(1)
        fig1 = px.bar(dept_attr, x='Department', y='Tasa de renuncia',
                      title='Tasa de renuncia por departamento (%)',
                      color='Tasa de renuncia',
                      color_continuous_scale=['#E1F5EE', '#E24B4A'])
        fig1.update_layout(height=320, showlegend=False, coloraxis_showscale=False)
        st.plotly_chart(fig1, use_container_width=True)

    with col2:
        fig2 = px.box(df, x='Attrition', y='MonthlyIncome',
                      title='Ingreso mensual vs Attrition',
                      color='Attrition',
                      color_discrete_map={'No': '#1D9E75', 'Yes': '#E24B4A'})
        fig2.update_layout(height=320, showlegend=False)
        st.plotly_chart(fig2, use_container_width=True)

    col3, col4 = st.columns(2)

    with col3:
        fig3 = px.histogram(df, x='Age', color='Attrition',
                            title='Distribución de edad por Attrition',
                            barmode='overlay', nbins=20,
                            color_discrete_map={'No': '#378ADD', 'Yes': '#E24B4A'},
                            opacity=0.75)
        fig3.update_layout(height=300)
        st.plotly_chart(fig3, use_container_width=True)

    with col4:
        ot_attr = (df.groupby('OverTime')['Attrition_bin']
                   .mean().reset_index()
                   .rename(columns={'Attrition_bin': 'Tasa'}))
        ot_attr['Tasa'] = (ot_attr['Tasa'] * 100).round(1)
        fig4 = px.bar(ot_attr, x='OverTime', y='Tasa',
                      title='Tasa de renuncia: horas extra (%)',
                      color='OverTime',
                      color_discrete_map={'No': '#1D9E75', 'Yes': '#E24B4A'})
        fig4.update_layout(height=300, showlegend=False)
        st.plotly_chart(fig4, use_container_width=True)