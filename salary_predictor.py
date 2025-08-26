import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression

st.set_page_config(page_title="Salary Predictor", page_icon="💼", layout="wide")
st.title("💼Salary Intelligence Dashboard")

@st.cache_data
def load_data():
    df = pd.read_csv("salary_dataset.csv")
    df.columns = df.columns.str.strip()
    return df
df = load_data()
st.sidebar.header("⚙️ Controls")

company_col = next((col for col in ["Company", "Company Name", "Employer"] if col in df.columns), None)
employment_col = next((col for col in ["Employment Type", "Employment", "Employment_Type"] if col in df.columns), None)

filtered_df = df.copy()
if company_col:
    companies = st.sidebar.multiselect("Select Company(s)", sorted(df[company_col].dropna().unique()))
    if companies:
        filtered_df = filtered_df[filtered_df[company_col].isin(companies)]

if employment_col:
    emps = st.sidebar.multiselect("Select Employment Type(s)", sorted(df[employment_col].dropna().unique()))
    if emps:
        filtered_df = filtered_df[filtered_df[employment_col].isin(emps)]

col1, col2, col3 = st.columns(3)
col1.metric("Total Records", len(filtered_df))
col2.metric("Average Salary", f"₹{filtered_df['Salary'].mean():,.0f}")
col3.metric("Highest Salary", f"₹{filtered_df['Salary'].max():,.0f}")

tab1, tab2, tab3 = st.tabs(["📊 Analysis", "🎯 Prediction", "📂 Data Explorer"])

with tab1:
    st.subheader("Salary Distribution")
    fig = px.histogram(filtered_df, x='Salary', nbins=30, color_discrete_sequence=["#00B4D8"])
    st.plotly_chart(fig, use_container_width=True)

    if company_col:
        st.subheader("🏆 Top Companies by Average Salary")
        top_companies = filtered_df.groupby(company_col)['Salary'].mean().sort_values(ascending=False).head(10).reset_index()
        fig = px.bar(top_companies, x=company_col, y='Salary', text='Salary', color='Salary', color_continuous_scale='Turbo')
        fig.update_traces(texttemplate='₹%{text:,.0f}', textposition='outside')
        st.plotly_chart(fig, use_container_width=True)

    if employment_col:
        st.subheader("💼 Average Salary by Employment Type")
        emp_salary = filtered_df.groupby(employment_col)['Salary'].mean().reset_index()
        fig = px.bar(emp_salary, x=employment_col, y='Salary', text='Salary', color='Salary', color_continuous_scale='Viridis')
        fig.update_traces(texttemplate='₹%{text:,.0f}', textposition='outside')
        st.plotly_chart(fig, use_container_width=True)

    numeric_cols = filtered_df.select_dtypes(exclude=['object']).columns.tolist()
    if len(numeric_cols) > 1:
        st.subheader("📈 Correlation Heatmap")
        corr = filtered_df[numeric_cols].corr()
        fig = px.imshow(corr, text_auto=True, color_continuous_scale='RdBu', zmin=-1, zmax=1)
        st.plotly_chart(fig, use_container_width=True)
with tab2:
    st.subheader("Enter Details to Predict Salary")

    target = 'Salary'
    features = [col for col in df.columns if col != target and col.lower() != 'salaries reported']
    X = df[features]
    y = df[target]

    categorical_cols = X.select_dtypes(include=['object']).columns.tolist()
    numeric_cols = X.select_dtypes(exclude=['object']).columns.tolist()

    preprocessor = ColumnTransformer([
        ('num', 'passthrough', numeric_cols),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols)
    ])

    model = Pipeline([
        ('preprocessor', preprocessor),
        ('regressor', LinearRegression())
    ])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model.fit(X_train, y_train)

    st.write(f"**Model R² Score:** {model.score(X_test, y_test):.2f}")

    input_data = []
    for col in features:
        if col in numeric_cols:
            val = st.number_input(col, value=float(df[col].mean()))
        else:
            options = sorted(df[col].dropna().astype(str).unique())
            val = st.selectbox(col, options)
        input_data.append(val)

    if st.button("Predict"):
        input_df = pd.DataFrame([input_data], columns=features)
        salary = model.predict(input_df)[0]
        st.success(f"💰 Estimated Salary: ₹{salary:,.0f}")
        st.info(f"📊 Dataset Average Salary: ₹{df['Salary'].mean():,.0f}")

with tab3:
    st.subheader("Interactive Data Table")
    st.data_editor(filtered_df, use_container_width=True, height=500)
    st.download_button("Download Filtered Data as CSV",
                       filtered_df.to_csv(index=False).encode('utf-8'),
                       file_name="filtered_salary_data.csv",
                       mime="text/csv")
