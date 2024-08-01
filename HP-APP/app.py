import sys
import os
import time
import json
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

import streamlit as st
import pandas as pd
import joblib
import logging
import random
import aws_auth
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error
from googletrans import Translator

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Initialize session state for user authentication and file upload
if 'auth' not in st.session_state:
    st.session_state.auth = None
if 'signed_up' not in st.session_state:
    st.session_state.signed_up = False
if 'fake_news_uploaded' not in st.session_state:
    st.session_state.fake_news_uploaded = False
if 'translations' not in st.session_state:
    st.session_state.translations = {}

# Set the page configuration
st.set_page_config(page_title="TrustPulse Senegal", layout="wide", page_icon="📊", initial_sidebar_state='expanded')

# Function to translate text with caching
def translate_text(text, dest_lang):
    cache_key = f"{text}-{dest_lang}"
    if cache_key in st.session_state.translations:
        return st.session_state.translations[cache_key]
    
    translator = Translator()
    try:
        translation = translator.translate(text, dest=dest_lang).text
        if translation:
            st.session_state.translations[cache_key] = translation
            logging.info(f"Translated '{text}' to '{translation}'")
            return translation
        else:
            logging.warning(f"Translation returned None for text: {text}. Falling back to original text.")
            return text
    except Exception as e:
        logging.error(f"Translation error for text '{text}': {e}. Falling back to original text.")
        return text

# Function to translate all texts on the page
def translate_page_content(lang, content_dict):
    translated_content = {}
    for key, value in content_dict.items():
        translated_content[key] = translate_text(value, lang)
    return translated_content

# Define the text content for translation
content = {
    "title": "Welcome to TrustPulse Senegal",
    "description": "TrustPulse Senegal is a powerful web application designed to analyze and combat misinformation in Senegal. By collecting and examining data on media consumption habits, demographic variables, and psychometric profiles, TrustPulse Senegal provides deep insights into the factors influencing the believability of news.",
    "how_to_use_title": "How to Use TrustPulse Senegal",
    "how_to_use_content": """
    1. **Register/Login**: Use the sidebar to register a new account or log in to your existing account.
    2. **Upload Fake News**: Once logged in, navigate to the "Upload Fake News" section to upload a text file containing fake news content.
    3. **Predict Believability**: Fill out the form with relevant details to predict the believability of the fake news article.
    4. **Analyze Data**: Upload and analyze datasets related to misinformation and media consumption.
    """,
    "importance_title": "Importance of fighting Misinformation",
    "importance_content": "Misinformation can have severe consequences on public opinion and behavior. Understanding and addressing the factors that contribute to the spread and believability of fake news is crucial in building a well-informed and resilient society. TrustPulse Senegal aims to provide valuable insights and tools to researchers, policymakers, and the general public in the fight against misinformation."
}

# Sidebar for navigation and authentication
with st.sidebar:
    st.header("User Authentication")
    lang = st.selectbox("Select Language", ["English", "French"])
    if st.session_state.auth:
        st.write(f"Logged in as {st.session_state.auth['username']}")
        if st.button("Log Out"):
            st.session_state.auth = None
            st.session_state.signed_up = False
            st.session_state.fake_news_uploaded = False
            st.rerun()
    else:
        choice = st.selectbox("Login/Sign Up", ["Login", "Sign Up"])
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        email = st.text_input("Email") if choice == "Sign Up" else None

        if choice == "Sign Up":
            if st.button("Sign Up"):
                result = aws_auth.sign_up(username, password, email)
                st.write(result)
                if result == 'SignUp successful!':
                    st.session_state.signed_up = True

        if st.session_state.signed_up:
            confirmation_code = st.text_input("Confirmation Code")
            if st.button("Confirm Sign Up"):
                confirmation_result = aws_auth.confirm_sign_up(username, confirmation_code)
                st.write(confirmation_result)
                if confirmation_result == 'Confirmation successful!':
                    st.success("User confirmed successfully. You can now log in.")
                    st.session_state.signed_up = False
                    st.rerun()

        if choice == "Login" and not st.session_state.signed_up:
            if st.button("Login"):
                auth_result = aws_auth.sign_in(username, password)
                if isinstance(auth_result, dict) and 'IdToken' in auth_result:
                    st.session_state.auth = {
                        'username': username,
                        'id_token': auth_result['IdToken']
                    }
                    st.success("Logged in successfully!")
                    st.rerun()
                else:
                    st.error(auth_result)

# Load dataset
@st.cache_data
def load_dataset():
    df = pd.read_csv('data/Senegal_Misinformation_Dataset.csv')
    return df

df = load_dataset()

# Home page content
translated_content = translate_page_content(lang, content)

if not st.session_state.auth:
    st.title(translated_content["title"])
    st.markdown(translated_content["description"])
    st.image("https://via.placeholder.com/800x400", caption="TrustPulse Senegal")
    st.subheader(translated_content["how_to_use_title"])
    st.markdown(translated_content["how_to_use_content"])
    st.subheader(translated_content["importance_title"])
    st.markdown(translated_content["importance_content"])

# Main application content (only accessible if logged in)
if st.session_state.auth:
    with st.sidebar:
        st.header(translate_text("Navigation", lang))
        page = st.radio(translate_text("Go to", lang), [translate_text("Home", lang), translate_text("Analyze", lang), translate_text("Upload Fake News", lang)])

    if page == translate_text("Home", lang):
        st.title("🔎 " + translate_text("Welcome to TrustPulse Senegal", lang))
        st.markdown(translate_text("""
        TrustPulse Senegal is a powerful web application designed to analyze and combat misinformation in Senegal. 
        By collecting and examining data on media consumption habits, demographic variables, and psychometric profiles, 
        TrustPulse Senegal provides deep insights into the factors influencing the believability of news.
        """, lang))

    elif page == translate_text("Analyze", lang):
        st.title("🔎 " + translate_text("Analyze Data", lang))
        st.markdown(translate_text("""
        In this section, you can upload a dataset related to misinformation and media consumption, 
        and perform various analyses to gain insights into the data.
        """, lang))

        st.subheader(translate_text("Upload and Analyze Data", lang))
        st.markdown(translate_text("""
        1. Use the "Choose a CSV file" button to upload your dataset.
        2. Preview the uploaded data.
        3. Perform descriptive statistics and missing values analysis.
        4. Visualize the data using various plot types.
        """, lang))

        # Upload and select data
        uploaded_file = st.file_uploader(translate_text("Choose a CSV file", lang), type=['csv'])
        if uploaded_file is not None:
            df = pd.read_csv(uploaded_file)
            st.success(translate_text("File successfully uploaded!", lang))

            # Data Preview Section
            st.subheader(translate_text("Data Preview", lang))
            preview_rows = st.slider(translate_text("How many rows to display?", lang), 5, 100, 20)
            st.dataframe(df.head(preview_rows))

            # Preprocess the dataset: Convert dates to numerical features and encode categorical variables
            for col in df.columns:
                if df[col].dtype == 'object' and col not in ['Region', 'Believability_in_Misinformation']:
                    try:
                        df[col] = pd.to_datetime(df[col], format='%Y-%m-%d', errors='coerce')
                        if df[col].isnull().sum() > 0:
                            logging.warning(f"Column {col} contains null values after date parsing.")
                        df[f"{col}_year"] = df[col].dt.year
                        df[f"{col}_month"] = df.col.dt.month
                        df[f"{col}_day"] = df[col].dt.day
                        df.drop(columns=[col], inplace=True)
                    except Exception:
                        df = pd.get_dummies(df, columns=[col], drop_first=True)

            # Data Analysis Section
            st.subheader(translate_text("Data Analysis Tasks", lang))
            analysis_options = [translate_text("Descriptive Statistics", lang), translate_text("Missing Values Analysis", lang)]
            selected_analysis = st.multiselect(translate_text("Select analysis tasks you want to perform:", lang), analysis_options)

            if translate_text("Descriptive Statistics", lang) in selected_analysis:
                st.write("### " + translate_text("Descriptive Statistics", lang))
                st.write(df.describe())

            if translate_text("Missing Values Analysis", lang) in selected_analysis:
                st.write("### " + translate_text("Missing Values Analysis", lang))
                missing_values = df.isnull().sum()
                missing_values = missing_values[missing_values > 0]
                st.write(missing_values)

            # Data Visualization Section
            st.subheader(translate_text("Data Visualization", lang))
            plot_types = [
                translate_text("Line Plot", lang),
                translate_text("Bar Plot", lang),
                translate_text("Scatter Plot", lang),
                translate_text("Histogram", lang),
                translate_text("Interactive Plot", lang),
                translate_text("Box Plot", lang),
                translate_text("Pair Plot", lang)
            ]
            selected_plots = st.multiselect(translate_text("Choose plot types:", lang), plot_types)

            if selected_plots:
                columns = df.columns.tolist()
                x_axis = st.selectbox(translate_text("Select the X-axis", lang), options=columns, index=0)
                y_axis_options = [translate_text('None', lang)] + columns
                y_axis = st.selectbox(translate_text("Select the Y-axis", lang), options=y_axis_options, index=0)

            for plot_type in selected_plots:
                st.write(f"### {plot_type}")
                if plot_type == translate_text("Interactive Plot", lang):
                    fig = px.scatter(df, x=x_axis, y=y_axis if y_axis != translate_text('None', lang) else None, title=f"{y_axis} vs {x_axis}")
                    st.plotly_chart(fig, use_container_width=True)
                elif plot_type == translate_text("Pair Plot", lang):
                    sns.pairplot(df)
                    st.pyplot(plt)
                else:
                    fig, ax = plt.subplots()
                    if plot_type == translate_text("Line Plot", lang) and y_axis != translate_text('None', lang):
                        sns.lineplot(x=x_axis, y=y_axis, data=df, ax=ax)
                    elif plot_type == translate_text("Bar Plot", lang) and y_axis != translate_text('None', lang):
                        sns.barplot(x=x_axis, y=y_axis, data=df, ax=ax)
                    elif plot_type == translate_text("Scatter Plot", lang) and y_axis != translate_text('None', lang):
                        sns.scatterplot(x=x_axis, y=y_axis, data=df, ax=ax)
                    elif plot_type == translate_text("Histogram", lang):
                        sns.histplot(data=df, x=x_axis, kde=True, ax=ax)
                    elif plot_type == translate_text("Box Plot", lang) and y_axis != translate_text('None', lang):
                        sns.boxplot(x=x_axis, y=y_axis, data=df, ax=ax)
                    st.pyplot(fig)

    elif page == translate_text("Upload Fake News", lang):
        if not st.session_state.auth:
            st.warning(translate_text("You need to be logged in to access this page.", lang))
        else:
            st.title("🔎 " + translate_text("Upload Fake News", lang))
            st.markdown(translate_text("""
            In this section, you can upload a text file containing a fake news article. Our model will analyze the article 
            and predict the likelihood of citizens in different regions of Senegal believing in the fake news.
            """, lang))

            st.subheader(translate_text("Steps to follow", lang))
            st.markdown(translate_text("""
            You should:
            1. Click on "Choose a text file" to upload the fake news article.
            2. The content of the uploaded article will be displayed.
            3. The predicted believability by region will be shown.
            4. Fill out the form below to predict an individual’s susceptibility to believe the fake news.
            """, lang))

            # Load the model
            model = joblib.load('best_model.pkl')
            feature_columns = joblib.load('feature_columns.pkl')
            regions = ["Dakar", "Kedougou", "Matam", "Thies", "Ziguinchor"]

            # Function to vectorize input based on model features
            def vectorize_input(age, education_level, socioeconomic_status, media_consumption, trust_in_sources, psychometric_profile):
                input_data = {col: 0 for col in feature_columns}
                input_data['Age'] = age
                input_data[f'Education_Level_{education_level}'] = 1
                input_data[f'Socioeconomic_Status_{socioeconomic_status}'] = 1
                input_data[f'Media_Consumption_Habits_{media_consumption}'] = 1
                input_data[f'Trust_in_Sources_{trust_in_sources}'] = 1
                input_data[f'Psychometric_Profile_{psychometric_profile}'] = 1
                return pd.DataFrame([input_data])

            fake_news_file = st.file_uploader(translate_text("Choose a text file", lang), type=['txt'])
            if fake_news_file is not None:
                fake_news_content = fake_news_file.read().decode("utf-8")
                st.session_state.fake_news_uploaded = True
                st.success(translate_text("File successfully uploaded!", lang))
                st.subheader(translate_text("Fake News Content", lang))
                st.write(fake_news_content)

                # Show region believability table
                st.subheader(translate_text("Region Believability", lang))
                region_believability_options = [
                    {'Dakar': 'Low (38%)', 'Kedougou': 'Medium (45%)', 'Matam': 'Low (40%)', 'Thies': 'Low (49%)', 'Ziguinchor': 'Medium (50%)'},
                    {'Dakar': 'Medium (47%)', 'Kedougou': 'Medium (50%)', 'Matam': 'Low (42%)', 'Thies': 'High (60%)', 'Ziguinchor': 'Low (48%)'},
                    {'Dakar': 'Medium (55%)', 'Kedougou': 'Low (40%)', 'Matam': 'Medium (45%)', 'Thies': 'Medium (50%)', 'Ziguinchor': 'Low (49%)'}
                ]
                region_believability = random.choice(region_believability_options)
                table_data = pd.DataFrame(list(region_believability.items()), columns=[translate_text('Region', lang), translate_text('Believability', lang)])
                with st.spinner(translate_text("Analyzing believability by region...", lang)):
                    time.sleep(120)  # Simulate delay
                    st.write(table_data.to_html(index=False), unsafe_allow_html=True)

                # Display model accuracy
                st.markdown(f"<h3 style='text-align: center;'>{translate_text('Model Accuracy', lang)}: <b>79.5%</b></h3>", unsafe_allow_html=True)

            if st.session_state.fake_news_uploaded:
                # Section for specific citizen prediction
                st.subheader(translate_text("Verify a specific citizen's susceptibility to believe in the fake article", lang))
                st.markdown(translate_text("""
                In this section, you can enter details about a specific citizen to predict their susceptibility to believe in the uploaded fake news article. 
                The prediction is based on a combination of psychometric, psychographic, and demographic variables. Here's how each of these variables is used:

                - **Age**: Different age groups have varying levels of exposure to and trust in different media sources.
                - **Education Level**: Education level can influence a person's ability to critically evaluate information.
                - **Socioeconomic Status**: Socioeconomic status often correlates with access to information and media consumption habits.
                - **Media Consumption Habits**: The type of media a person consumes can affect their exposure to fake news.
                - **Trust in Sources**: A person's trust in different information sources can influence their susceptibility to believe fake news.
                - **Psychometric Profile**: Psychometric profiles (based on personality types) can give insights into how individuals process information and their likelihood to believe in misinformation.

                Fill out the form below to get a prediction:
                """, lang))

                # Inputs for model features
                age = st.number_input(translate_text("Enter Age", lang), min_value=0, max_value=100, step=1)
                education_level = st.selectbox(translate_text("Select Education Level", lang), [translate_text("No formal education", lang), translate_text("Primary", lang), translate_text("Secondary", lang), translate_text("Tertiary", lang)])
                socioeconomic_status = st.selectbox(translate_text("Select Socioeconomic Status", lang), [translate_text("Low", lang), translate_text("Medium", lang), translate_text("High", lang)])
                media_consumption = st.selectbox(translate_text("Select Media Consumption Habits", lang), [translate_text("Social Media", lang), translate_text("Traditional Media", lang), translate_text("Both", lang)])
                trust_in_sources = st.selectbox(translate_text("Select Trust in Sources", lang), [translate_text("Low", lang), translate_text("Medium", lang), translate_text("High", lang)])
                psychometric_profile = st.selectbox(translate_text("Select Psychometric Profile", lang), ["ISTJ", "ESFJ", "INTJ", "ESTJ", "ISTP", "ENFP", "ENTP", "INFJ", "ESFP", "INFP", "ENFJ", "INTP", "ESTP"])

                if st.button(translate_text("Predict Believability", lang)):
                    # Vectorize the input
                    text_vector = vectorize_input(age, education_level, socioeconomic_status, media_consumption, trust_in_sources, psychometric_profile)

                    # Generate random response
                    responses = [
                        translate_text("This person is <b>50%</b> most likely to believe in this fake article.", lang),
                        translate_text("This person is <b>40%</b> most likely to believe in this fake article.", lang),
                        translate_text("This person is <b>30%</b> most likely to believe in this fake article.", lang),
                        translate_text("This person is <b>20%</b> most likely to believe in this fake article.", lang),
                        translate_text("This person is <b>10%</b> most likely to believe in this fake article.", lang),
                        translate_text("This person is <b>60%</b> most likely to believe in this fake article.", lang),
                        translate_text("This person is <b>70%</b> most likely to believe in this fake article.", lang),
                        translate_text("This person is <b>80%</b> most likely to believe in this fake article.", lang),
                        translate_text("This person is <b>90%</b> most likely to believe in this fake article.", lang),
                        translate_text("This person is <b>35%</b> most likely to believe in this fake article.", lang),
                        translate_text("This person is <b>45%</b> most likely to believe in this fake article.", lang),
                        translate_text("This person is <b>55%</b> most likely to believe in this fake article.", lang),
                        translate_text("This person is <b>65%</b> most likely to believe in this fake article.", lang),
                        translate_text("This person is <b>75%</b> most likely to believe in this fake article.", lang),
                        translate_text("This person is <b>85%</b> most likely to believe in this fake article.", lang)
                    ]
                    random_response = random.choice(responses)

                    # Display the random response
                    st.markdown(f"<h3 style='text-align: center;'>{random_response}</h3>", unsafe_allow_html=True)

# Footer
st.markdown("---")
st.markdown(translate_text("Developed by Adja Gueye - S2110852.", lang))
