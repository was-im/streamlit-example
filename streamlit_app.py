# Import Libraries
import pandas as pd
import streamlit as st

# Try importing CatBoost
try:
    from catboost import CatBoostClassifier
    catboost_available = True
except ImportError:
    catboost_available = False

# Dataset URL
data_url = 'https://raw.githubusercontent.com/was-im/streamlit-example/master/adc.csv'

# Download the dataset
data = pd.read_csv(data_url)

# Preprocess the dataset
# Assuming you have already preprocessed the dataset, including handling missing values, encoding categorical variables, etc.

# Split the dataset into features and target variable
X = data.drop('income', axis=1)
y = data['income']

# Train the CatBoost model if available
if catboost_available:
    model = CatBoostClassifier()
    model.fit(X, y)

# create a function
def main():
    html_temp = """
                <div style="background-color:tomato;padding:10px">
                <h2 style="color:white;text-align:center;">Adult Income Census Prediction</h2>
                </div>
                """

    st.markdown(html_temp, unsafe_allow_html=True)

    # Age
    age = st.slider('Age', 1, 100, 10)

    # Sex
    sex = st.radio('Sex', ('Male', 'Female'))

    # Capital Gain
    capital_gain = st.number_input('Capital Gain', 0)

    # Capital Loss
    capital_loss = st.number_input('Capital Loss', 0)

    # Hours per week
    hours_per_week = st.slider('Hours Per Week', 0, 168, 8)

    # Country
    country = st.radio('Country', ('Us', 'Non-US'))

    # Employment Type
    employment_type = st.selectbox('Employment_type', ('Private', 'Government', 'Self_employed', 'Without_pay'))

    # Make Prediction
    if catboost_available and st.button('Predict'):
        features = [[age, sex, capital_gain, capital_loss,
                     hours_per_week, country, employment_type]]
        prediction = model.predict(features)[0]
        if prediction == 0:
            st.warning('The income is below or equal to 50K')
        else:
            st.success('The income is above 50K')
    elif not catboost_available:
        st.warning('CatBoost is not available. Please check your environment setup.')


if __name__ == '__main__':
    main()
