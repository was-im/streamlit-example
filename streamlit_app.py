
import streamlit as st
import pandas as pd

# Load the dataset from GitHub
data_url = 'https://github.com/was-im/streamlit-example/blob/master/adc.csv'
data = pd.read_csv(data_url, encoding='utf-8', error_bad_lines=False, na_values=["NA", "N/A"])

def main():
    st.set_page_config(page_title="Adult Income Census Prediction", layout="wide")

   
    # Main content
    st.title("Adult Income Census Prediction")
    st.write("Customize the input parameters and click 'Predict' to see the income prediction result.")

    # Create input columns
    col1, col2 = st.beta_columns(2)

    with col1:
        st.subheader("Personal Information")
        age = st.slider('Age', 18, 100, 10)
        sex = st.radio('Sex', ('Male', 'Female'))

    with col2:
        st.subheader("Financial Information")
        capital_gain = st.number_input('Capital Gain', 0)
        capital_loss = st.number_input('Capital Loss', 0)

    hours_per_week = st.slider('Hours Per Week', 0, 80, 8)

    st.subheader("Additional Information")
    country = st.selectbox('Country', ('US', 'Non-US'))
    employment_type = st.selectbox('Employment Type', ('Private', 'Government', 'Self-employed', 'Without pay'))

    # Make Prediction
    if st.button('Predict'):
        prediction = simulate_prediction(age, sex, capital_gain, capital_loss, hours_per_week, country, employment_type)

        if prediction == 0:
            st.warning('The income is below or equal to 50K')
        else:
            st.success('The income is above 50K')

def simulate_prediction(age, sex, capital_gain, capital_loss, hours_per_week, country, employment_type):
    # Placeholder code to simulate the prediction process
    # You can replace this with your actual model prediction code
    # For demonstration purposes, this code will always return 1
    return 1

if __name__ == '__main__':
    main()


