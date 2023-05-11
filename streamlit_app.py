import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import subprocess

# Load the dataset from GitHub
data_url = 'https://github.com/was-im/streamlit-example/blob/master/adc.csv'
data = pd.read_csv(data_url, encoding='utf-8', error_bad_lines=False)

# Select relevant columns
selected_columns = ['age', 'sex', 'hours.per.week', 'native.country', 'workclass', 'capital.gain', 'capital.loss', 'income']
data = data[selected_columns]

# Convert categorical variables to numerical using label encoding
label_encoder = LabelEncoder()
data['sex'] = label_encoder.fit_transform(data['sex'])
data['native.country'] = label_encoder.fit_transform(data['native.country'])
data['workclass'] = label_encoder.fit_transform(data['workclass'])

# Split the data into features (X) and target (y)
X = data.drop('income', axis=1)
y = data['income']

# Train a Random Forest classifier
clf = RandomForestClassifier(random_state=42)
clf.fit(X, y)

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
    country = st.selectbox('Country', data['native.country'].unique())
    employment_type = st.selectbox('Employment Type', data['workclass'].unique())
    
        # Make Prediction
    if st.button('Predict'):
        prediction = simulate_prediction(age, sex, capital_gain, capital_loss, hours_per_week, country, employment_type)

        if prediction == 0:
            st.warning('The income is below or equal to 50K')
        else:
            st.success('The income is above 50K')

def simulate_prediction(age, sex, capital_gain, capital_loss, hours_per_week, country, employment_type):
    # Prepare user input for prediction
    user_input = pd.DataFrame({
        'age': [age],
        'sex': [sex],
        'hours.per.week': [hours_per_week],
        'native.country': [country],
        'workclass': [employment_type],
        'capital.gain': [capital_gain],
        'capital.loss': [capital_loss]
    })

    # Convert categorical variables to numerical
    user_input['sex'] = label_encoder.transform(user_input['sex'])
    user_input['native.country'] = label_encoder.transform(user_input['native.country'])
    user_input['workclass'] = label_encoder.transform(user_input['workclass'])

    # Make a prediction using the trained model
    prediction = clf.predict(user_input)[0]

    return prediction

if __name__ == '__main__':
    main()
