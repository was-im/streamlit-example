# Import Libraries
import streamlit as st
import pickle

# Load saved model
pickle_in = open('catboost.pkl', 'rb')
classifier = pickle.load(pickle_in)

# create a function
def main():

    html_temp =  """
                <div style="background-color:tomato;padding:10px">
                <h2 style="color:white;text-align:center;">Adult Income Census Prediction</h2>
                </div

                """

    st.markdown(html_temp, unsafe_allow_html=True)

    # Age
    age = st.slider('Age', 1, 100, 10)

    # Sex
    gen_display = ('Male', 'Female')
    gen_options = list(range(len(gen_display)))
    sex = st.radio('Sex', gen_options, format_func=lambda x: gen_display[x])

    # Capital Gain
    capital_gain = st.number_input('Capital Gain', 0)

    # Capital Loss
    capital_loss = st.number_input('Capital Loss', 0)

    # Hours per week
    hours_per_week = st.slider('Hours Per Week', 0, 168, 8)

    # Country
    con_display = ('Us', 'Non-US')
    con_options = list(range(len(con_display)))
    country = st.radio('Country', con_options,format_func=lambda x: con_display[x])

    # Employment Type
    emp_display = ('Private', 'Government', 'Self_employed', 'Without_pay')
    emp_options = list(range(len(emp_display)))
    employment_type = st.selectbox('Employment_type', emp_options, format_func=lambda x: emp_display[x])

    # Make Prediction
    if st.button('Predict'):
        features = [[age, sex, capital_gain, capital_loss,
                     hours_per_week, country, employment_type]]
        prediction = classifier.predict(features)
        lc = [str(i) for i in prediction]
        ans = int("".join(lc))
        if ans == 0:
            st.warning('The income is below or equal to 50K')
        else:
            st.success('The income is above 50K')


if __name__ == '__main__':
    main()
