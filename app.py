
import streamlit as st
import pandas as pd
import pickle
from datetime import datetime

# Load the saved model, preprocessor, and label encoder
@st.cache_resource
def load_model_components():
    with open('best_model.pkl', 'rb') as f:
        model = pickle.load(f)
    with open('preprocessor.pkl', 'rb') as f:
        preprocessor = pickle.load(f)
    with open('label_encoder.pkl', 'rb') as f:
        label_encoder = pickle.load(f)
    with open('feature_metadata.pkl', 'rb') as f:
        feature_metadata = pickle.load(f)
    return model, preprocessor, label_encoder, feature_metadata

model, preprocessor, label_encoder, feature_metadata = load_model_components()

# Initialize session state for predictions if not already present
if 'predictions' not in st.session_state:
    st.session_state.predictions = pd.DataFrame(columns=feature_metadata['original_columns'] + ['Predicted_EPDS_Result', 'Prediction_Time'])

st.title('Postpartum Depression Prediction App')
st.write('Enter the details below to predict the likelihood of postpartum depression.')

# Custom questions mapping
custom_questions = {
    'Age': 'Your age',
    'Residence': 'Current residence',
    'Education Level': 'Level of education',
    'Marital status': 'Marital status',
    'Occupation before latest pregnancy': 'Your occupation before latest pregnancy',
    'Occupation After Your Latest Childbirth': 'Your occupation after your latest childbirth',
    "Husband's education level": "Husband's level of education",
    'Husband’s monthly income': 'Husband’s monthly income',
    'Total children': 'Number of total children',
    'Family type': 'Family type',
    'Number of household members': 'Number of household members',
    'Relationship with the in-laws': 'Relationship with the in-laws',
    'Relationship with husband': 'How is your relationship with your husband?',
    'Relationship with the newborn': 'Relationship/Bonding with the newborn',
    'Relationship between father and newborn': 'How is the relationship between your husband and your child?',
    'Feeling about motherhood': 'What is your feeling about motherhood?',
    'Recieved Support': 'What kind of support did you receive before and after childbirth?',
    'Need for Support': 'Do you wish/feel that you should be offered more help?',
    'Major changes or losses during pregnancy': 'Did you have any major changes or losses for pregnancy?',
    'Abuse': 'Has anyone in your family/husband/in-laws ever treated you like this?',
    'Trust and share feelings': 'Trust and share feelings with close friends?',
    'Number of the latest pregnancy': 'Number of the latest pregnancy',
    'Pregnancy length': 'Pregnancy length',
    'Pregnancy plan': 'Was this latest pregnancy planned?',
    'Regular checkups': 'Had regular checkups?',
    'Fear of pregnancy': 'Fear of pregnancy?',
    'Diseases during pregnancy': 'Diseases during pregnancy?',
    'Age of newborn': 'Age of newborn',
    'Mode of delivery': 'Mode of delivery',
    'Gender of newborn': 'Gender of newborn',
    'Birth compliancy': 'Any birth complications?',
    'Breastfeed': 'Do you breastfeed?',
    'Newborn illness': 'Any illness of the infant?',
    'Worry about newborn': "Are you constantly worried about the newborn's health and activities?",
    'Relax/sleep when newborn is tended': 'Can you relax/sleep when the newborn is monitored by someone else?',
    'Relax/sleep when the newborn is asleep': 'Can you relax/sleep when the newborn is asleep?',
    'Angry after latest child birth': 'Have you been feeling angry, irritated, and difficult to calm down after childbirth?',
    'Feeling for regular activities': 'How do you feel to perform regular activities like before pregnancy?',
    'Depression before pregnancy (PHQ2)': 'Depression before pregnancy (PHQ2)',
    'Depression during pregnancy (PHQ2)': 'Depression during pregnancy (PHQ2)',
    'PHQ9 Score': 'PHQ9 Score',
    'EPDS Score': 'EPDS Score'
}

# Create input fields based on feature_metadata
input_data = {}

# Group inputs into columns for better layout
num_cols = len(feature_metadata['original_columns'])
cols = st.columns(3)

input_index = 0
for col_name in feature_metadata['original_columns']:
    current_col = cols[input_index % 3]
    question_text = custom_questions.get(col_name, f'Enter {col_name}') # Use custom question or default

    if col_name in feature_metadata['numerical_cols']:
        if col_name == 'Age':
            input_data[col_name] = current_col.number_input(question_text, min_value=18, max_value=60, value=25)
        elif col_name == 'Number of the latest pregnancy':
            input_data[col_name] = current_col.number_input(question_text, min_value=1, max_value=10, value=1)
        elif col_name == 'PHQ9 Score':
            input_data[col_name] = current_col.number_input(question_text, min_value=0, max_value=27, value=10)
        elif col_name == 'EPDS Score': # Although 'EPDS Score' is in numerical_cols, it's often closely related to the target, so it might not be a direct input for prediction, but for this example, we'll keep it as input.
            input_data[col_name] = current_col.number_input(question_text, min_value=0, max_value=30, value=12)
        else:
            input_data[col_name] = current_col.number_input(question_text, value=0.0)
    elif col_name in feature_metadata['categorical_cols']:
        options = feature_metadata['categorical_options'][col_name]
        input_data[col_name] = current_col.selectbox(question_text, options)
    else:
        # Fallback for any unexpected columns, though original_columns should cover everything
        input_data[col_name] = current_col.text_input(question_text, 'N/A')
    input_index += 1

if st.button('Predict'):
    # Convert input_data to a DataFrame
    input_df = pd.DataFrame([input_data])

    # Ensure column order matches the original training data before preprocessing
    input_df = input_df[feature_metadata['original_columns']]

    # Preprocess the input data
    # ColumnTransformer expects a 2D array, so we need to pass a DataFrame
    processed_input = preprocessor.transform(input_df)

    # Make prediction
    prediction_encoded = model.predict(processed_input)
    prediction_label = label_encoder.inverse_transform(prediction_encoded)[0]

    st.success(f'The predicted EPDS Result is: **{prediction_label}**')

    # Store the prediction and inputs in session state
    new_record = input_df.copy()
    new_record['Predicted_EPDS_Result'] = prediction_label
    new_record['Prediction_Time'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    st.session_state.predictions = pd.concat([st.session_state.predictions, new_record], ignore_index=True)

st.subheader('Recorded Predictions')
if not st.session_state.predictions.empty:
    st.dataframe(st.session_state.predictions)

    # Option to download the table
    csv_data = st.session_state.predictions.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="Download Recorded Predictions as CSV",
        data=csv_data,
        file_name=f"predictions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
        mime="text/csv",
    )
else:
    st.info("No predictions recorded yet.")
