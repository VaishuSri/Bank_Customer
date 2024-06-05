import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

# Load the data
df = pd.read_csv("Bank_Churn.csv")
df.drop(columns=['Unnamed: 0'], inplace=True)

# Streamlit app
st.title("Bank Customer Churn Analysis")

st.sidebar.title("Navigation")
page = st.sidebar.selectbox("Choose a page", ["Home","EDA", "Churn Prediction","About"])
if page=="Home":
    st.write("""
**Welcome to the Bank Customer Churn Analysis App!**

This app is designed to help you explore and understand the factors that contribute to customer churn in a bank. Customer churn refers to the phenomenon where customers stop doing business with a company, and understanding this can be critical for banks to retain their customers and improve their services.

**Key Features of This App:**
1. **Exploratory Data Analysis (EDA):**
   - Visualize and analyze various aspects of the customer data.
   - Gain insights into factors affecting customer churn, such as age, gender, geography, and more.
   - Use interactive visualizations to explore the data easily.

2. **Churn Prediction:**
   - Leverage machine learning to predict whether a customer is likely to churn.
   - Train a Random Forest Classifier on the dataset.
   - Input new customer data to get churn predictions in real-time.

**Why is Customer Churn Analysis Important?**
- **Retention Strategies:** By identifying the reasons why customers leave, banks can develop targeted strategies to retain them.
- **Improved Services:** Understanding customer needs and pain points can help in enhancing the overall service quality.
- **Cost Efficiency:** Retaining existing customers is often more cost-effective than acquiring new ones.

**How to Use This App:**
1. Navigate to the "EDA" section to explore the data and understand the churn patterns through various visualizations.
2. Use the "Churn Prediction" section to predict whether a new customer is likely to churn based on their profile.

Dive in to uncover the insights and take actionable steps to improve customer retention!
""")

   
if page == "EDA":
    col1, col2 = st.columns(2)
    with col1:
        st.header("Exploratory Data Analysis")

        churned = df[df['Exited'] == 1]
        non_churned = df[df['Exited'] == 0]

        # Plotting
        plt.figure(figsize=(14, 6))

        
        st.subheader("Histogram of CreditScore")
        churned = df[df['Exited'] == 1]
        non_churned = df[df['Exited'] == 0]
        
        fig, ax = plt.subplots(figsize=(18, 6))

        # Histogram for CreditScore
        ax.hist(non_churned['CreditScore'], bins=15, alpha=0.5, label='Non-Churned', color='blue', edgecolor='k')
        ax.hist(churned['CreditScore'], bins=15, alpha=0.7, label='Churned', color='red', edgecolor='k')
        ax.set_xlabel('CreditScore')
        ax.set_ylabel('Frequency')
        ax.legend()
        ax.set_title('Histogram of CreditScore')
        st.pyplot(fig)
        
        st.subheader("Churn Rate by Gender")
        churn_rate_gender = df.groupby('Gender')['Exited'].mean().reset_index()
        fig, ax = plt.subplots()
        sns.barplot(x='Gender', y='Exited', data=churn_rate_gender, ax=ax)
        ax.set_title("Churn Rate by Gender")
        st.pyplot(fig)

        # Calculate churn rate by geography
        churn_rate = df.groupby('Geography')['Exited'].mean().reset_index()
        churn_rate.columns = ['Geography', 'ChurnRate']

        st.subheader("Churn Rate by Geography")
        churn_rate_geo = df.groupby('Geography')['Exited'].mean().reset_index()
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.barplot(x='Geography', y='Exited', data=churn_rate_geo, ax=ax)
        ax.set_xlabel('Geography')
        ax.set_ylabel('Churn Rate')
        ax.set_title('Churn Rate by Geography')
        st.pyplot(fig)
        
     
        
    with col2:
        st.subheader("Age Distribution by Churn")
        fig, ax = plt.subplots()
        sns.boxplot(x='Exited', y='Age', data=df, ax=ax)
        ax.set_title("Age Distribution by Churn")
        st.pyplot(fig)

        st.subheader("Correlation Heatmap")
        fig, ax = plt.subplots()
        sns.heatmap(df.corr(), annot=True, cmap='coolwarm', ax=ax)
        ax.set_title("Correlation Heatmap")
        st.pyplot(fig)

        st.subheader("Basic Information")
        st.write(df.describe())


if page == "Churn Prediction":
    st.header("Churn Prediction")

    # Preprocess data
    df['Gender'] = LabelEncoder().fit_transform(df['Gender'])

    # Encode 'Geography' column
    geography_mapping = {'France': 1, 'Germany': 2, 'Spain': 3}
    df['Geography'] = df['Geography'].map(geography_mapping)

    X = df.drop(['Exited'], axis=1)
    y = df['Exited']

    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Train the model
    model = RandomForestClassifier()
    model.fit(X_train, y_train)

    # Predictions
    y_pred = model.predict(X_test)
    report = classification_report(y_test, y_pred, output_dict=True)

    #st.subheader("Model Performance")
    #st.write("Accuracy:", report['accuracy'])
    #st.write(pd.DataFrame(report).transpose())

    # User input for prediction
    st.subheader("Predict Churn for a New Customer")
    gender = st.selectbox("Gender", ['Male', 'Female'])
    age = st.number_input("Age", min_value=18, max_value=100, value=30)
    geography = st.selectbox("Geography", ['France', 'Germany', 'Spain'])
    balance = st.number_input("Balance", min_value=0, value=10000)
    tenure = st.number_input("Tenure", min_value=0, max_value=10, value=3)
    products = st.number_input("Products", min_value=1, max_value=4, value=1)
    is_active_member = st.selectbox("Is Active Member", [0, 1])
    estimated_salary = st.number_input("Estimated Salary", min_value=0, value=50000)
    creditscore=st.number_input("CreditScore",min_value=350,max_value=850)
    hascrcard=st.selectbox("HasCrCard",[0,1])

    if st.button("Predict"):
        input_data = pd.DataFrame({
            'CreditScore':[creditscore],
            'Geography': [geography_mapping[geography]],	
            'Gender': [1 if gender == 'Male' else 0],
            'Age': [age],	
            'Tenure': [tenure],
            'Balance': [balance],
            'NumOfProducts': [products],
            'HasCrCard':[hascrcard],
            'IsActiveMember': [is_active_member],	
            'EstimatedSalary': [estimated_salary]})
        prediction = model.predict(input_data)[0]
        st.write("Churn Prediction:", "Yes" if prediction == 1 else "No")
if page=="About":
       st.write("""
**About Me!**
    Hi, I am G Srivaishnavi B.E Graduate, have an 2+yrs of IT Experience.
Thank You,
Thank you for visiting our project! Your interest and support mean a lot to us. We hope you find the tools and resources provided here useful and engaging. 
Together, we can make data analysis more accessible and powerful for everyone. If you enjoy using our application, please consider contributing to our project or spreading the word.
Your feedback and contributions are always welcome.

Github_Link:https:https://github.com/VaishuSri/Bank_Customer.git
Application_Link:https://bank-customer.onrender.com/
""")

