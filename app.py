import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
import helper
import preprocessor
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC

models = {
    'Logistic Regression': LogisticRegression(),
    'Random Forest': RandomForestClassifier(),
    'Multinomial Naive Bayes': MultinomialNB(),
    'Support Vector Machine': SVC()
}

st.set_page_config(page_title="Whatsapp Chat Analyzer", page_icon=":bar_chart:", layout="wide")
st.sidebar.image('whatsapp.png', width=75)
st.sidebar.title("Whatsapp Chat Analyzer")

uploaded_file = st.sidebar.file_uploader("Choose a file")
consent_given = st.sidebar.checkbox("Give consent for training data usage. Your chats won't be uploaded to the internet. It will just be used to enhance our training model. This will help us improve the accuracy of our models")

if uploaded_file is not None:
    bytes_data = uploaded_file.getvalue()
    data = bytes_data.decode('utf-8')
    df = preprocessor.preprocess(data)
    st.header("DataFrame")
    st.dataframe(df, use_container_width=True)

    user_list = df['user'].unique().tolist()
    if 'group notification' in user_list:
        user_list.remove('group notification')
    user_list.sort()
    user_list.insert(0, 'Overall')
    selected_user = st.sidebar.selectbox("Show analysis wrt", user_list)

    if st.sidebar.button("Show Analysis"):
        num_messages, words, num_media, num_links = helper.fetch_stats(selected_user, df)
        st.title("Statistics")
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.header("Total Messages")
            st.title(num_messages)
        with col2:
            st.header("Total Words")
            st.title(words)
        with col3:
            st.header("Media Shared")
            st.title(num_media)
        with col4:
            st.header("Links Shared")
            st.title(num_links)

        col1, col2 = st.columns(2)
        with col1:
            timeline = helper.monthly_timeline(selected_user, df)
            st.header("Monthly Timeline")
            st.line_chart(timeline.set_index('time')['message'], height=500)
        with col2:
            daily_timeline = helper.get_daily_timeline(selected_user, df)
            st.header("Daily Timeline")
            st.line_chart(daily_timeline.set_index('only_date')['message'], height=500)

        st.title("Activity Map")
        col1, col2 = st.columns(2)
        with col1:
            st.header("Most busy day")
            busy_day = helper.week_activity_map(selected_user, df)
            st.bar_chart(busy_day, height=600)
        with col2:
            st.header("Most busy month")
            busy_month = helper.month_activity_map(selected_user, df)
            st.bar_chart(busy_month, height=600)

        user_heatmap = helper.activity_heat_map(selected_user, df)
        fig, ax = plt.subplots()
        ax = sns.heatmap(user_heatmap)
        st.header("Activity Heatmap")
        st.pyplot(fig, use_container_width=True)

        if selected_user == 'Overall':
            x, percent_new_df = helper.most_busy_users(df)
            col1, col2 = st.columns(2)
            with col1:
                st.header('Most Busy Users')
                st.bar_chart(x, height=600)
            with col2:
                st.header('Percentage of Messages Sent by Each User')
                st.dataframe(percent_new_df, use_container_width=True, hide_index=True)

        st.header("Wordcloud")
        df_wc = helper.create_wordcloud(selected_user, df)
        st.image(df_wc.to_array())

        col1, col2 = st.columns(2)
        with col1:
            st.header("Most Common Words")
            most_common_df = helper.most_common_words(selected_user, df)
            st.bar_chart(most_common_df.set_index('Common Word'), height=500)
        with col2:
            st.header("Most Common Words by Count")
            st.dataframe(most_common_df, use_container_width=True, hide_index=True)

        emoji_df = helper.get_emojis(selected_user, df)
        st.header("Most Used Emojis")
        col1, col2 = st.columns(2)
        with col1:
            fig, ax = plt.subplots()
            ax.pie(emoji_df['Count'].head(), labels=emoji_df['Emoji'].head(), autopct="%0.2f")
            st.pyplot(fig, use_container_width=True)
        with col2:
            st.dataframe(emoji_df, use_container_width=True, hide_index=True)

        training_data = helper.generate_training_data(df, 'positive-words.txt', 'negative-words.txt', 'stop_hinglish.txt')
        X = training_data['message']
        y = training_data['sentiment']
        vectorizer = CountVectorizer()
        X_vec = vectorizer.fit_transform(X)
        X_train, X_test, y_train, y_test = train_test_split(X_vec, y, test_size=0.2, random_state=42)

        # Train the models
        trained_models = {}
        for model_name, model in models.items():
            st.write(f"Training {model_name}...")
            model.fit(X_train, y_train)
            trained_models[model_name] = model

        # Evaluate the models
        for model_name, model in trained_models.items():
            st.header(f"Evaluation Results for {model_name}")
            y_pred = model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            report = classification_report(y_test, y_pred, output_dict=True)

            # Display accuracy
            st.success(f"{model_name} Accuracy: {accuracy:.2%}")

            # Display classification report
            st.info(f"{model_name} Classification Report:")
            # Convert the classification report dictionary into a DataFrame
            report_df = pd.DataFrame(report).transpose()
            st.table(report_df)

        if consent_given:
            file_path = os.path.join('training_data.csv')
            if not os.path.exists(file_path):
                training_data.to_csv(file_path, index=False)
            else:
                with open(file_path, 'a', encoding='utf-8') as file:
                    training_data.to_csv(file, mode='a', header=False, index=False)
