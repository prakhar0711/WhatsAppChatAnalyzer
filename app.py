import streamlit as st
import preprocessor
import helper
import matplotlib.pyplot as plt
import seaborn as sns

# Set page configuration and sidebar title
st.set_page_config(page_title="Whatsapp Chat Analyzer", page_icon=":bar_chart:", layout="wide")
st.sidebar.image('whatsapp.png', width=75)
st.sidebar.title("Whatsapp Chat Analyzer")

# Upload file
uploaded_file = st.sidebar.file_uploader("Choose a file")
if uploaded_file is not None:
    bytes_data = uploaded_file.getvalue()
    data = bytes_data.decode('utf-8')
    df = preprocessor.preprocess(data)
    st.header("DataFrame")
    st.dataframe(df, use_container_width=True)

    # Fetch unique users
    user_list = df['user'].unique().tolist()
    if 'group notification' in user_list:
        user_list.remove('group notification')
    user_list.sort()
    user_list.insert(0, 'Overall')
    selected_user = st.sidebar.selectbox("Show analysis wrt", user_list)

    # Inside your Streamlit app, add the following code for the keyword search feature

    # Keyword search
    search_keyword = st.sidebar.text_input("Search for keyword")
    if st.sidebar.button("Search"):
        # Perform keyword search

        if search_keyword:
            keyword_results = helper.search_keywords(selected_user, df, search_keyword)
            st.subheader(f"Messages containing '{search_keyword}':")
            st.dataframe(keyword_results)

            # Perform sentiment analysis
            sentiment_results = helper.analyze_sentiment_in_keyword_messages(selected_user, df, search_keyword)
            st.subheader("Sentiment Analysis of Messages containing the keyword:")
            st.dataframe(sentiment_results[['message', 'sentiment_polarity']])

            # Visualize sentiment distribution
            st.subheader("Sentiment Distribution:")
            st.bar_chart(sentiment_results['sentiment_polarity'].value_counts())
        else:
            st.warning("Please enter a keyword to search.")

    # Show analysis button
    if st.sidebar.button("Show Analysis"):
        # Stats area
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

        # Monthly timeline
        col1, col2 = st.columns(2)
        with col1:
            timeline = helper.monthly_timeline(selected_user, df)
            st.header("Monthly Timeline")
            st.line_chart(timeline.set_index('time')['message'], height=500)
        with col2:
            # Daily timeline
            daily_timeline = helper.get_daily_timeline(selected_user, df)
            st.header("Daily Timeline")
            st.line_chart(daily_timeline.set_index('only_date')['message'], height=500)

        # Activity map
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

        # Activity heatmap
        user_heatmap = helper.activity_heat_map(selected_user, df)
        fig, ax = plt.subplots()
        ax = sns.heatmap(user_heatmap)
        st.header("Activity Heatmap")
        st.pyplot(fig, use_container_width=True)

        # Busiest users in the group
        if selected_user == 'Overall':
            x, percent_new_df = helper.most_busy_users(df)
            col1, col2 = st.columns(2)
            with col1:
                st.header('Most Busy Users')
                st.bar_chart(x, height=600)
            with col2:
                st.header('Percentage of Messages Sent by Each User')
                st.dataframe(percent_new_df, use_container_width=True, hide_index=True)

        # Wordcloud
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

        # Most used emoji
        emoji_df = helper.get_emojis(selected_user, df)
        st.header("Most Used Emojis")
        col1, col2 = st.columns(2)
        with col1:
            fig, ax = plt.subplots()
            ax.pie(emoji_df['Count'].head(), labels=emoji_df['Emoji'].head(), autopct="%0.2f")
            st.pyplot(fig, use_container_width=True)
        with col2:
            st.dataframe(emoji_df, use_container_width=True, hide_index=True)

    # sentiment analysis
    if st.sidebar.button("Sentiment Analysis"):
        sentiment_df = helper.perform_sentiment_analysis(selected_user, df)
        st.title("Sentiment Analysis")
        st.dataframe(sentiment_df[['message', 'sentiment_polarity']])

        sentiment_df = helper.perform_sentiment_analysis(selected_user, df)
        st.title("Sentiment Analysis Visualization")
        st.bar_chart(sentiment_df['sentiment_polarity'], height=600)