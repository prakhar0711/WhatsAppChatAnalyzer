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
    user_list.remove('group notification')
    user_list.sort()
    user_list.insert(0, 'Overall')
    selected_user = st.sidebar.selectbox("Show analysis wrt", user_list)

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
            fig, ax = plt.subplots()
            ax.plot(timeline['time'], timeline['message'], color='green')
            plt.xticks(rotation='vertical')
            ax.set_xlabel('Month')  # Placeholder for x-axis
            ax.set_ylabel('Number of Messages')
            st.header("Monthly Timeline")
            st.pyplot(fig, use_container_width=True)
        with col2:
            # Daily timeline
            daily_timeline = helper.get_daily_timeline(selected_user, df)
            fig, ax = plt.subplots()
            ax.plot(daily_timeline['only_date'], daily_timeline['message'], color='red')
            plt.xticks(rotation='vertical')
            ax.set_xlabel('Date')  # Placeholder for x-axis
            ax.set_ylabel('Number of Messages')
            st.header("Daily Timeline")
            st.pyplot(fig, use_container_width=True)

        # Activity map
        st.title("Activity Map")
        col1, col2 = st.columns(2)
        with col1:
            st.header("Most busy day")
            busy_day = helper.week_activity_map(selected_user, df)
            fig, ax = plt.subplots()
            ax.bar(busy_day.index, busy_day.values, color=['red', 'green', 'blue', 'orange', 'purple', 'yellow', 'cyan'])
            plt.xticks(rotation='vertical')
            ax.set_xlabel('Day')  # Placeholder for x-axis
            ax.set_ylabel('Number of Messages')
            st.pyplot(fig, use_container_width=True)
        with col2:
            st.header("Most busy month")
            busy_month = helper.month_activity_map(selected_user, df)
            fig, ax = plt.subplots()
            ax.bar(busy_month.index, busy_month.values, color=['red', 'green', 'blue', 'orange', 'purple', 'yellow', 'cyan'])
            plt.xticks(rotation='vertical')
            ax.set_xlabel('Month')  # Placeholder for x-axis
            ax.set_ylabel('Number of Messages')
            st.pyplot(fig, use_container_width=True)

        # Activity heatmap
        user_heatmap = helper.activity_heat_map(selected_user, df)
        fig, ax = plt.subplots()
        ax = sns.heatmap(user_heatmap)
        st.header("Activity Heatmap")
        st.pyplot(fig, use_container_width=True)

        # Busiest users in the group
        if selected_user == 'Overall':
            x, percent_new_df = helper.most_busy_users(df)
            fig, ax = plt.subplots()
            col1, col2 = st.columns(2)
            with col1:
                st.header('Most Busy Users')
                ax.bar(x.index, x.values, color=['red', 'green', 'blue', 'orange', 'purple', 'yellow', 'cyan'])
                ax.set_xlabel('User')  # Placeholder for x-axis
                ax.set_ylabel('Number of Messages')
                plt.xticks(rotation='vertical')
                st.pyplot(fig, use_container_width=True)
            with col2:
                st.header('Percentage of Messages Sent by Each User')
                st.dataframe(percent_new_df, use_container_width=True, hide_index=True)

        # Wordcloud
        st.header("Wordcloud")
        df_wc = helper.create_wordcloud(selected_user, df)
        fig, ax = plt.subplots()
        col1, col2 = st.columns(2)
        with col1:
            ax.imshow(df_wc)
            st.pyplot(fig, use_container_width=True)

        # Most common words
        most_common_df = helper.most_common_words(selected_user, df)
        fig, ax = plt.subplots()
        ax.barh(most_common_df['Common Word'], most_common_df['Word Count'], color=['red', 'green', 'blue', 'orange', 'purple', 'yellow', 'cyan'])
        ax.set_xlabel('Average Use')  # Placeholder for x-axis
        ax.set_ylabel('Common Word')
        col1, col2 = st.columns(2)
        with col1:
            st.header("Most Common Words")
            plt.xticks(rotation='vertical')
            st.pyplot(fig, use_container_width=True)
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