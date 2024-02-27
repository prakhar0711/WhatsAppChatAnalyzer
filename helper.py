from collections import Counter
from urlextract import URLExtract
from wordcloud import wordcloud
import pandas as pd
import emoji
import streamlit as st
from collections import defaultdict
import re

extract = URLExtract()


@st.cache_data
def fetch_stats(selected_user, df):
    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]

    # fetch number of messages
    num_messages = df.shape[0]

    # fetch number of words
    words = []
    for message in df['message']:
        words.extend(message.split())

    # fetch number of media
    num_media = df[df['message'] == '<Media omitted>\n'].shape[0]

    # fetch number of links
    links = []
    for message in df['message']:
        links.extend(extract.find_urls(message))

    return num_messages, len(words), num_media, len(links)


@st.cache_data
def most_busy_users(df):
    df = df[df['user'] != 'group notification']
    x = df['user'].value_counts().head()
    df = round((df['user'].value_counts() / df.shape[0]) * 100, 2).reset_index().rename(
        columns={'index': 'name', 'user': 'percent(%)'})
    return x, df


@st.cache_data
def create_wordcloud(selected_user, df):
    f = open('stop_hinglish.txt', 'r', encoding='utf-8')
    stop_words = f.read()
    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]

    temp = df[df['user'] != 'group notification']
    temp = temp[temp['message'] != '<Media omitted>\n']

    def remove_stop_words(message):
        y = []
        for word in message.lower().split():
            if word not in stop_words:
                y.append(word)
        return " ".join(y)

    wc = wordcloud.WordCloud(width=500, height=500, min_font_size=10, background_color='white')
    temp['message'].apply(remove_stop_words)
    df_wc = wc.generate(temp['message'].str.cat(sep=" "))
    return df_wc


@st.cache_data
def most_common_words(selected_user, df):
    f = open('stop_hinglish.txt', 'r', encoding='utf-8')
    stop_words = f.read()
    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]

    temp = df[df['user'] != 'group notification']
    temp = temp[temp['message'] != '<Media omitted>\n']

    words = []
    for message in temp['message']:
        for word in message.lower().split():
            if word not in stop_words:
                words.append(word)

    most_common_df = pd.DataFrame(Counter(words).most_common(20))
    most_common_df = most_common_df.rename(columns={0: 'Common Word', 1: 'Word Count'})
    return most_common_df


@st.cache_data
def get_emojis(selected_user, df):
    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]

    emojis = []
    for message in df['message']:
        emojis.extend([c for c in message if c in emoji.EMOJI_DATA.keys()])

    emoji_df = pd.DataFrame(Counter(emojis).most_common(), columns=['Emoji', 'Count'])
    return emoji_df


@st.cache_data
def monthly_timeline(selected_user, df):
    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]

    timeline = df.groupby(['year', 'month_num', 'month']).count()['message'].reset_index()
    time = []
    for i in range(timeline.shape[0]):
        time.append(timeline['month'][i] + "-" + str(timeline['year'][i]))

    timeline['time'] = time
    return timeline


@st.cache_data
def get_daily_timeline(selected_user, df):
    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]

    daily_timeline = df.groupby('only_date').count()['message'].reset_index()
    return daily_timeline


@st.cache_data
def week_activity_map(selected_user, df):
    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]
    return df['day_name'].value_counts()


@st.cache_data
def month_activity_map(selected_user, df):
    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]
    return df['month'].value_counts()


def activity_heat_map(selected_user, df):
    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]
    user_heatmap = df.pivot_table(index='day_name', columns='time_period', values='message', aggfunc='count').fillna(0)
    return user_heatmap


def remove_emojis(text):
    # Pattern to remove emojis
    emoji_pattern = re.compile("["
                               u"\U0001F600-\U0001F64F"  # emoticons
                               u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                               u"\U0001F680-\U0001F6FF"  # transport & map symbols
                               u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                               u"\U00002500-\U00002BEF"  # chinese char
                               u"\U00002702-\U000027B0"
                               u"\U00002702-\U000027B0"
                               u"\U000024C2-\U0001F251"
                               u"\U0001f926-\U0001f937"
                               u"\U00010000-\U0010ffff"
                               u"\u2640-\u2642"
                               u"\u2600-\u2B55"
                               u"\u200d"
                               u"\u23cf"
                               u"\u23e9"
                               u"\u231a"
                               u"\ufe0f"  # dingbats
                               u"\u3030"
                               "]+", flags=re.UNICODE)
    return emoji_pattern.sub(r'', text)


# Read positive and negative words files
def read_sentiment_words(positive_words_file, negative_words_file):
    with open(positive_words_file, 'r', encoding='utf-8') as file:
        positive_words = file.read().splitlines()

    with open(negative_words_file, 'r', encoding='utf-8') as file:
        negative_words = file.read().splitlines()

    return positive_words, negative_words


# Assign sentiment label to messages
def assign_sentiment_label(message, positive_words, negative_words):
    positive_count = sum(1 for word in message.split() if word in positive_words)
    negative_count = sum(1 for word in message.split() if word in negative_words)

    if positive_count > negative_count:
        return 'positive'
    elif negative_count > positive_count:
        return 'negative'
    else:
        return 'neutral'


# Generate training data for sentiment analysis
def generate_training_data(df, positive_words_file, negative_words_file):
    positive_words, negative_words = read_sentiment_words(positive_words_file, negative_words_file)
    df = df[df['message'] != '<Media omitted>\n']
    training_data = defaultdict(list)
    for message in df['message']:
        # Remove emojis
        message = remove_emojis(message)
        # Remove links
        message = remove_links(message)
        # Remove numbers
        message = remove_numbers(message)
        sentiment_label = assign_sentiment_label(message, positive_words, negative_words)
        training_data['message'].append(message)
        training_data['sentiment'].append(sentiment_label)

    training_df = pd.DataFrame(training_data)
    return training_df


def remove_links(text):
    return re.sub(r'http\S+', '', text)


def remove_numbers(text):
    return re.sub(r'\d+', '', text)
