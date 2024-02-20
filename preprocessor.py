import re
import pandas as pd

def preprocess(data):
    # Define patterns for both formats
    pattern1 = r'\d{1,2}/\d{1,2}/\d{2,4},\s\d{1,2}:\d{2}\s-\s'  # Format: 12/10/2023, 18:47 -
    pattern2 = r'\[\d{1,2}/\d{1,2}/\d{2,4},\s\d{1,2}:\d{2}:\d{2}\s[AP]M\]\s'  # Format: [20/01/24, 5:08:18 PM]

    # Try pattern1 first
    messages = re.split(pattern1, data)[1:]
    dates = re.findall(pattern1, data)

    # If pattern1 doesn't match, try pattern2
    if not dates:
        messages = re.split(pattern2, data)[1:]
        dates = re.findall(pattern2, data)

    if not dates:
        raise ValueError("No date pattern matched in the input data.")

    # Create DataFrame
    df = pd.DataFrame({'user_message': messages, 'message_date': dates})
    df['message_date'] = df['message_date'].str.strip()

    # Adjust date format based on the pattern
    if '[' in dates[0]:
        date_format = r'[%d/%m/%y, %I:%M:%S %p]'
    else:
        date_format = r'%d/%m/%Y, %H:%M -'

    df['message_date'] = pd.to_datetime(df['message_date'], format=date_format)
    df.rename(columns={'message_date': 'date'}, inplace=True)

    # Extract user and message
    users = []
    messages = []
    for message in df['user_message']:
        entry = re.split('([\w\W]+?):\s', message)
        if entry[1:]:  # user name
            users.append(entry[1])
            messages.append(entry[2])
        else:
            users.append('group notification')
            messages.append(entry[0])

    df['user'] = users
    df['message'] = messages
    df.drop(columns=['user_message'], inplace=True)

    # Extract additional date-related information
    df['only_date'] = df['date'].dt.date
    df['year'] = df['date'].dt.year
    df['month_num'] = df['date'].dt.month
    df['month'] = df['date'].dt.month_name()
    df['day'] = df['date'].dt.day
    df['hour'] = df['date'].dt.hour
    df['minute'] = df['date'].dt.minute
    df['day_name'] = df['date'].dt.day_name()

    # Define time period
    period = []
    for hour in df[['day_name', 'hour']]['hour']:
        if hour == 23:
            period.append(str(hour) + "-" + str('00'))
        elif hour == 0:
            period.append(str('00') + "-" + str(hour + 1))
        else:
            period.append(str(hour) + "-" + str(hour + 1))
    df['time_period'] = period

    return df
