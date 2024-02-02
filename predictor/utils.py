import os
import altair as alt
import pandas as pd


def no_scaling(df, feature):
    return df

def z_scaling(df, feature):

    mean = df[feature].mean()
    std = df[feature].std()
    df[feature] = (df[feature] - mean) / std

    return df

def min_max_scaling(df, feature):
    
    min_ = df[feature].min()
    max_ = df[feature].max()

    df[feature] = (df[feature] - min_) / (max_ - min_)
    return df

def visualize(data, title="Trading Session"):
    # Ensure data is a DataFrame and has the necessary columns
    #if not isinstance(data, pd.DataFrame) or not all(col in data.columns for col in ['close', 'date', 'position', 'action']):
    #    raise ValueError("Data must be a DataFrame with 'close', 'date', 'position', and 'action' columns")

    # Specify y-axis scale for stock prices
    scale = alt.Scale(domain=(min(data['close']) - 1, max(data['close']) + 1), clamp=True)

    # Plot a line chart for stock positions
    actual = alt.Chart(data).mark_line(
        color='green',
        opacity=0.5
    ).encode(
        x='date:T',
        y=alt.Y('close:Q', axis=alt.Axis(format='$.2f', title='Price'), scale=scale)
    ).interactive(bind_y=False)

    # Plot the BUY and SELL actions as points
    points = alt.Chart(data).transform_calculate(
    actionLabel="datum.action == 1 ? 'BUY' : 'SELL'"
    ).mark_point(
        filled=True
    ).encode(
        x='date:T',
        y='close:Q',
        color='actionLabel:N'
    ).interactive(bind_y=False)

    # Merge the charts
    chart = alt.layer(actual, points, title=title).properties(height=400, width=900)

    return chart



def save_params(file_path, ticker, features, lags, episodes, learning_rate, 
                              tau, gamma, epsilon_decay, buy_and_hold, perf_on_test, comment):
    # Create a DataFrame from the parameters
    data = {
        'ticker': [ticker],
        'features': [features],
        'lags': [lags],
        'episodes': [episodes],
        'learning_rate': [learning_rate],
        'tau': [tau],
        'gamma': [gamma],
        'epsilon_decay': [epsilon_decay],
        'buy_and_hold' : [buy_and_hold],
        'perf_on_test' : [perf_on_test],
        'comment' : [comment]

    }
    df = pd.DataFrame(data)

    if os.path.exists(file_path):
        # Check if the file is empty or has headers
        try:
            header_exists = pd.read_csv(file_path, nrows=0).columns.tolist()
        except pd.errors.EmptyDataError:
            header_exists = False

        # Append data with or without header
        df.to_csv(file_path, mode='a', header=not header_exists, index=False)
    else:
        # Create a new file and write headers
        df.to_csv(file_path, mode='w', header=True, index=False)