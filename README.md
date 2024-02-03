# Deep Reinforcement Learning for Time Series Prediction

This project implements a deep reinforcement learning algorithm specifically designed for predicting (financial) time series. It utilizes deep Q-learning, employing a deep neural network to map the state vector to the Q values, which signify the quality of specific actions (buy or sell).

## Overview

Stock prediction, while ideally suited for a reinforcement learning framework due to the direct link between actions and rewards (profit or loss), poses significant challenges. The primary hurdle is the low signal-to-noise ratio prevalent in stock price data. Despite these challenges, I am optimistic about the potential for success. The algorithm has been initially validated using a noisy sinusoidal signal, demonstrating that the agent can learn to make profitable trades after a few epochs. Real stock data, however, exhibit more complex patterns, necessitating additional training data and potential code adjustments.

## Getting Started

### Create a Conda Environment
To set up your environment to run the code, you can create a Conda environment using the provided `.yml` file:

```bash
conda env create -f environment.yml
```
This will create an environment named `tradebot`. You can change the environment name directly in the `environment.yml` file if desired.

### Example Notebook

`demo.ipynb` serves as an illustrative example of how to apply the reinforcement learning algorithm.

### Data and Features

- **Ticker**: By default, the algorithm uses `"sinus_noise_std_1"` to generate an artificial time series consisting of a sinus signal with superimposed noise from a Student's t-distribution with 5 degrees of freedom. This will give you a first intuition. It's also fully compatible with any ticker supported by Yahoo Finance, such as `'BTC-USD'` or `'AAPL'`.

- **Features and Asset Classes**: The algorithm allows for the experimentation with numerous combinations of features and asset classes. Features must be defined as a list of tuples, where the first element is the feature (e.g., a technical indicator) and the second element is the scaling method (e.g., z-score, min-max scaling). There are a few custom-built features like `ratio_sma_ws1_ws2`, which computes the ratio of two moving averages of window size `ws1` and `ws2`. Users are encouraged to test various feature combinations to discover the most effective predictors for different asset classes. The `FeatureEngineer` class is built on top of `stockstats`, enabling also the integration of those features supported by the library (e.g., `log-ret`, `close_[ws]_sma` for log returns, and simple moving average based on the closing price over a specified window size).

- - **Network Architecture**: The core of this model is an LSTM (Long Short-Term Memory) network, which is particularly well-suited for learning from sequences of data, making it ideal for time series prediction. The model also employs batch learning for efficient training over large datasets.

## Collaboration

I am always open to collaborations. Feel free to reach out to me at horstmann.tobias@googlemail.com.

