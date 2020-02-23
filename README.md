# stock investment strategy based on deep learning
This is the code of the timeing part of'LSTM加LLT加多因子' on【JoinQuant】https://www.joinquant.com/algorithm/live/liveUrlShareIndex?backtestId=678ee450e975016a28945ad05114dbdd    (password:gblxjo)

Here I utilized the daily data of 20 index on the Chinese stock market form 2012 to 2018.

With those data, I construted features like 'RSI', 'KDJ', 'MACD', etc. I concatenate those feature vectors everyday into a time series of length 15.Then I applied the 'LLT' filter to reduce noise, and label the index as 1 if its 'LLT' value goes up in the day after the end of the time series or label it as 0 otherwise.

With the features and labels, I built a LSTM network to fit them and thus derived a model that can be used to predict the market trend with the market data of the past 15 days as an input.

