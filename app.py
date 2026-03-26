import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

st.set_page_config(page_title="Stock Predictor", layout="wide")

st.title("📈 Advanced Stock Price Predictor")

stock = st.text_input("Enter Stock Symbol (Example: AAPL, RELIANCE.NS)")

if st.button("Analyze"):

    data = yf.download(stock, period="1y")

    if data.empty:
        st.error("Invalid stock name ❌")
    else:
        st.subheader("Stock Data")
        st.dataframe(data.tail())

        # Closing price chart
        st.subheader("📊 Closing Price Chart")
        st.line_chart(data['Close'])

        # ML Prediction
        df = data[['Close']]
        df['Prediction'] = df['Close'].shift(-10)

        X = np.array(df.drop(['Prediction'], axis=1))[:-10]
        y = np.array(df['Prediction'])[:-10]

        model = LinearRegression()
        model.fit(X, y)

        future = np.array(df.drop(['Prediction'], axis=1))[-10:]
        forecast = model.predict(future)

        # Create future dates
        future_dates = pd.date_range(start=data.index[-1], periods=11)[1:]

        forecast_df = pd.DataFrame(forecast, index=future_dates, columns=['Prediction'])

        st.subheader("🔮 Future Prediction (Next 10 Days)")
        st.write(forecast_df)

        # Plot graph with prediction
        st.subheader("📉 Prediction Graph")

        plt.figure()
        plt.plot(data['Close'], label="Actual Price")
        plt.plot(forecast_df['Prediction'], label="Predicted Price")
        plt.legend()

        st.pyplot(plt)