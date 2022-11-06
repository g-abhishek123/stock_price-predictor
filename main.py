import prophet as prophet
import streamlit as st # creates a web app
from datetime import date
import yfinance as yf    # to download stock data
import fbprophet as Prophet    # for forecasting algorithm made by fb
from fbprophet.plot import plot_plotly  # to plot nice/interactive graphs
from plotly import graph_objs as go
from prophet import Prophet
from prophet.plot import plot_plotly

START = "2015-01-01"
TODAY = date.today().strftime("%Y-%m-%d")
st.title("Stock Prediction App")

stocks = ("TSLA","GOOG","MSFT","GME","NFLX")    # tupples for stocks
selected_stock = st.selectbox("Select dataset for prediction",stocks)

n_years = st.slider("Years of prediction",1,5)
period = n_years*365

@st.cache
def load_data(ticker):
    data = yf.download(ticker,START,TODAY)
    data.reset_index(inplace=True) # data in first column
    return data

data_load_state = st.text("Load data...")
data = load_data(selected_stock)
data_load_state.text("Loading data done")

st.subheader('Raw Data')
st.write(data.tail()) # tail signifies current date data

def plot_raw_data():
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data['Date'], y=data['Open'],name='stock open'))
    fig.add_trace(go.Scatter(x=data['Date'], y=data['Close'], name='stock close'))
    fig.layout.update(title_text="Time Series Data",xaxis_rangeslider_visible=True)
    st.plotly_chart(fig)
plot_raw_data()

# Forecasting
df_train = data[['Date', 'Close']]
df_train = data.rename(columns={"Date":"ds","Close":"y"})

m = Prophet() # m for model
m.fit(df_train)
future = m.make_future_dataframe(periods=period)
forecast = m.predict(future)

st.subheader('Forecast data')
st.write(forecast.tail())

st.write('forecast_data')
fig1 = plot_plotly(m,forecast)
st.plotly_chart(fig1)

st.write('forecast_data')
fig2 = m.plot_components(forecast)
st.write(fig2)
