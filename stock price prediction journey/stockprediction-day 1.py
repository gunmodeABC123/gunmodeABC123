import yfinance as yf
import matplotlib.pyplot as plt
ticker = 'AAPL'
stock = yf.Ticker(ticker)

data = stock.history(period='1d', interval='1m')
time = []
open_prices = data['Open'].tolist()
temp_time = data.index.tolist()
for i in range(len(temp_time)):
    time.append(int(str(temp_time[i]).split(" ")[1].split("-")[0].split(":")[0])*60 + int(str(temp_time[i]).split(" ")[1].split("-")[0].split(":")[1]))
print(len(time),len(open_prices))

plt.plot(time, open_prices)
plt.show()

