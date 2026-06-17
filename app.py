import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Load dataset
df = pd.read_csv("data/sales_data.csv")

# Convert date
df["Date"] = pd.to_datetime(df["Date"])

# Create numerical index
df["Index"] = range(len(df))

X = df[["Index"]]
y = df["Sales"]

# Train model
model = LinearRegression()
model.fit(X, y)

# Future prediction
future = pd.DataFrame({"Index":[10,11,12,13,14]})

predictions = model.predict(future)

print("Future Sales Forecast:")
print(predictions)

# Plot
plt.figure(figsize=(8,5))
plt.plot(df["Index"], y, marker="o", label="Historical")

plt.plot(
    future["Index"],
    predictions,
    marker="o",
    linestyle="--",
    label="Forecast"
)

plt.xlabel("Month")
plt.ylabel("Sales")
plt.title("Sales Forecast")
plt.legend()

plt.savefig("outputs/forecast.png")
plt.show()
