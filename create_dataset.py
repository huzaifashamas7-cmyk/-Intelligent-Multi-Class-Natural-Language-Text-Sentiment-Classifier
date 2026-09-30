import pandas as pd

data = [
    # Positive
    ["I love this product", "Positive"],
    ["This product is amazing", "Positive"],
    ["The service was excellent", "Positive"],
    ["I am very happy with my purchase", "Positive"],
    ["The food was delicious", "Positive"],
    ["This movie was fantastic", "Positive"],
    ["The app is very useful", "Positive"],
    ["Customer support was very helpful", "Positive"],
    ["The phone works perfectly", "Positive"],
    ["I really enjoyed this experience", "Positive"],

    # Negative
    ["I hate this product", "Negative"],
    ["This product is terrible", "Negative"],
    ["The service was awful", "Negative"],
    ["I am very unhappy with my purchase", "Negative"],
    ["The food was disgusting", "Negative"],
    ["This movie was boring", "Negative"],
    ["The app is difficult to use", "Negative"],
    ["Customer support was not helpful", "Negative"],
    ["The phone keeps crashing", "Negative"],
    ["I had a very bad experience", "Negative"],

    # Neutral
    ["The product is available in three colors", "Neutral"],
    ["The phone has 8GB RAM", "Neutral"],
    ["The meeting starts at 3 PM", "Neutral"],
    ["The movie is two hours long", "Neutral"],
    ["The laptop has a 15 inch display", "Neutral"],
    ["The order was placed on Monday", "Neutral"],
    ["The course has twelve lectures", "Neutral"],
    ["The package contains two items", "Neutral"],
    ["The phone weighs 190 grams", "Neutral"],
    ["The company was founded in 2010", "Neutral"]
]

# Create a DataFrame
df = pd.DataFrame(data, columns=["text", "sentiment"])

# Save the dataset as CSV
df.to_csv("dataset.csv", index=False)

print("Dataset created successfully!")
print()
print(df)