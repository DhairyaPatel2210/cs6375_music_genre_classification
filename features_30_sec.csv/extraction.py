import pandas as pd
from matplotlib import pyplot as plt
df = pd.read_csv("./features_30_sec.csv")
filenames = df["filename"]
c = {}
for file in filenames:
    genre = file.split(".")[0]
    c[genre] = c.get(genre,0) + 1


# Extract genre labels and frequencies
genres = list(c.keys())
frequencies = list(c.values())

# Create a bar plot (histogram)
plt.figure(figsize=(10, 6))
plt.bar(genres, frequencies, color='grey')
plt.xlabel('Genre')
plt.ylabel('Frequency')
plt.title('Data Distribution')
plt.xticks(rotation=45)  # Rotate x-axis labels for better readability

plt.tight_layout()
plt.show()
# print(c)