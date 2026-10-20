import pandas as pd
import matplotlib.pyplot as plt

df = pd.DataFrame({
    'Age': [20, 21, None, 23],
    'Marks': [60, 70, 80, 90]
})

df.fillna(df.mean(), inplace=True)
print(df)
print(df.describe())

df.plot(kind='bar')
plt.show()