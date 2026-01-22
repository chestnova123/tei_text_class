import pandas as pd
from sklearn.preprocessing import OneHotEncoder
import seaborn as sb

data_in = pd.read_csv("training_data.csv")
data_in.head()

# check counts of rendition values
data_in.rendition.value_counts()

# remove renditions that are very rare
counts = data_in["rendition"].value_counts()
data = data_in[data_in["rendition"].isin(counts[counts > 100].index)]

data.describe()
data_in.describe()
data.rendition.value_counts()

data.to_csv("data_cleaned.csv")

# one-hot encoding for rendition values
enc = OneHotEncoder(handle_unknown="ignore")
enc.fit(data[["rendition"]])
enc.categories_

data_enc = enc.transform(data[["rendition"]])

print(type(data_enc))
print(data_enc.shape)

renditions = pd.DataFrame(
    data=data_enc.toarray(), index=data.index, columns=enc.categories_[0]
)

data_encoded = pd.concat([data.drop(columns="rendition"), renditions], axis=1)

data_encoded.head()

# plot relationships
sb.pairplot(data_encoded)

# save data
data_encoded.to_csv("data_encoded.csv")
