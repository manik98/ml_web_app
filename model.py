from sklearn import datasets
myiris = datasets.load_iris()
x = myiris.data
y = myiris.target

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range=(0, 1))
x_scaled = scaler.fit_transform(x)
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x_scaled, y, test_size=0.2, random_state=42)

from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(x_train, y_train)

import pickle

pickle.dump(knn, open('model.pkl','wb'))

model = pickle.load(open('model.pkl','rb'))
