from sklearn.svm import SVR
from mlens.ensemble import BlendEnsemble, SuperLearner
from sklearn.metrics import f1_score
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import VotingRegressor
import numpy as np
from utils.financial_data_preprocesing import Financial_Data_Preprocesing as FDP
import matplotlib.pyplot as plt


fdp = FDP("^NYA", start = (2012,1,3), end = (2018,12,31))
fdp.fit()
fdp.transform()


X_train = fdp.df.iloc[30:3*fdp.n//4]
y_train = fdp.df["Close"].iloc[31:3*fdp.n//4+1]

X_test = fdp.df.iloc[3*fdp.n//4:fdp.n-1]
y_test = fdp.df["Close"].iloc[3*fdp.n//4+1:fdp.n]


# Stacking
rng3 = np.random.RandomState(0)
def f1(y, p): return f1_score(y, p, average='micro')
ensemble = SuperLearner(scorer=f1, random_state=rng3)
ensemble.add([SVR(kernel="rbf", C=100), SVR(kernel="poly")])
ensemble.add_meta(LinearRegression())
ensemble.fit(X_train, y_train)
y_predict = ensemble.predict(X_test)


plt.title("Close Price")
plt.plot(
    [*y_train.index, *y_test.index],
    fdp.min_max_untransformation([*y_train.values, *y_test.values], "Close"),
    color = "red",
    label="true value"
)
plt.plot(
    y_test.index,
    fdp.min_max_untransformation(y_predict, "Close"),
    color = "blue",
    label = "predict value"
)
plt.legend(loc='best')
plt.show()