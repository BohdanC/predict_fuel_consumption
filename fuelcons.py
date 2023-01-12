import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

data = pd.read_csv("automobileEDA.csv")
data.info()
arr = []
for i in data['engine-size']:
    j = i/62.02
    arr.append(round(j, 1))
data['engine-size'] = arr
data['num-of-cylinders'] = data['num-of-cylinders'].replace(['two', 'three', 'four', 'five',
                                                             'six','eight', 'twelve'],[2, 3, 4, 5,
                                                                                       6, 8, 12])

Data1 = data.copy()



Dataset = Data1[["length","width","height","curb-weight","num-of-cylinders","bore",
                 "engine-size","horsepower","peak-rpm","compression-ratio","city-L/100km"]]


print('length before deleting duplicate records', len(Dataset))
Dataset = Dataset.drop_duplicates()
print('length after deleting duplicate records', len(Dataset))


print("\nОбрані дані:\n")
print(Dataset)

Dataset.info()

print("\n-------------------------------------------------\n")
count_nan = Dataset.isna().sum()
print(count_nan)


plt.figure(figsize=(31,16))
correlation = Dataset.corr(method = 'pearson')
sns.heatmap(correlation,annot=True,cmap="RdYlGn",annot_kws={"size": 10})
plt.title('Correlation Matrix', fontsize=16)
plt.show()

plt.figure(figsize=(31,16))
plt.subplot(4,3,1)
sns.regplot(x = 'length', y = 'city-L/100km', data = Dataset)
plt.subplot(4,3,2)
sns.regplot(x = 'width', y = 'city-L/100km', data = Dataset)
plt.subplot(4,3,3)
sns.regplot(x = 'height', y = 'city-L/100km', data = Dataset)
plt.subplot(4,3,4)
sns.regplot(x = 'curb-weight', y = 'city-L/100km', data = Dataset)
plt.subplot(4,3,5)
sns.regplot(x = 'num-of-cylinders', y = 'city-L/100km', data = Dataset)
plt.subplot(4,3,6)
sns.regplot(x = 'bore', y = 'city-L/100km', data = Dataset)
plt.subplot(4,3,7)
sns.regplot(x = 'engine-size', y = 'city-L/100km', data = Dataset)
plt.subplot(4,3,8)
sns.regplot(x = 'horsepower', y = 'city-L/100km', data = Dataset)
plt.subplot(4,3,9)
sns.regplot(x = 'peak-rpm', y = 'city-L/100km', data = Dataset)
plt.subplot(4,3,10)
sns.regplot(x = 'compression-ratio', y = 'city-L/100km', data = Dataset)
plt.show()

#Modeling Multiple Linear Regression
Dataset.drop(columns = ['height', 'peak-rpm', 'compression-ratio'], axis=1, inplace = True)
Dataset.info()

y = Dataset[["city-L/100km"]]
x = Dataset.drop(columns = y)


def new_score (df):
    df_z = df.copy()
    for column in df_z.columns :
        df_z[column] = (df_z[column]-df_z[column].mean())/df_z[column].std()
    return df_z

x_new = new_score(x)


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(x_new, y,test_size=0.25,random_state = 0)

from sklearn.linear_model import LinearRegression

regressi = LinearRegression()
regressi.fit(X_train,y_train)

y_pred = regressi.predict(X_train)

prediction = pd.DataFrame(y_pred,columns = ["Predicted"])
print(prediction)

print("b_0",regressi.intercept_)
print("Коеф регресії", regressi.coef_)

from  sklearn import metrics

print('Cередня абсолютна похибка:', metrics.mean_absolute_error(y_train, y_pred))
print('Середня квадратична помилка:', metrics.mean_squared_error(y_train, y_pred))
print('Середньоквадратична помилка:', np.sqrt(metrics.mean_squared_error(y_train, y_pred)))

from sklearn.metrics import r2_score

print("R^2: {}".format(r2_score(y_true=y_train,y_pred=y_pred)))

prediction.reset_index(drop=True, inplace=True)
y_train.reset_index(drop=True, inplace=True)

Value = pd.concat( [prediction, y_train], axis=1)

residuals = y_train.values-prediction.values
residu = pd.DataFrame(residuals,columns = ["Residuals"])

residu.reset_index(drop=True, inplace=True)
Val_pre = pd.concat([Value,residu],axis = 1)
print(Val_pre)

mean_residuals = np.mean(residuals)
print("Cередня похибка {}".format(mean_residuals))

p = sns.scatterplot(data=Val_pre, x=Val_pre.Predicted, y=Val_pre.Residuals)

plt.xlabel('Прогнозовані результати')
plt.ylabel('Похибка')

plt.ylim(-10,10)
plt.xlim(0,25)

p = sns.regplot(x = [0,25],y=[0,0],color='blue')

p = plt.title('Графік похибок прогнозованих даних в навчальній вибірці')
plt.show()

y_pred_test = regressi.predict(X_test)

from  sklearn import metrics

print('Cередня абсолютна похибка тестових даних:', metrics.mean_absolute_error(y_test, y_pred_test))
print('Середня квадратична помилка тестових даних:', metrics.mean_squared_error(y_test, y_pred_test))
print('Середньоквадратична помилка тестових даних:', np.sqrt(metrics.mean_squared_error(y_test, y_pred_test)))

prediction_test = pd.DataFrame(y_pred_test,columns = ["Predicted"])


prediction.reset_index(drop=True, inplace=True)
y_test.reset_index(drop=True, inplace=True)

Value_test = pd.concat([prediction_test, y_test], axis=1)

residuals = y_test.values-prediction_test.values
residu = pd.DataFrame(residuals,columns = ["Residuals"])
residu.reset_index(drop=True, inplace=True)
Value_test = pd.concat([Value_test,residu],axis = 1)
print("\nПрогнозовані та реальні результати\n")
print(Value_test)

mean_residuals = np.mean(residuals)
print("Cередня похибка в тестових даних{}".format(mean_residuals))

p = sns.scatterplot(data=Value_test, x=Value_test.Predicted, y=Value_test.Residuals)

plt.xlabel('Прогнозовані результати')
plt.ylabel('Похибка')

plt.ylim(-10,10)
plt.xlim(0,25)


p = sns.regplot(x = [0,25],y=[0,0],color='blue')

p = plt.title('Графік похибок прогнозованих даних в тестовій вибірці')


plt.show()

fig, ax = plt.subplots()
ax.set_title('Порівняння прогнозованих результатів з реальними')
ax.plot(Value_test.Predicted,color='green', label='Прогнозовані результати')
ax.plot(Value_test['city-L/100km'],color='red', label='Реальні результати витрати палива')
ax.set_xlabel('Індекс')
ax.set_ylabel('Витрата палива')
ax.legend(loc='lower right')
plt.show()
