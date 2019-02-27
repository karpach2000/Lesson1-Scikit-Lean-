import seaborn as sns
import matplotlib.pyplot as plt 
import numpy as np 
import seaborn as sns; sns.set()
from sklearn.linear_model import LinearRegression


print("Каждая строка в данной таблице соответсвует одному цветку\n")
iris = sns.load_dataset('iris')
print(iris.head())

sns.pairplot(iris, hue = 'species', size=1.5)
plt.title('Визуализация набора данных Iris')
plt.show()
X_iris = iris.drop('species', axis = 1)
print('Размерность матрицы признаков ',  X_iris.shape)
y_iris = iris['species']
print('Размерность целевого массива ', y_iris.shape)

print('Пример обучения с линейной регрессией\n')
rng =np.random.RandomState(42)
x = 10 * rng.rand(50)
y = 2 * x - 1 + rng.rand(50)
plt.title('Данные для линейной регресси')
plt.scatter(x,y);
plt.show()

model = LinearRegression(copy_X = True, fit_intercept = True, n_jobs = 1, normalize=False)#fit_intercept = True хотим выполнить подбор точк пересечения с осью координат
X = x[:, np.newaxis]#Формирование из данных матриц признаков целевого вектора
print("Начало обучения модели....")
model.fit(X, y)
print("Окончание обучения модели....")
print("Угловой коэффицент ", model.coef_)
print("Точка пересечения с осью координат", model.intercept_)

print('Предсказание меток для новых данных')
xfit = np.linspace(-1, 11)
print('новые данные:\n', xfit)
Xfit = xfit[:, np.newaxis]
yfit = model.predict(Xfit)
plt.title('Простая линейная регрессионная аппроксимация наших данных')
plt.scatter(x, y)#График исходных данных
plt.plot(xfit, yfit)#обученная модель
plt.show()

print('Пример обучения с учителем')
