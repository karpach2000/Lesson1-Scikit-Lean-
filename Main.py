#import seaborn as sns
import matplotlib.pyplot as plt 
import numpy as np 
import seaborn as sns; sns.set()
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
from sklearn.datasets import load_digits


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
plt.scatter(x,y)
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
Xtrain, Xtest, ytrain, ytest = train_test_split(X_iris, y_iris, random_state = 1)#обучающая последовательность и контрольная последовательность
model = GaussianNB()#создаем экземпляр модели
model.fit(Xtrain, ytrain)#обучаем модель  на данных
y_model = model.predict(Xtest)#предсказываем значения для новых данных
print('Процент предсказанных меток, соответсвующих истиному значению: ', accuracy_score(ytest, y_model))

print('Пример обучения без учителя')
model = PCA(n_components=2)
model.fit(X_iris)
X_2D = model.transform(X_iris)
iris['PCA1'] = X_2D[:,0]
iris['PCA2'] = X_2D[:,1]
sns.lmplot("PCA1", "PCA2", hue = 'species', data = iris, fit_reg = False)
plt.title('Проекция данных набора Iris на двумерное пространство.')
plt.show()


#кластеризаация
print('Обучение без учителя: кластеризация наборов данных Iris')
model = GaussianMixture(n_components = 3, covariance_type = 'full')#создаем экземпляр модели с гиперпараметрами
model.fit(X_iris)
y_gmm = model.predict(X_iris)
iris['cluster'] = y_gmm
sns.lmplot("PCA1", "PCA2", data = iris, hue = 'species', col = 'cluster', fit_reg = False)
plt.title('Проекция данных набора Iris на двумерное пространство.')
plt.show()

#анализ рукописных цифр
print('анализ рукописных цифр')
digits = load_digits()
#digits = images.shape#массив из 1797 выборок картинок 8 на 8 пикселей
fig, axes = plt.subplots(10, 10, figsize = (8, 8), subplot_kw={'xticks':[], 'yticks':[]}, gridspec_kw=dict(hspace = 0.1, wspace= 0.1))
for i, ax in enumerate(axes.flat):
    ax.imshow(digits.images[i], cmap = 'binary', interpolation = 'nearest')
    ax.text(0.05, 0.05, str(digits.target[i]), transform = ax.transAxes, color = 'green')
plt.show()
