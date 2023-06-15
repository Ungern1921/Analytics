import numpy as np         # библиотека для матриц и математики
import pandas as pd        # библиотека дл работы с табличками
from scipy import stats    # модуль для работы со статистикой


# библиотеки для визуализации
import matplotlib.pyplot as plt
import seaborn as sns

plt.style.use('ggplot')    # стиль графиков

norm_rv = stats.norm(loc=0, scale=1)  # задали генератор с нормальным распределением

sample = norm_rv.rvs(1000)  # сгенерируем 1000 значений
print('first 10 value of generator:' ,sample[:10], '\n')

print ('f(1) = ', norm_rv.pdf(1), '\n') # плотность генератора от единицы

x = np.linspace(-3, 3, 100)
pdf = norm_rv.pdf(x)

plt.plot(x, pdf)
plt.ylabel('$f(x)$')
plt.xlabel('$x$')

# На ней же нарисуем f(1)
plt.scatter([1,2], [norm_rv.pdf(1), norm_rv.pdf(2)], color="blue");

plt.show()

print ('F(1) = ', norm_rv.cdf(1), '\n') # функция распределения генератора от единицы

x = np.linspace(-3, 3, 100)
pdf = norm_rv.pdf(x)

plt.plot(x, pdf)
plt.ylabel('$f(x)$')
plt.xlabel('$x$')

# На ней же нарисуем f(1)
plt.scatter([1], [norm_rv.pdf(1)], color="blue");

# на ту же картинку добавили новую часть, штриховку
xq = np.linspace(-3, 1, 100)
yq = norm_rv.pdf(xq)
plt.fill_between(xq, 0, yq, color='blue', alpha=0.2)

plt.axvline(1, color='blue', linestyle="--", lw=2);

plt.show()

x = np.linspace(-3, 3, 100)
cdf = norm_rv.cdf(x)

plt.plot(x, cdf)
plt.ylabel('$f(x)$')
plt.xlabel('$x$')

# На ней же нарисуем F(1)
plt.scatter([1], [norm_rv.cdf(1)], color="blue");

plt.show()

print ('P(1 < X < 3) = F(3) - F(1) =', norm_rv.cdf(3) - norm_rv.cdf(1), '\n') # вероятность, что наша сгенерируемая нормальная случайная величина X между единицей и тройкой

x = np.linspace(-5, 5, 100)
pdf = norm_rv.pdf(x)

plt.plot(x, pdf)
plt.ylabel('$f(x)$')
plt.xlabel('$x$')

# На ней же нарисуем f(1)
plt.scatter([1, 3], [norm_rv.pdf(1), norm_rv.pdf(3)], color="blue");

# на ту же картинку добавили новую часть, штриховку
xq = np.linspace(1, 3)
yq = norm_rv.pdf(xq)
plt.fill_between(xq, 0, yq, color='blue', alpha=0.2)

plt.axvline(1, color='blue', linestyle="--", lw=2)
plt.axvline(3, color='blue', linestyle="--", lw=2);

plt.show()

q = norm_rv.ppf(0.1)
q = norm_rv.ppf(0.5)  # медиана

print('mediana = ', q, '\n');



x = np.linspace(-3, 3, 100)
pdf = norm_rv.pdf(x)

plt.plot(x, pdf)
plt.ylabel('$f(x)$')
plt.xlabel('$x$')

xq = np.linspace(-3, q)
yq = norm_rv.pdf(xq)
plt.fill_between(xq, 0, yq, color='blue', alpha=0.2)

plt.axvline(q, color='blue', linestyle="--", lw=2)

y_max = plt.ylim()[1]
plt.text(q + 0.1, 0.8*y_max, round(q,2), color='blue', fontsize=16)

plt.show()

exp_rv = stats.expon(scale=5)

print('exp_gen : ', exp_rv.rvs(5), '\n');

print('viborky : ', sample[:10], '\n');

print('выборочное среднее = ', np.mean(sample), '\n');

print('выборочная дисперсия = ', np.var(sample) , '\n');

print('выборочное стандартное отклонение = ', np.std(sample)  , '\n');

print('выборочная медиана = ', np.median(sample)  , '\n');

plt.hist(sample, bins=1000);  # bins отвечает за число столбцов

plt.show()



x = np.linspace(-3, 3, 100)
pdf = norm_rv.pdf(x)

# плотность 
plt.plot(x, pdf, lw=3)

# гистограмма, параметр density отнормировал её. 
plt.hist(sample, bins=30, density=True);

plt.ylabel('$f(x)$')
plt.xlabel('$x$');

plt.show()

# для построения ECDF используем библиотеку statsmodels
from statsmodels.distributions.empirical_distribution import ECDF

ecdf = ECDF(sample)   # строим эмпирическую функцию по выборке

plt.step(ecdf.x, ecdf.y)
plt.ylabel('$F(x)$', fontsize=20)
plt.xlabel('$x$', fontsize=20);

plt.show()

x = np.linspace(-3, 3, 100)

# теоретическа cdf 
cdf = norm_rv.cdf(x)
plt.plot(x, cdf, label='theoretical CDF')

# эмпирическая сdf
ecdf = ECDF(sample)
plt.step(ecdf.x, ecdf.y, label='empirical CDF')

plt.ylabel('$F(x)$')
plt.xlabel('$x$')
plt.legend(loc='upper left');

plt.show()

np.arange(1,11)

np.random.choice(np.arange(1,11), 
                 size=5,
                 replace=False) # выборка без повторений
                 


np.random.choice(np.arange(1,11), 
                 size=5,
                replace=True) # с повторениями

np.random.choice(['карась', 'плотва', 'щука'], 
                 size=10, 
                 p=[0.5, 0.2, 0.3]) # с повторениями
                 
rv = stats.norm(loc=3, scale=2)
rv.rvs(5)

stats.norm(loc=3, scale=2).rvs(5)

stats.norm(loc=3, scale=2).rvs(5, random_state=111)



