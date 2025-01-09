# Convolutional Neural Network for Digit Recognition 

## CNN состоит из:
### Слой Input: 
Задает форму входных данных, ожидаемых моделью. 
### Сверточные слои (Conv2D): 
Эти слои выполняют операции свертки, которые помогают модели извлекать ключевые признаки из изображений, 
такие как края, текстуры и формы.
### Слои подвыборки (MaxPooling2D): 
Эти слои уменьшают размерность данных, сохраняя при этом важные признаки, 
что помогает ускорить обучение и снизить риск переобучения.
### Полносвязные слои (Dense): 
Эти слои отвечают за классификацию, принимая вектор признаков от предыдущих слоев 
и принимая решение о классе (цифре) изображения.

# Линейная модель CNN Sequential включает:
## 1. Слой Input
Входные данные — изображения размером 28x28 пикселей с одним каналом (градации серого).
## 2. Первый слой свертки (Conv2D):
Этот слой применяет 64 фильтра размером 3x3 к входным данным, извлекая важные признаки, такие как края и текстуры.
Активация relu.
## 3. Первый слой подвыборки (MaxPooling2D):
Уменьшает размерность данных, агрегируя значения в области 2x2 пикселя и выбирая максимальное значение в каждой области.
Помогает уменьшить количество параметров и вычислительные затраты, сохраняя важные признаки.
## 4. Второй слой свертки (Conv2D):
Применяет 128 фильтров размером 3x3 к данным, извлекая более сложные признаки.
Активация relu.
## 5. Второй слой подвыборки (MaxPooling2D):
Опять уменьшает размер данных, агрегируя значения в области 2x2 пикселя, выбирая максимальное значение в каждой области.
Продолжает уменьшать размер данных, сохраняя важные признаки, и помогает снизить вычислительные затраты.
## 6. Слой выравнивания (Flatten):
Преобразует многомерный массив данных в одномерный вектор. Для дальнейшего подключения к полносвязным (Dense) слоям.
## 7. Полносвязный слой (Dense):
Обрабатывает данные, используя 64 нейрона. 
В каждом нейроне происходит линейная комбинация входных данных и применение функции активации relu.
## 8. Слой Dropout
Используется для регуляризации нейронной сети - случайным образом обнуляет (выключает) определённую долю нейронов в слое 
во время обучения, что помогает предотвратить переобучение модели. 
Уменьшает зависимость модели от конкретных нейронов.
## 9. Выходной полносвязный слой (Dense):
Обрабатывает выходные данные и использует 10 нейронов, соответствующих 10 классам (цифры 0-9). 
Применяет функцию активации softmax для получения вероятностей классов.
(softmax - нормализует выходные значения, преобразуя их в вероятности классов, которые суммируются до 1)


x_train - cодержит данные, используемые для обучения модели нейросети.

x_test - cодержит данные, используемые для тестирования производительности обученной модели. 

### Параметры обучения: 

### Оптимизатор Adam (Adaptive Moment Estimation): 
адаптируется к обучению моделей, изменяя скорость обучения на основе момента первого и второго порядка. 

### Функция потерь sparse_categorical_crossentropy:
используется для задач многоклассовой классификации, где метки представлены в виде целых чисел. 
Измеряет разницу между предсказанными вероятностями и истинными метками. 

### Метрика accuracy, по которой будет оцениваться производительность модели. 
Показывает, какой процент предсказаний модели правильный. 
