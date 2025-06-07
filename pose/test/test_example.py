import unittest

import numpy as np
import torch
from features.extract_lengths import (
    KeypointConnections,
    KeypointScheme,
    extract_lengths,
    extract_near_cosines,
)
from mvp.feature_bake import decay_filt, derive_first, derive_second
from mvp.models.MLP.MLP import MLPClassifier_deep


class MyTest(unittest.TestCase):
    def test_ok(self):
        assert 2 + 2 == 4

    def test_not_ok(self):
        assert 2 + 2 == 5

    """Тестирование функций для извлечения признаков из ключевых точек с использованием равностороннего треугольника."""

    def setUp(self):
        """Настройка равностороннего треугольника со стороной длиной 1 для тестирования."""
        # Сохраняем оригинальные соединения для восстановления после тестов
        self.original_connections = KeypointConnections[KeypointScheme._17].copy()

        # Изменяем соединения для формирования треугольника, используя точки 0,
        # 1 и 2
        KeypointConnections[KeypointScheme._17] = [
            [0, 1],
            [1, 2],
            [2, 0],
        ]

        # Создаем ключевые точки для одного образца в пакете
        self.keypoints = np.zeros(
            (1, 17, 3)
        )  # Используем 17 ключевых точек, как ожидает схема

        # Устанавливаем координаты для равностороннего треугольника со стороной
        # длиной 1
        self.keypoints[0, 0, :] = [0, 0, 0]
        self.keypoints[0, 1, :] = [1, 0, 0]
        self.keypoints[0, 2, :] = [0.5, np.sqrt(3) / 2, 0]

    def tearDown(self):
        """Восстанавливаем исходные соединения после тестирования."""
        KeypointConnections[KeypointScheme._17] = self.original_connections

    def test_extract_lengths(self):
        """Тест, проверяющий, что extract_lengths возвращает правильные расстояния."""
        lengths = extract_lengths(self.keypoints, KeypointScheme._17)

        # Для равностороннего треугольника со стороной длиной 1 все расстояния
        # должны быть равны 1
        expected_lengths = np.ones((1, 3))
        np.testing.assert_almost_equal(lengths, expected_lengths)

    def test_extract_near_cosines(self):
        """Тест, проверяющий, что extract_near_cosines возвращает правильные косинусы."""
        cosines = extract_near_cosines(self.keypoints, KeypointScheme._17)

        # Для равностороннего треугольника угол при каждой вершине составляет 60 градусов
        # Косинус 60 градусов равен 0.5
        expected_cosines = np.full((1, 3), 0.5)
        np.testing.assert_almost_equal(cosines, expected_cosines)

    def test_derive_cosine_function(self):
        """
        Тест, проверяющий, что функция derive правильно вычисляет производные косинуса.
        Первая производная cos(t) должна быть -sin(t)
        Вторая производная cos(t) должна быть -cos(t)
        """
        # Создаем косинусную волну на протяжении нескольких периодов
        num_points = 1000
        t = np.linspace(0, 4 * np.pi, num_points)

        # Создаем батч с одним образцом, значениями косинуса и одним измерением признаков
        # Форма: (размер_батча=1, временные_шаги=num_points, признаки=1)
        cosine_data = np.cos(t).reshape(1, -1, 1)

        # Вычисляем производные
        first_derivative = derive_first(cosine_data)
        second_derivative = derive_second(cosine_data)

        # Изменяем форму для более удобного сравнения
        cosine_flat = cosine_data.flatten()[20:-20]  # Удаляем краевые эффекты
        first_deriv_flat = first_derivative.flatten()[20:-20]
        second_deriv_flat = second_derivative.flatten()[20:-20]

        # Ожидаемые теоретические производные
        expected_first_deriv = -np.sin(t)[20:-20]
        expected_second_deriv = -np.cos(t)[20:-20]

        # Проверяем фазовые соотношения:

        # 1. Первая производная должна быть приблизительно -sin(t)
        # Это означает, что она должна быть сдвинута на 90° относительно исходного косинуса
        # (корреляция близка к нулю)
        correlation_first = np.corrcoef(cosine_flat, first_deriv_flat)[0, 1]
        self.assertAlmostEqual(correlation_first, 0, delta=0.1)

        # Первая производная должна хорошо коррелировать с теоретической
        # -sin(t)
        correlation_with_theory = np.corrcoef(first_deriv_flat, expected_first_deriv)[
            0, 1
        ]
        self.assertGreater(correlation_with_theory, 0.9)

        # 2. Вторая производная должна быть приблизительно -cos(t)
        # Это означает, что она должна быть сдвинута на 180° относительно исходного косинуса
        # (корреляция близка к -1)
        correlation_second = np.corrcoef(cosine_flat, second_deriv_flat)[0, 1]
        self.assertAlmostEqual(correlation_second, -1, delta=0.1)

        # Вторая производная должна хорошо коррелировать с теоретической
        # -cos(t)
        correlation_with_theory = np.corrcoef(second_deriv_flat, expected_second_deriv)[
            0, 1
        ]
        self.assertGreater(correlation_with_theory, 0.9)

    def test_decay_filt(self):
        """
        Тест функции decay_filt.
        Проверяет:
        1. На нулевом массиве результат будет нулевым
        2. Для массива с первым элементом 1, остальные элементы должны
        формировать затухающую экспоненту с коэффициентом (1-K)
        """
        # Параметры для тестирования
        K_values = [0.007, 0.05]
        time_steps = 100

        for K in K_values:
            # Тест 1: Нулевой массив - одномерный по времени
            zero_array = np.zeros(time_steps)
            filtered_zeros = decay_filt(zero_array, K)
            np.testing.assert_array_almost_equal(filtered_zeros, zero_array)

            # Тест 2: Импульсный массив - одномерный по времени
            impulse_array = np.zeros(time_steps)
            impulse_array[0] = 1.0

            filtered_impulse = decay_filt(impulse_array, K)

            # Вычисляем ожидаемую затухающую экспоненту
            expected = np.zeros(time_steps)
            for i in range(time_steps):
                expected[i] = (1 - K) ** i

            np.testing.assert_array_almost_equal(filtered_impulse, expected)

            # Тест 3: Многомерный массив (проверяем фильтрацию по каждому
            # измерению)
            features = 3
            multi_array = np.zeros((time_steps, features))
            multi_array[0] = 1.0  # Устанавливаем первую строку в 1

            filtered_multi = decay_filt(multi_array, K)

            # Ожидаемый результат - затухающая экспонента в каждом столбце
            expected_multi = np.zeros((time_steps, features))
            for i in range(time_steps):
                expected_multi[i] = (1 - K) ** i

            np.testing.assert_array_almost_equal(filtered_multi, expected_multi)

    def test_mlp_classifier_deep_dimensions(self):
        """
        Тест для проверки корректности размерностей входного и выходного тензоров
        модели MLPClassifier_deep.
        """
        # Задаем параметры модели
        input_size = 128
        hidden_size = 64
        hidden_size_2 = 32
        num_classes = 5

        # Создаем экземпляр модели
        model = MLPClassifier_deep(input_size, hidden_size, hidden_size_2, num_classes)

        # Параметры входного тензора
        batch_size = 10
        # Нужна последовательность, т.к. в forward используется индексация [:,
        # 0, :]
        seq_length = 3

        # Создаем случайный входной тензор
        input_tensor = torch.randn(batch_size, seq_length, input_size)

        # Переводим модель в режим оценки для отключения dropout
        model.eval()

        # Пропускаем тензор через модель
        with torch.no_grad():
            output_tensor = model(input_tensor)

        # Проверяем размерности выходного тензора
        expected_output_shape = (batch_size, num_classes)

        # Проверяем, что размерности соответствуют ожидаемым
        self.assertEqual(
            output_tensor.shape,
            expected_output_shape,
            f"Выходной тензор имеет форму {output_tensor.shape}, "
            f"ожидалась форма {expected_output_shape}",
        )

        # Также можно проверить, что выходной слой действительно имеет
        # num_classes выходов
        self.assertEqual(
            model.fc4.out_features,
            num_classes,
            f"Выходной слой должен иметь {num_classes} выходов, но имеет {model.fc4.out_features}",
        )

        # Проверяем, что входной слой имеет правильную размерность
        self.assertEqual(
            model.fc0.in_features,
            input_size,
            f"Входной слой должен принимать {input_size} входов, но принимает {model.fc0.in_features}",
        )
