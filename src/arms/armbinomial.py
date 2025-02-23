"""
Module: arms/armbinomial.py
Description: Contains the implementation of the ArmBinomial class for the Binomial distribution arm.

"""

import numpy as np

from arms import Arm


class ArmBinomial(Arm):
    def __init__(self, n: int, p: float):
        """
        Inicializa el brazo con distribución binomial.

        :param n: Número de ensayos (entero positivo).
        :param p: Probabilidad de éxito en cada ensayo (0 <= p <= 1).
        """
        assert n > 0, "El número de ensayos n debe ser un entero positivo."
        assert 0 <= p <= 1, "La probabilidad p debe estar en el rango [0, 1]."

        self.n = n
        self.p = p

    def pull(self):
        """
        Genera una recompensa siguiendo una distribución binomial.

        :return: Número de éxitos obtenidos en n ensayos.
        """
        reward = np.random.binomial(self.n, self.p)
        return reward

    def get_expected_value(self) -> float:
        """
        Devuelve el valor esperado de la distribución binomial.

        :return: Valor esperado de la distribución.
        """
        return self.n * self.p

    def __str__(self):
        """
        Representación en cadena del brazo Binomial.

        :return: Descripción detallada del brazo Binomial.
        """
        return f"ArmBinomial(n={self.n}, p={self.p})"

    @classmethod
    def generate_arms(cls, k: int, n: int, seed: float, p_min: float = 0.1, p_max: float = 0.9):
        """
        Genera k brazos con probabilidades únicas en el rango [p_min, p_max], asegurando reproducibilidad.

        :param k: Número de brazos a generar.
        :param n: Número de ensayos por brazo.
        :param seed: Semilla para la generación de números aleatorios.
        :param p_min: Valor mínimo de la probabilidad de éxito.
        :param p_max: Valor máximo de la probabilidad de éxito.
        :return: Lista de brazos generados.
        """
        assert k > 0, "El número de brazos k debe ser mayor que 0."
        assert n > 0, "El número de ensayos n debe ser un entero positivo."
        assert 0 <= p_min < p_max <= 1, "Los valores de p_min y p_max deben estar en el rango [0, 1] y p_min < p_max."

        # Fijar la semilla para la reproducibilidad
        np.random.seed(seed)

        # Generar k valores únicos de p con dos decimales
        p_values = set()
        while len(p_values) < k:
            p = round(np.random.uniform(p_min, p_max), 2)
            p_values.add(p)

        p_values = list(p_values)
        arms = [cls(n, p) for p in p_values]  # Crear objetos de la clase ArmBinomial

        return arms
