import numpy as np
from algorithms.algorithm import Algorithm

class Softmax(Algorithm):
    def __init__(self, k: int, tau: float = 0.1):
        """
        Inicializa el algoritmo Softmax para el problema del bandido multibrazo.

        :param k: Número de brazos.
        :param tau: Parámetro de temperatura para el método Softmax.
        :raises ValueError: Si tau no es positivo.
        """
        assert tau > 0, "El parámetro tau debe ser mayor que 0."

        super().init(k)
        self.tau = tau

    def select_arm(self) -> int:
        """
        Selecciona un brazo basado en la política Softmax.

        :return: Índice del brazo seleccionado.
        """
        # Calculamos la distribución Softmax
        exp_q = np.exp(self.values / self.tau)
        probabilities = exp_q / np.sum(exp_q)

        # Seleccionamos un brazo basado en la distribución de probabilidad
        chosen_arm = np.random.choice(self.k, p=probabilities)
        return chosen_arm
    
    def update(self, chosen_arm: int, reward: float):
        """
        Actualiza la recompensa promedio Q_t(a) según la fórmula
        """
        # Actualizamos el valor estimado de la recompensa del brazo seleccion
        self.values[chosen_arm] = self.values[chosen_arm] + (1.0 / self.k) * (reward - self.values[chosen_arm])
                
