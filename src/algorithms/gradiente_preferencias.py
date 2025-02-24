import numpy as np
from algorithms.algorithm import Algorithm

class GradientBandit(Algorithm):
    def __init__(self, k: int, alpha: float = 0.1):
        """
        Inicializa el método de Gradiente de Preferencias.

        :param k: Número de brazos.
        :param alpha: Tasa de aprendizaje para actualizar las preferencias.
        """
        assert alpha > 0, "El parámetro alpha debe ser mayor que 0."

        super().__init__(k)
        self.alpha = alpha
        self.preferences = np.zeros(k)  # H_t(a), preferencias iniciales en 0
        self.average_reward = 0  # R̄_t, recompensa promedio

    def select_arm(self) -> int:
        """
        Selecciona un brazo basado en la política Softmax sobre las preferencias.
        
        :return: Índice del brazo seleccionado.
        """
        # Calculamos la distribución Softmax sobre las preferencias H_t(a)
        exp_prefs = np.exp(self.preferences - np.max(self.preferences))  # Evita problemas numéricos
        probabilities = exp_prefs / np.sum(exp_prefs)

        # Seleccionamos un brazo basado en la distribución de probabilidad
        chosen_arm = np.random.choice(self.k, p=probabilities)
        return chosen_arm

    def update(self, chosen_arm: int, reward: float):
        """
        Actualiza las preferencias de cada acción usando el gradiente de preferencias.

        :param chosen_arm: Índice del brazo seleccionado.
        :param reward: Recompensa obtenida tras seleccionar el brazo.
        """
        # Calculamos la política actual π_t(a) sobre las preferencias
        exp_prefs = np.exp(self.preferences - np.max(self.preferences))  # Evita desbordamientos
        probabilities = exp_prefs / np.sum(exp_prefs)

        # Actualizamos la recompensa promedio
        self.average_reward += (reward - self.average_reward) / (np.sum(self.counts) + 1)

        # Actualización de las preferencias con la regla de gradiente de preferencias
        for a in range(self.k):
            if a == chosen_arm:
                self.preferences[a] += self.alpha * (reward - self.average_reward) * (1 - probabilities[a])
            else:
                self.preferences[a] -= self.alpha * (reward - self.average_reward) * probabilities[a]
