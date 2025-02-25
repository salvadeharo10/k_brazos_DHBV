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

        super().__init__(k)  # Inicializa self.k y self.counts
        self.alpha = alpha
        self.preferences = np.zeros(k)  # H_t(a), preferencias iniciales en 0
        self.average_reward = 0  # R̄_t, recompensa promedio

    def select_arm(self) -> int:
        """
        Selecciona un brazo basado en la política Softmax sobre las preferencias.
        
        :return: Índice del brazo seleccionado.
        """
        # Aplicamos Softmax a las preferencias con normalización numérica
        exp_prefs = np.exp(self.preferences - np.max(self.preferences))  
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
        # Incrementamos el contador del brazo seleccionado
        self.counts[chosen_arm] += 1

        # Actualizamos la recompensa promedio global
        total_pulls = np.sum(self.counts)  # Número total de selecciones
        if total_pulls > 0:
            self.average_reward += (reward - self.average_reward) / total_pulls

        # Calculamos la política actual π_t(a) sobre las preferencias
        exp_prefs = np.exp(self.preferences - np.max(self.preferences))  
        probabilities = exp_prefs / np.sum(exp_prefs)

        # Aplicamos la regla de Gradiente de Preferencias
        for a in range(self.k):
            if a == chosen_arm:
                self.preferences[a] += self.alpha * (reward - self.average_reward) * (1 - probabilities[a])
            else:
                self.preferences[a] -= self.alpha * (reward - self.average_reward) * probabilities[a]

    def reset(self):
        """
        Reinicia el estado del algoritmo, poniendo a 0 las preferencias y la recompensa promedio.
        """
        self.preferences = np.zeros(self.k)  # Reiniciar preferencias H_t(a)
        self.average_reward = 0  # Reiniciar recompensa promedio
        self.counts = np.zeros(self.k, dtype=int)  # Reiniciar conteo de selecciones
