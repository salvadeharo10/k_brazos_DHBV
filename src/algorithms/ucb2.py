import numpy as np
from math import log, sqrt, ceil
from algorithms.algorithm import Algorithm

class UCB2(Algorithm):
    def __init__(self, k: int, alpha: float):
        """
        Inicializa el algoritmo UCB2 con k brazos y un parámetro de ajuste alpha.
        :param k: Número de brazos.
        :param alpha: Parámetro de ajuste para el balance entre exploración y explotación.
        """
        assert 0 < alpha < 1, "El parámetro alpha debe ser mayor que 0 y menor que 1."

        super().__init__(k)
        self.alpha = alpha
        self.epochs = np.zeros(k, dtype=int)  # Número de épocas de cada brazo

    def select_arm(self) -> int:
        """
        Selecciona el brazo con el índice máximo de UCB2.
        :return: Índice del brazo seleccionado.
        """
        t = np.sum(self.counts)  # Número total de selecciones realizadas
        ucb_values = np.zeros(self.k)
        
        for a in range(self.k):
            # Fórmula UCB2 corregida
            if self.counts[a] > 0:
                tau_k_a = ceil((1 + self.alpha) ** self.epochs[a])  # Calcula τ(k_a)
                ucb_values[a] = self.values[a] + sqrt(((1 + self.alpha) * log((np.e * t) / tau_k_a)) / (2 * tau_k_a))
            else:
                # Si el brazo no ha sido seleccionado, asigna un valor muy alto para explorarlo
                ucb_values[a] = float('inf')
        
        # Selecciona el brazo con el valor UCB2 más alto
        return np.argmax(ucb_values)

    def update(self, chosen_arm: int, reward: float):
        """
        Actualiza el valor promedio del brazo elegido y el número de veces que se ha seleccionado.
        :param chosen_arm: Índice del brazo que fue seleccionado.
        :param reward: Recompensa obtenida por seleccionar el brazo.
        """
        super().update(chosen_arm, reward)
        
        # Actualizamos el número de épocas (ka) para el brazo seleccionado
        self.epochs[chosen_arm] += 1

    def reset(self):
        """
        Reinicia el estado del algoritmo UCB2.
        """
        super().reset()
        self.epochs = np.zeros(self.k, dtype=int)  # Reinicia las épocas de cada brazo
