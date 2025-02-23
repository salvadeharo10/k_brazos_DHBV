import numpy as np
from algorithms.algorithm import Algorithm

class UCB1(Algorithm):
    def __init__(self, k: int, c: float = 1):
        """
        Inicializa el algoritmo UCB1.
        
        :param k: Número de brazos.
        :param c: Parámetro para ajustar exploración.
        """
        super().__init__(k)
        self.c = c 
        self.total_counts = 0  # Contador total de selecciones

    def select_arm(self) -> int:
        """
        Selecciona un brazo basado en la política UCB1.
        
        :return: Índice del brazo seleccionado.
        """
        self.total_counts += 1  # Incrementar el conteo total de selecciones
        
        # Si hay brazos no seleccionados, seleccionamos uno de ellos primero
        for arm in range(self.k):
            if self.counts[arm] == 0:
                return arm
        
        # Aplicamos la fórmula de UCB1
        ucb_values = self.values + self.c * np.sqrt((2 * np.log(self.total_counts)) / self.counts)
        
        return np.argmax(ucb_values)

    def update(self, chosen_arm: int, reward: float):
        """
        Actualiza las recompensas promedio estimadas de cada brazo con la nueva recompensa obtenida.
        
        :param chosen_arm: Índice del brazo seleccionado.
        :param reward: Recompensa obtenida.
        """
        super().update(chosen_arm, reward)
