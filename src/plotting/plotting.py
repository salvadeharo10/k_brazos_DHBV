"""
Module: plotting/plotting.py
Description: Contiene funciones para generar gráficas de comparación de algoritmos.

Author: Luis Daniel Hernández Molinero
Email: ldaniel@um.es
Date: 2025/01/29

This software is licensed under the GNU General Public License v3.0 (GPL-3.0),
with the additional restriction that it may not be used for commercial purposes.

For more details about GPL-3.0: https://www.gnu.org/licenses/gpl-3.0.html
"""

from typing import List, Dict

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from algorithms import Algorithm, EpsilonGreedy


def get_algorithm_label(algo: Algorithm) -> str:
    """
    Genera una etiqueta descriptiva para el algoritmo incluyendo sus parámetros.

    :param algo: Instancia de un algoritmo.
    :type algo: Algorithm
    :return: Cadena descriptiva para el algoritmo.
    :rtype: str
    """
    label = type(algo).__name__
    if isinstance(algo, EpsilonGreedy):
        label += f" (epsilon={algo.epsilon})"
    # elif isinstance(algo, OtroAlgoritmo):
    #     label += f" (parametro={algo.parametro})"
    # Añadir más condiciones para otros algoritmos aquí
    else:
        raise ValueError("El algoritmo debe ser de la clase Algorithm o una subclase.")
    return label


def plot_average_rewards(steps: int, rewards: np.ndarray, algorithms: List[Algorithm]):
    """
    Genera la gráfica de Recompensa Promedio vs Pasos de Tiempo.

    :param steps: Número de pasos de tiempo.
    :param rewards: Matriz de recompensas promedio.
    :param algorithms: Lista de instancias de algoritmos comparados.
    """
    sns.set_theme(style="whitegrid", palette="muted", font_scale=1.2)

    plt.figure(figsize=(14, 7))
    for idx, algo in enumerate(algorithms):
        label = get_algorithm_label(algo)
        plt.plot(range(steps), rewards[idx], label=label, linewidth=2)

    plt.xlabel('Pasos de Tiempo', fontsize=14)
    plt.ylabel('Recompensa Promedio', fontsize=14)
    plt.title('Recompensa Promedio vs Pasos de Tiempo', fontsize=16)
    plt.legend(title='Algoritmos')
    plt.tight_layout()
    plt.show()


def plot_optimal_selections(steps: int, optimal_selections: np.ndarray, algorithms: List[Algorithm]):
    """
    Genera la gráfica de Porcentaje de Selección del Brazo Óptimo vs Pasos de Tiempo.

    :param steps: Número de pasos de tiempo.
    :param optimal_selections: Matriz de porcentaje de selecciones óptimas.
    :param algorithms: Lista de instancias de algoritmos comparados.
    """
    sns.set_theme(style="whitegrid", palette="muted", font_scale=1.2)

    plt.figure(figsize=(14, 7))
    for idx, algo in enumerate(algorithms):
        label = get_algorithm_label(algo)
        plt.plot(range(steps), optimal_selections[idx], label=label, linewidth=2)

    plt.xlabel("Pasos de Tiempo", fontsize=14)
    plt.ylabel("Porcentaje de Selección del Brazo Óptimo", fontsize=14)
    plt.title("Porcentaje de Selección del Brazo Óptimo vs Pasos de Tiempo", fontsize=16)
    plt.yticks(np.arange(0, 101, 10))  # Establecer ticks del eje Y de 0 a 100 en saltos de 10
    plt.legend(title="Algoritmos")
    plt.tight_layout()
    plt.show()


def plot_arm_statistics(arm_data, algorithms):
    """
    Genera gráficos individuales para cada algoritmo mostrando el desempeño de los brazos.

    :param arm_data: Lista de diccionarios con estadísticas de cada brazo por algoritmo.
    :param algorithms: Lista de instancias de algoritmos evaluados.
    """
    num_algorithms = len(algorithms)

    for algo_idx, (stats, algorithm) in enumerate(zip(arm_data, algorithms)):
        fig, ax = plt.subplots(figsize=(8, 6))  # Crear un nuevo gráfico para cada algoritmo

        # Extraer información relevante de cada brazo
        arm_indices = [entry["arm"] for entry in stats]
        avg_rewards = [entry["average_reward"] for entry in stats]
        selection_counts = [entry["times_selected"] for entry in stats]
        is_optimal = [entry["optimal"] for entry in stats]

        # Etiquetas del eje X con porcentaje de selección y marcando el óptimo
        x_labels = [f"Brazo {idx} - {round((count/1000)*100, 2)}% - {'Óptimo' if opt else 'No'}" 
                    for idx, count, opt in zip(arm_indices, selection_counts, is_optimal)]

        # Asignación de colores: verde para el brazo óptimo, azul para los demás
        bar_colors = ["green" if opt else "blue" for opt in is_optimal]

        # Generar el gráfico de barras
        ax.bar(x_labels, avg_rewards, color=bar_colors)

        # Configuración de los ejes y el título
        ax.set_xlabel("Frecuencia de Selección del Brazo")
        ax.set_ylabel("Recompensa Media")
        ax.set_title(f"Análisis de Selección de Brazos - {algorithm.__class__.__name__}")
        ax.set_xticklabels(x_labels, rotation=45, ha="right", fontsize=10)

        # Ajustar diseño y mostrar
        plt.tight_layout()
        plt.show()

        # Imprimir los valores de recompensa promedio por brazo
        for i, reward in enumerate(avg_rewards, 1):
            print(f"Recompensa Media del Brazo {i}: {reward}")


def plot_regret(steps: int, regret_accumulated: np.ndarray, algorithms: List[Algorithm], theoretical_bound=None):
    """
    Genera la gráfica de Regret Acumulado vs Pasos de Tiempo
    :param steps: Número de pasos de tiempo.
    :param regret_accumulated: Matriz de regret acumulado (algoritmos x pasos).
    :param algorithms: Lista de instancias de algoritmos comparados.
    :param theoretical_bound: (Opcional) Cota teórica del regret C * ln(T) para comparación.
    """
    plt.figure(figsize=(10, 6))
    for i, algorithm in enumerate(algorithms):
        plt.plot(range(steps), regret_accumulated[i], label=algorithm.name)
    
    if theoretical_bound is not None:
        plt.plot(range(steps), theoretical_bound, 'k--', label="Cota Teórica C * ln(T)")
    
    plt.xlabel("Pasos de tiempo")
    plt.ylabel("Regret Acumulado")
    plt.title("Evolución del Regret Acumulado")
    plt.legend()
    plt.grid(True)
    plt.show()

