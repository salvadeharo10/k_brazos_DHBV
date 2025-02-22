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

from typing import List

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
    plt.legend(title="Algoritmos")
    plt.tight_layout()
    plt.show()


def plot_arm_statistics(arm_stats: Dict[str, Dict[str, np.ndarray]], algorithms: List[Algorithm]):
    """
    Genera gráficos separados de Selección de Arms: Ganancias vs Pérdidas para cada algoritmo.
    :param arm_stats: Diccionario con estadísticas de cada brazo por algoritmo (ganancia media y número de selecciones).
    :param algorithms: Lista de instancias de algoritmos comparados.
    """
    num_algorithms = len(algorithms)
    fig, axes = plt.subplots(1, num_algorithms, figsize=(num_algorithms * 5, 5), sharey=True)
    if num_algorithms == 1:
        axes = [axes]
    
    for i, algorithm in enumerate(algorithms):
        arm_means = arm_stats[algorithm.name]['means']
        arm_counts = arm_stats[algorithm.name]['counts']
        num_arms = len(arm_means)
        
        bars = axes[i].bar(range(num_arms), arm_means, tick_label=[f"{j}\n({arm_counts[j]} selecciones)" for j in range(num_arms)])
        
        optimal_arm = np.argmax(arm_means)
        bars[optimal_arm].set_color('r')
        
        axes[i].set_xlabel("Brazo")
        axes[i].set_ylabel("Ganancia media")
        axes[i].set_title(f"{algorithm.name}: Promedio de Ganancias por Brazo")
        axes[i].grid(axis='y')
    
    plt.show()


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

