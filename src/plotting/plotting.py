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


import matplotlib.pyplot as plt
import numpy as np

def plot_arm_statistics(arm_data, algorithms):
    """
    Genera gráficos individuales para cada algoritmo mostrando el desempeño de los brazos.

    :param arm_data: Lista de listas de diccionarios con estadísticas de cada brazo por algoritmo.
    :param algorithms: Lista de instancias de algoritmos evaluados.
    """
    num_algorithms = len(algorithms)
    
    for algo_idx, (stats, algorithm) in enumerate(zip(arm_data, algorithms)):
        if not isinstance(stats, list):  # Verificar si stats es una lista de diccionarios
            print(f"Error: Se esperaba una lista de diccionarios, pero se recibió {type(stats)}")
            continue
        
        fig, ax = plt.subplots(figsize=(10, 6))  # Crear un nuevo gráfico para cada algoritmo
        
        # Extraer información relevante de cada brazo
        arm_indices = [entry["arm"] for entry in stats]
        avg_rewards = [entry["average_reward"] for entry in stats]
        selection_counts = [entry["selection_count"] for entry in stats]
        is_optimal = [entry["is_optimal"] for entry in stats]

        # Etiquetas para el eje X con porcentaje de selección y marcando el óptimo
        x_labels = [f"Brazo {idx}\n{round((count/1000)*100, 2)}% - {'Óptimo' if opt else 'No'}" 
                    for idx, count, opt in zip(arm_indices, selection_counts, is_optimal)]

        # Colores agradables: verde para el brazo óptimo, tonos de azul para los demás
        color_palette = ["#2ECC71" if opt else "#3498DB" for opt in is_optimal]

        # Generar el gráfico de barras
        bars = ax.bar(x_labels, avg_rewards, color=color_palette, edgecolor="black", alpha=0.8)

        # Obtener etiqueta mejorada del algoritmo
        algorithm_label = get_algorithm_label(algorithm)

        # Configuración de los ejes y el título
        ax.set_xlabel("Frecuencia de Selección del Brazo", fontsize=12, fontweight='bold')
        ax.set_ylabel("Recompensa Media", fontsize=12, fontweight='bold')
        ax.set_title(f"Análisis de Selección de Brazos - {algorithm_label}", fontsize=14, fontweight='bold')
        ax.set_xticklabels(x_labels, rotation=45, ha="right", fontsize=10)
        ax.grid(axis='y', linestyle='--', alpha=0.7)
        
        # Agregar leyenda
        legend_labels = {"Óptimo": "#2ECC71", "No Óptimo": "#3498DB"}
        legend_patches = [plt.Line2D([0], [0], color=color, lw=6, label=f"{label}") for label, color in legend_labels.items()]
        ax.legend(handles=legend_patches, fontsize=10, title_fontsize=11, loc="upper right")
        
        # Ajustar diseño y mostrar
        plt.tight_layout()
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
        plt.plot(range(steps), regret_accumulated[i], label=get_algorithm_label(algorithm))
    
    if theoretical_bound is not None:
        plt.plot(range(steps), theoretical_bound, 'k--', label="Cota Teórica C * ln(T)")
    
    plt.xlabel("Pasos de tiempo")
    plt.ylabel("Regret Acumulado")
    plt.title("Evolución del Regret Acumulado")
    plt.legend()
    plt.grid(True)
    plt.show()

