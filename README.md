# k_brazos_DHBV

## Información
- **Alumnos:**
  - de Haro Orenes, Salvador - `salvadorde.haroo@um.es`
  - Vidal García, Miguel - `miguel.vidalg@um.es`
  - Belmonte Martínez, Jose María - `josemaria.belmontem@um.es`
 
## Descripción
En este repositorio se recopilan una serie de notebooks que conforman el estudio realizado sobre el problema del bandido de *k*-brazos, un modelo fundamental en el campo del aprendizaje por refuerzo. Se han implementado y comparado distintas familias de algoritmos para resolver este problema, evaluando su desempeño en entornos estacionarios mediante simulaciones.
En particular, se ha realizado un análisis detallado de la familia de algoritmos $\varepsilon$-greedy, explorando sus variantes y su impacto en el equilibrio entre exploración y explotación. Asimismo, se han implementado y evaluado otros métodos, como los basados en Upper Confidence Bound (UCB) y técnicas de ascenso del gradiente.
A su vez, en los distintos escenarios, se consideraron distintas distribuciones de probabilidad en los brazos del bandido (Normal, Binomial y Bernoulli), con el fin de analizar como varía la convergencia de los algoritmos en función de la distribución empleada.

## Estructura
- `src/` → Código Python relacionado con los experimentos: Implementación de los algoritmos y sus políticas; Implementación de los brazos con distintas distribuciones de probabilidad; Otras funcionalidades, tales como plotear gráficas.
- `docs/` → Documentación en PDF sobre el trabajo realizado.
- `main.ipynb` → Notebook principal de presentación, que permite navegar entre los distintos notebooks con los experimentos.
- `1_bandit_experiments_epsilongreedy.ipynb` → Experimentos para la familia de algoritmos $\epsilon$-greedy.
- `2_bandit_experiments_ucb.ipynb` → Experimentos para la familia de algoritmos UCB, en particular UCB1 y UCB2.
- `3_bandit_experiments_ascensoGradiente.ipynb` → Experimentos para la familia de algoritmos de Ascenso del Gradiente, en particular Softmax y Gradiente de Preferencias.
- `4_bandit_experiments_torneoFinal.ipynb` → Comparativa final de los mejores algoritmos para las distintas distribuciones de probabilidad en el bandido.

## Instalación y uso
Se recomienda la ejecución de los notebooks en Google Colab. Para este fin, se aconseja abrir el notebook `main.ipynb`, donde se encontrarán enlaces directos para abrir los notebooks con los distintos experimentos en Colab.

## Tecnologías utilizadas y entorno de desarrollo
El proyecto ha sido desarrollado utilizando Python y herramientas del ecosistema de aprendizaje automático y análisis de datos. Se ha utilizado Google Colab como entorno de ejecución para facilitar la ejecución remota de los experimentos. 

### **Lenguaje y herramientas principales**
- **Python** → Lenguaje principal del proyecto.
- **Google Colab** → Entorno de ejecución basado en Jupyter Notebooks para pruebas y experimentos.
- **GitHub** → Control de versiones y almacenamiento del código.
