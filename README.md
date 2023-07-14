# Proyecto de la asignatura Minería de Datos

## Descripción general del proyecto

Este proyecto de minería de datos tiene como objetivo crear un agente de aprendizaje por refuerzo para el juego CarRacing-v2. El objetivo principal del agente es aprender a conducir un automóvil en una pista de carreras, maximizando la puntuación obtenida y evitando obstáculos. El proyecto no pudo ser finalizado y el codigo actualmente no cumple con el objetivo propuesto. Esto sera solucionado en versiones posteriores del codigo sin embargo, sirve como ejemplo para visualizar la estructura de esta red neuronal y funciona como modelo explicativo. Este trabajo se baso en el ejemplo del siguiente ejemplo:Capstone Project – Car Racing Using DQN
 https://learning.oreilly.com/library/view/hands-on-reinforcement-learning/9781788836524/4929bee3-df49-40e9-977f-9360293ad8ed.xhtml



## Contexto del proyecto

El aprendizaje por refuerzo es una forma de aprendizaje automático basado en la interacción de un agente con un ambiente para aprender a realizar acciones que maximicen una recompensa acumulada a lo largo del tiempo. En este proyecto, utilizamos algoritmos de aprendizaje por refuerzo para entrenar un agente que aprenda a conducir el automóvil de manera óptima en el juego CarRacing-v2. 

## Metodología

La metodología utilizada en este proyecto consta de los siguientes pasos:

1. Adquisición de datos: el entorno de simulación proporcionado por el juego CarRacing-v2 nos permite recopilar datos del juego, como capturas de pantalla y acciones de los agentes.

2. Modelado: Se utiliza una red neuronal convolucional (CNN) roja para simular la función de valor del agente y predecir el mejor curso de acción basado en los fotogramas del juego. Usando técnicas de duelo, la arquitectura de red neuronal QNetworkDueling se usa para estimar los valores de las relaciones estado-acción.

3. Entrenamiento del agente: El agente es entrenado usando el algoritmo de aprendizaje por refuerzo conocido como DQN (Deep Q-Network). El agente interactúa con el entorno, elige acciones y actualiza su modelo de valor en función de las recompensas recibidas a lo largo del entrenamiento.

4. Evaluación del modelo: El desempeño del agente capacitado se evalúa mediante métricas de desempeño, como el puntaje promedio obtenido en el juego CarRacing-v2.


## Estructura del repositorio

El repositorio está organizado de la siguiente manera:

- main.py: El archivo principal del proyecto que contiene el código para entrenar y evaluar el agente DQN.
- model.py: El archivo que contiene la implementación de la arquitectura de red neuronal QNetworkDueling.
- utils.py: El archivo que contiene funciones de utilidad y clases auxiliares utilizadas en el proyecto.

Todos los archivos necesarios para ejecutar el proyecto estan presentes en el repositorio

## Dependencias

Para ejecutar este proyecto, se requiere la instalación de las siguientes bibliotecas de Python:

- numpy
- random
- gym
- opencv-python
- tensorflow

Para instalar las dependencias, ejecute el siguiente comando:

pip install numpy random gym opencv-python tensorflow



## Ejecución de los notebooks

Si se utilizan notebooks de Jupyter en este proyecto, se deben seguir los siguientes pasos para ejecutarlos:

1. Crear un entorno virtual (opcional): Se recomienda crear un entorno virtual antes de instalar las dependencias y ejecutar los notebooks. Esto ayudará a mantener las dependencias del proyecto separadas de otras instalaciones de Python en su sistema.

2. Instalar las dependencias: Una vez creado el entorno virtual, instale las dependencias necesarias utilizando el comando mencionado anteriormente.

3. Ejecutar los notebooks: Abra los notebooks en el entorno de Jupyter y ejecute las celdas secuencialmente. Asegúrese de seguir las instrucciones proporcionadas en los notebooks para configurar cualquier configuración especial requerida.

## Pasos seguidos y resultados obtenidos

Los pasos utilizados en este proyecto para desarrollar el agente de aprendizaje por refuerzo fueron los siguientes:

1. Se obtuvieron datos del juego del entorno CarRacing-v2, incluidas capturas de pantalla de la pantalla y las acciones del agente.

2. Los datos del juego se preprocesaron utilizando técnicas de cambio de tamaño y conversión a escala de grises para analizarlos y modelarlos posteriormente.

3. Para representar la función de valor del agente y determinar el mejor curso de acción, se implementó una arquitectura CNN llamada QNetworkDueling.

4. El agente fue entrenado utilizando el algoritmo de aprendizaje por refuerzo DQN, interactuando con el entorno CarRacing-v2 y actualizando su modelo de valor en función de las recompensas recibidas.

5. La efectividad del agente no se pudo evaluar debido a los problemas sin resolver.

## Ejemplos reproducibles

Este código muestra cómo crear y utilizar las clases EnvWrapper, CustomLambda, QNetwork, QNetworkDueling, ReplayMemoryFast y DQN para entrenar un agente DQN en el juego CarRacing-v2. El agente utiliza una red neuronal con la arquitectura QNetworkDueling y una memoria de repetición rápida para el aprendizaje. Se muestra un ejemplo de entrenamiento del agente durante 10 episodios, con el total de recompensas por cada episodio. 

Este ejemplo se basa en la estructura y configuración previamente definidas en el proyecto. Asegúrate de tener todas las dependencias y los datos necesarios para ejecutar el código de manera adecuada.

A continuación se muestra un ejemplo de código para entrenar y evaluar el agente DQN en el juego CarRacing-v2:
```python
# Importar las bibliotecas necesarias
import numpy as np
import random
import gym
from gym.spaces import Box
import cv2
import tensorflow as tf
from tensorflow.keras.layers import Layer

# Clase EnvWrapper: envoltorio para el entorno de juego. Esta clase envuelve el juego y nos permite interactuar con él.
class EnvWrapper:

# Clase CustomLambda: capa personalizada de Keras. Esta clase ayuda a realizar cálculos en la red neuronal.
class CustomLambda(Layer):

# Clase QNetwork: clase base para redes Q. Esta clase representa una red neuronal que aprende a tomar decisiones en el juego.
class QNetwork(object):


# Clase QNetworkDueling: red neuronal para el agente DQN. Esta clase es una versión especializada de la red neuronal que se adapta específicamente a los juegos.
class QNetworkDueling(QNetwork):


# Clase ReplayMemoryFast: memoria de repetición rápida. Esta clase es una memoria que guarda recuerdos de lo que ha sucedido en el juego, para que el agente pueda aprender de ellos.
class ReplayMemoryFast:


# Clase DQN: agente de aprendizaje profundo. Esta clase es el agente de aprendizaje en sí, que utiliza todas las otras clases y algoritmos para mejorar su rendimiento en el juego.
class DQN(object):


# Crear una instancia de EnvWrapper
env_wrapper = EnvWrapper("CarRacing-v2", debug=True)

state_size = (84, 84, 4)
action_size = env_wrapper.action_space.shape[0]

# Crear la sesión de TensorFlow
session = tf.compat.v1.InteractiveSession()

# Crear una instancia de DQN
agent = DQN(state_size=state_size,
            action_size=action_size,
            session=session,
            summary_writer=None,
            exploration_period=1000000,
            minibatch_size=32,
            discount_factor=0.99,
            experience_replay_buffer=1000000,
            target_qnet_update_frequency=20000,
            initial_exploration_epsilon=1.0,
            final_exploration_epsilon=0.1,
            reward_clipping=1.0)

# variables
session.run(tf.compat.v1.global_variables_initializer())

# Entrenamiento
for episode in range(1, 11):
    state = env_wrapper.reset()
    total_reward = 0
    done = False

    while not done:
        action = agent.action(state, training=True)
        next_state, reward, done, _ = env_wrapper.step(action)
        agent.store(state, action, reward, next_state, done)
        agent.train()
        state = next_state
        total_reward += reward

    print("Episode:", episode, "Total Reward:", total_reward)

env_wrapper.env.close()
