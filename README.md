# TFM: Diseño e Implementación de un Sistema Basado en Aprendizaje por Refuerzo para Jugar al Ajedrez

## Descripción

El proyecto tiene como objetivo estudiar y desarrollar sistemas que utilizan el aprendizaje por refuerzo para abordar el juego del ajedrez.

## Instalación

El proyecto se ha desarrollado en Ubuntu 22.04.4. Para la instalación y configuración seguir los siguientes pasos:

1. **Clonar el repositorio:**
   ```bash
   git clone https://github.com/JavierDongWu/TFM_JavierDongWu_RL_Chess.git
   cd TFM_JavierDongWu_RL_Chess

2. **Requisitos:**

   Instalar las librerías necesarias utilizando pip: pettingzoo[classic]>=1.24.0, stable-baselines3>=2.0.0, sb3-contrib>=2.0.0
   ```bash
   pip install pettingzoo[classic]
   pip install stable-baselines3
   pip install sb3-contrib
   pip install chess

3. **Instalar Stockfish y su librería:**

   No es necesario si no se va a utilizar en los programas.
   ```bash
   sudo apt install stockfish
   pip install stockfish

## Uso

1. **PettingZoo_rewards:**

   Sistema basado en la implementación de PettingZoo añadiendo recompensas parciales.

   Primero, configurar el fichero global_variables.py con los parámetros de entrenamiento deseados, después ejecutar el programa de entrenamiento:

   ```bash
   python3 ppo_chess.py

2. **PettingZoo_only_one_agent:**

   Sistema basado en la implementación de PettingZoo utilizando información de un solo agente.

   Igualmente, configurar el fichero global_variables.py con los parámetros de entrenamiento deseados, después ejecutar el programa de entrenamiento:
   
   ```bash
   python3 black_agent_chess.py

3. **RLC_and_play_chess_program:**

   Sistema basado en el repositorio de Reinforcement Learning Chess de Arjan Groen. Incluye también el programa para integrar y comparar jugadores.

   De la misma manera, en el fichero global_variables.py se pueden configurar tanto los parámetros del programa de entrenamiento como los del programa para comparar jugadores.

   Programa de entrenamiento:   
   ```bash
   python3 neural_training_models.py
   ```

   Programa de comparación de jugadores:  
   ```bash
   python3 play_chess.py
   ```

   Nota: El modelo PPO utilizado por defecto en este último programa es el que tiene como nombre: ppo_model.zip. Para utilizar otro modelo, cambiar el nombre de ese modelo a ppo_model.zip, sustituyendo al antiguo.
