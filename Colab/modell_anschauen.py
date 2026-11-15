import os
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
import keyboard
import time

MODELS = r"C:\Users\gfock\Documents\VSC\RL\Colab\Models"
ENVS   = r"C:\Users\gfock\Documents\VSC\RL\Colab\Environements"
FPS=20

for e in sorted(os.listdir(MODELS)):
        print(e)

MODEL = input("Welches Modell möchtest du Anschauen gabagoo: ").strip().removesuffix(".zip")

env = make_vec_env("Ant-v5", n_envs=1, vec_env_cls=DummyVecEnv,
                   env_kwargs={"render_mode": "human"})
env = VecNormalize.load(os.path.join(ENVS, f"{MODEL}_vecnormalize.pkl"), env)

env.training = False 
env.norm_reward = False

model = PPO.load(os.path.join(MODELS,MODEL), env=env)

obs = env.reset()

while not keyboard.is_pressed('esc'):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)
        # CLAUDE###################
        ant = env.envs[0].unwrapped                  # durch alle Wrapper zur Mujoco-Env
        viewer = ant.mujoco_renderer.viewer          # das offene Fenster
        viewer.cam.lookat[:] = ant.data.body("torso").xpos   # Kamera schaut auf den Torso
        viewer.cam.distance = 5.0                    # optional: Abstand
        viewer.cam.elevation = -20.0                 # optional: Blickwinkel
        ############################
        time.sleep(1 / FPS)
env.close()
