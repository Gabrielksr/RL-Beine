import os
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize
import webbrowser
from tensorboard import program
import random
from datetime import datetime
import math


MODELS = r"C:\Users\gfock\Documents\VSC\RL\Colab\Models"
ENVS   = r"C:\Users\gfock\Documents\VSC\RL\Colab\Environements"
LOGS   = r"C:\Users\gfock\Documents\VSC\RL\Colab\logs"
os.makedirs(MODELS, exist_ok=True)
os.makedirs(ENVS, exist_ok=True)
os.makedirs(LOGS, exist_ok=True)

#linearer decay
def linear_schedule(start_lr, end_lr=1e-5):
    def lr(progress_remaining):
        return end_lr + progress_remaining * (start_lr - end_lr)
    return lr

#cos decay
def cosine_schedule(start_lr, end_lr=1e-5):
    def lr(progress_remaining):
        progress = 1 - progress_remaining
        return end_lr + 0.5 * (start_lr - end_lr) * (1 + math.cos(math.pi * progress))
    return lr


if __name__ == "__main__":
    torch.set_num_threads(1)
    N_ENVS = 14
    # fortlaufende Nummer vorne, damit neuere Läufe im Ordner immer weiter unten stehen von Claude
    nummern = [int(f.split("_")[0]) for f in os.listdir(MODELS) if f.split("_")[0].isdigit()]
    NAME = f"{max(nummern, default=0) + 1:02d}_" + (input("Speichername: ").strip() or f"lauf_C_{N_ENVS}envs")
    print("Speichere als", NAME)
    # Datum+Uhrzeit vorne, damit die Logs chronologisch von Claude
    LOG_NAME = f"{datetime.now():%Y-%m-%d_%H-%M}_{NAME}"
    # seed= random.randint(0,3)


    env = make_vec_env("Ant-v5", n_envs=N_ENVS, vec_env_cls=SubprocVecEnv)
    env = VecNormalize(env, norm_obs=True, norm_reward=True)

    model = PPO("MlpPolicy",env,
                learning_rate=3e-4,
                policy_kwargs=dict(net_arch=dict(pi=[256,256], vf=[256, 256])),
                verbose=1,
                device="cpu", 
                n_steps=512,
                batch_size= 64,
                target_kl=0.03,
                tensorboard_log=LOGS)

    #Tensorboard
    tb = program.TensorBoard()
    tb.configure(argv=[None, "--logdir", LOGS])
    url = tb.launch()                        
    print("TensorBoard running at", url)
    webbrowser.open(url)          

    model.learn(total_timesteps=10_000_000, tb_log_name=LOG_NAME, log_interval=1)


    model.save(os.path.join(MODELS, NAME))
    env.save(os.path.join(ENVS, NAME + "_vecnormalize.pkl"))
    print("Gespeichert in", MODELS, "und", ENVS)