""" Von Claude
Wrapper, die Ant-v5 realistischer bzw. schwieriger machen.
Jeder Wrapper simuliert ein Problem, das beim echten SpotMicro auftreten wird.

Übersicht:
  HeavyBody       fest schwererer Körper          -> Akku, Pi und Elektronik wiegen mehr als geplant
  WeakMotors      fest schwächere Motoren         -> Servos haben begrenztes Drehmoment
  RandomPhysics   Masse/Reibung/Motor zufällig    -> Modell stimmt nie genau (Domain Randomization)
  ServoSpeedLimit begrenzte Bewegungsgeschwindigkeit -> Servos drehen nicht beliebig schnell
  ActionDelay     verzögerte Befehle              -> Servos und Pi reagieren mit Verzögerung
  ObsNoise        Rauschen auf Messwerten         -> IMU und Servo-Positionen sind ungenau
  RandomPush      zufällige Stösse                -> Unebenheiten, Anstossen

Benutzung im Training:
  from ant_wrappers import make_robot_ant
  env = make_vec_env(make_robot_ant, n_envs=14, vec_env_cls=SubprocVecEnv)

Einzelne Herausforderungen ein- oder ausschalten:
  env = make_vec_env(make_robot_ant, n_envs=14, vec_env_cls=SubprocVecEnv,
                     env_kwargs=dict(push=False, delay=0))
"""
from collections import deque
import numpy as np
import gymnasium as gym


# ---------------------------------------------------------------------------
# Feste Änderungen (gleich in jeder Episode)
# ---------------------------------------------------------------------------

class HeavyBody(gym.Wrapper):
    """Macht den ganzen Roboter um einen festen Faktor schwerer.
    factor=1.5 -> 50 % schwerer. Trägheit wird mitskaliert, damit die Physik stimmt."""

    def __init__(self, env, factor=1.5):
        super().__init__(env)
        m = self.env.unwrapped.model
        m.body_mass[:] *= factor
        m.body_inertia[:] *= factor


class WeakMotors(gym.Wrapper):
    """Macht alle Motoren um einen festen Faktor schwächer.
    factor=0.8 -> Motoren haben nur noch 80 % der Kraft."""

    def __init__(self, env, factor=0.8):
        super().__init__(env)
        self.env.unwrapped.model.actuator_gear[:, 0] *= factor


# ---------------------------------------------------------------------------
# Zufällige Änderungen (anders in jeder Episode)
# ---------------------------------------------------------------------------

class RandomPhysics(gym.Wrapper):
    """Domain Randomization: Masse, Reibung und Motorstärke bei jedem Reset zufällig ändern.
    mass=0.2 heisst ±20 %. Baut auf den Werten auf, die beim Erstellen gelten
    (also auch auf HeavyBody/WeakMotors, falls die vorher angewendet wurden)."""

    def __init__(self, env, mass=0.2, friction=0.3, motor=0.2):
        super().__init__(env)
        m = self.env.unwrapped.model
        self.mass0 = m.body_mass.copy()
        self.inertia0 = m.body_inertia.copy()
        self.friction0 = m.geom_friction.copy()
        self.gear0 = m.actuator_gear.copy()
        self.mass, self.friction, self.motor = mass, friction, motor

    def reset(self, **kwargs):
        result = self.env.reset(**kwargs)  # zuerst reset, damit der Seed auch die Zufallswerte festlegt
        m = self.env.unwrapped.model
        rng = self.np_random
        scale = rng.uniform(1 - self.mass, 1 + self.mass, size=self.mass0.shape)
        m.body_mass[:] = self.mass0 * scale
        m.body_inertia[:] = self.inertia0 * scale[:, None]
        m.geom_friction[:, 0] = self.friction0[:, 0] * rng.uniform(1 - self.friction, 1 + self.friction)
        m.actuator_gear[:, 0] = self.gear0[:, 0] * rng.uniform(
            1 - self.motor, 1 + self.motor, size=self.gear0.shape[0])
        return result


# ---------------------------------------------------------------------------
# Servos
# ---------------------------------------------------------------------------

class ServoSpeedLimit(gym.ActionWrapper):
    """Begrenzt, wie stark sich ein Befehl pro Schritt ändern darf.
    Echte Servos brauchen Zeit, um eine neue Position zu erreichen (DS3218: ca. 0.15 s für 60°).
    max_change=0.5 -> der Befehl darf sich pro Schritt (0.05 s) um höchstens 0.5 ändern
    (Aktionsbereich bei Ant: -1 bis 1)."""

    def __init__(self, env, max_change=0.5):
        super().__init__(env)
        self.max_change = max_change
        self.last = np.zeros(self.action_space.shape)

    def reset(self, **kwargs):
        self.last = np.zeros(self.action_space.shape)
        return self.env.reset(**kwargs)

    def action(self, action):
        action = np.clip(action, self.last - self.max_change, self.last + self.max_change)
        self.last = action
        return action


class ActionDelay(gym.Wrapper):
    """Verzögerung: Befehle kommen erst nach einigen Schritten an.
    Bei Ant-v5 ist 1 Schritt = 0.05 s, also delay=1 entspricht 50 ms."""

    def __init__(self, env, delay=1):
        super().__init__(env)
        self.delay = delay

    def reset(self, **kwargs):
        zero = np.zeros(self.action_space.shape)
        self.buffer = deque([zero] * self.delay, maxlen=self.delay + 1)
        return self.env.reset(**kwargs)

    def step(self, action):
        self.buffer.append(action)
        return self.env.step(self.buffer.popleft())


# ---------------------------------------------------------------------------
# Sensoren und Umgebung
# ---------------------------------------------------------------------------

class ObsNoise(gym.ObservationWrapper):
    """Sensorrauschen: Echte IMU und Servo-Positionen sind nie exakt.
    std=0.02 -> normalverteiltes Rauschen mit Standardabweichung 0.02 auf jedem Messwert."""

    def __init__(self, env, std=0.02):
        super().__init__(env)
        self.std = std

    def observation(self, obs):
        return obs + self.np_random.normal(0, self.std, size=obs.shape)


class RandomPush(gym.Wrapper):
    """Stösse: Ab und zu wird der Körper seitlich angestossen.
    force in Newton. Der Ant ist leicht (unter 1 kg), also klein anfangen und langsam erhöhen.
    prob=0.01 -> im Schnitt ein Stoss alle 100 Schritte (5 s)."""

    def __init__(self, env, force=5.0, prob=0.01, duration=5):
        super().__init__(env)
        self.force, self.prob, self.duration = force, prob, duration
        self.left = 0

    def reset(self, **kwargs):
        self.left = 0
        self.env.unwrapped.data.xfrc_applied[:] = 0
        return self.env.reset(**kwargs)

    def step(self, action):
        data = self.env.unwrapped.data
        if self.left == 0 and self.np_random.random() < self.prob:
            angle = self.np_random.uniform(0, 2 * np.pi)
            data.xfrc_applied[1, :2] = self.force * np.array([np.cos(angle), np.sin(angle)])  # Körper = 1
            self.left = self.duration
        elif self.left > 0:
            self.left -= 1
            if self.left == 0:
                data.xfrc_applied[1, :] = 0
        return self.env.step(action)


# ---------------------------------------------------------------------------
# Alles zusammen
# ---------------------------------------------------------------------------

def make_robot_ant(heavy=1.0, weak=1.0, random_physics=True, speed_limit=0.5,
                   delay=1, noise=0.02, push=True, **kwargs):
    """Ant mit den Herausforderungen des echten Roboters.
    Ausschalten: heavy=1.0, weak=1.0, random_physics=False, speed_limit=None,
                 delay=0, noise=0, push=False
    Weitere kwargs gehen an gym.make (z.B. render_mode="rgb_array")."""
    env = gym.make("Ant-v5", **kwargs)

    # Reihenfolge ist wichtig: zuerst feste Änderungen, dann Zufall darauf
    if heavy != 1.0:
        env = HeavyBody(env, factor=heavy)
    if weak != 1.0:
        env = WeakMotors(env, factor=weak)
    if random_physics:
        env = RandomPhysics(env)

    # Servos: zuerst Geschwindigkeitsgrenze, dann Verzögerung
    if speed_limit is not None:
        env = ServoSpeedLimit(env, max_change=speed_limit)
    if delay > 0:
        env = ActionDelay(env, delay=delay)

    # Sensoren und Umgebung
    if noise > 0:
        env = ObsNoise(env, std=noise)
    if push:
        env = RandomPush(env)
    return env


def make_plain_ant(**kwargs):
    """Normaler Ant ohne Herausforderungen (zum Vergleich)."""
    return make_robot_ant(random_physics=False, speed_limit=None, delay=0,
                          noise=0, push=False, **kwargs)
