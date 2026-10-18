#Videos aufnehmen von Claude
"""
videos_aufnehmen.py  -  Nimmt Videos von trainierten Stable-Baselines3-Modellen auf.

Funktioniert mit jedem Modell in Models/ und jedem Seed:
  * erkennt den Algorithmus (PPO, A2C, SAC, TD3, DDPG) automatisch aus der .zip
  * sucht die passende VecNormalize-Datei in Environements/ automatisch
  * passt die Ant-Beobachtung (27 / 29 / 105 / 107 Werte) automatisch an das Modell an
  * schreibt pro Seed ein MP4 mit Einblendung (Modell, Seed, Schritt, Belohnung, Distanz)
  * schreibt eine Tabelle ergebnisse.csv mit Return, Laenge und Distanz pro Video

Ordnerstruktur (wie in deinem Projekt):
  main.py
  videos_aufnehmen.py        <- diese Datei
  Models/        *.zip
  Environements/ *_vecnormalize.pkl
  videos/        <- wird erzeugt

Beispiele (Terminal):
  python videos_aufnehmen.py                                   # alle Modelle, Seed 0
  python videos_aufnehmen.py --model lauf_C_14envs --seeds 0-4
  python videos_aufnehmen.py --model all --seeds 0,7,42 --steps 1000
  python videos_aufnehmen.py --model Models/ameise_ppo_target_kl1.zip --stochastic
  python videos_aufnehmen.py --list                            # zeigt Modelle + gefundene pkl

Im Notebook:
  from videos_aufnehmen import aufnehmen
  aufnehmen("lauf_C_14envs", seeds=range(5))

Benoetigt: pip install "gymnasium[mujoco]" stable-baselines3 imageio imageio-ffmpeg pillow
"""

from __future__ import annotations

import argparse
import csv
import json
import zipfile
from pathlib import Path

import numpy as np

# --------------------------------------------------------------------------- Pfade
try:
    BASIS = Path(__file__).resolve().parent
except NameError:  # im Notebook gibt es kein __file__
    BASIS = Path.cwd()

MODELL_ORDNER = BASIS / "Models"
VECNORM_ORDNER = [BASIS / "Environements", BASIS / "Environments", BASIS / "Models", BASIS]
VIDEO_ORDNER = BASIS / "videos"


# --------------------------------------------------------------------------- Hilfsfunktionen
def seeds_lesen(text) -> list[int]:
    """'3' -> [3] · '0-4' -> [0,1,2,3,4] · '0,7,42' -> [0,7,42] · '0-2,10' -> [0,1,2,10]"""
    if isinstance(text, int):
        return [text]
    if not isinstance(text, str):
        return [int(s) for s in text]
    seeds: list[int] = []
    for teil in text.replace(" ", ",").split(","):
        if not teil:
            continue
        if "-" in teil and not teil.startswith("-"):
            a, b = teil.split("-")
            seeds += list(range(int(a), int(b) + 1))
        else:
            seeds.append(int(teil))
    return seeds


def modelle_finden(auswahl: str) -> list[Path]:
    """'all' -> alle .zip in Models/ · Name ohne .zip · oder ein Pfad."""
    if auswahl in ("all", "alle", "*"):
        gefunden = sorted(MODELL_ORDNER.glob("*.zip"))
        if not gefunden:
            raise FileNotFoundError(f"Keine .zip-Modelle in {MODELL_ORDNER}")
        return gefunden
    kandidaten = [Path(auswahl), BASIS / auswahl, MODELL_ORDNER / auswahl,
                  MODELL_ORDNER / f"{auswahl}.zip", Path(f"{auswahl}.zip")]
    for k in kandidaten:
        if k.is_file():
            return [k.resolve()]
    teilweise = sorted(p for p in MODELL_ORDNER.glob("*.zip") if auswahl.lower() in p.stem.lower())
    if teilweise:
        return teilweise
    raise FileNotFoundError(f"Modell '{auswahl}' nicht gefunden. Vorhanden: "
                            + ", ".join(p.stem for p in MODELL_ORDNER.glob("*.zip")))


def modell_daten(zip_pfad: Path) -> dict:
    """Liest die Metadaten ('data') aus der SB3-zip, ohne das Modell zu laden."""
    with zipfile.ZipFile(zip_pfad) as z:
        return json.loads(z.read("data").decode("utf-8"))


def algo_erkennen(zip_pfad: Path) -> str:
    """Erkennt den Algorithmus an typischen Hyperparametern in der zip."""
    d = modell_daten(zip_pfad)
    policy = json.dumps(d.get("policy_class", "")).lower()
    if "buffer_size" in d:                     # Off-Policy
        if "target_entropy" in d or "sac" in policy:
            return "SAC"
        if "policy_delay" in d and d.get("policy_delay", 1) != 1:
            return "TD3"
        if "td3" in policy:
            return "TD3"
        if "exploration_rate" in d or "dqn" in policy:
            return "DQN"
        return "DDPG"
    if "clip_range" in d or "n_epochs" in d:
        return "PPO"
    return "A2C"


def obs_groesse(zip_pfad: Path) -> int | None:
    """Liest die Grösse des Beobachtungsraums, mit dem das Modell trainiert wurde."""
    d = modell_daten(zip_pfad)
    raum = d.get("observation_space", {})
    form = raum.get("_shape") or raum.get("shape")
    if form:
        return int(np.prod(form))
    return None


def vecnorm_finden(zip_pfad: Path) -> Path | None:
    """Sucht <modellname>_vecnormalize.pkl, <modellname>.pkl oder eine pkl, die mit dem Namen beginnt."""
    name = zip_pfad.stem
    for ordner in VECNORM_ORDNER:
        if not ordner.is_dir():
            continue
        for kandidat in (f"{name}_vecnormalize.pkl", f"{name}_vec_normalize.pkl",
                         f"{name}_vecnorm.pkl", f"{name}.pkl", f"vecnormalize_{name}.pkl"):
            if (ordner / kandidat).is_file():
                return ordner / kandidat
        # nur "<name>_vec..." zulassen, sonst passt "x_steps" auch auf "x_steps_linear_decay_..."
        passend = sorted(p for p in ordner.glob("*.pkl")
                         if p.stem.lower().startswith(name.lower() + "_vec"))
        if passend:
            return passend[0]
    return None


def ant_kwargs_fuer(n_obs: int | None, env_id: str) -> dict:
    """Ant-v4/v5: wählt die Beobachtungs-Optionen so, dass die Grösse zum Modell passt."""
    if n_obs is None or not env_id.startswith("Ant"):
        return {}
    tabelle = {
        105: dict(),  # Standard v5
        107: dict(exclude_current_positions_from_observation=False),
        27: dict(include_cfrc_ext_in_observation=False),
        29: dict(include_cfrc_ext_in_observation=False, exclude_current_positions_from_observation=False),
    }
    if env_id.endswith("v4"):  # v4: Kontaktkräfte standardmässig aus
        tabelle = {27: dict(), 29: dict(exclude_current_positions_from_observation=False),
                   105: dict(use_contact_forces=True),
                   107: dict(use_contact_forces=True, exclude_current_positions_from_observation=False)}
    return tabelle.get(n_obs, {})


# --------------------------------------------------------------------------- Einblendung
def einblenden(bild: np.ndarray, zeilen: list[str]) -> np.ndarray:
    try:
        from PIL import Image, ImageDraw, ImageFont
    except ImportError:
        return bild
    img = Image.fromarray(bild)
    zeichnen = ImageDraw.Draw(img, "RGBA")
    try:
        schrift = ImageFont.truetype("arial.ttf", max(14, img.height // 30))
    except OSError:
        schrift = ImageFont.load_default()
    hoehe = schrift.getbbox("Ag")[3] + 6
    breite = max(zeichnen.textlength(z, font=schrift) for z in zeilen) + 20
    zeichnen.rectangle([8, 8, 8 + breite, 8 + hoehe * len(zeilen) + 8], fill=(0, 0, 0, 140))
    for i, z in enumerate(zeilen):
        zeichnen.text((18, 14 + i * hoehe), z, fill=(255, 255, 255), font=schrift)
    return np.asarray(img)


# --------------------------------------------------------------------------- Kern
def aufnehmen(model="all", seeds="0", env_id="Ant-v5", steps=1000, episoden=1,
              deterministic=True, vecnorm=None, algo=None, env_kwargs=None,
              breite=1280, hoehe=720, fps=None, overlay=True, ausgabe=None):
    """Nimmt für jedes Modell × jeden Seed ein Video auf. Gibt die Ergebnisliste zurück."""
    import gymnasium as gym
    import imageio.v2 as imageio
    import stable_baselines3 as sb3
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

    ausgabe = Path(ausgabe) if ausgabe else VIDEO_ORDNER
    seeds = seeds_lesen(seeds)
    ergebnisse = []

    for zip_pfad in modelle_finden(str(model)) if not isinstance(model, Path) else [model]:
        name = zip_pfad.stem
        algo_name = (algo or algo_erkennen(zip_pfad)).upper()
        AlgoKlasse = getattr(sb3, algo_name)
        modell = AlgoKlasse.load(zip_pfad, device="cpu")
        n_obs = int(np.prod(modell.observation_space.shape))
        kwargs = {**ant_kwargs_fuer(n_obs, env_id), **(env_kwargs or {})}
        pkl = Path(vecnorm) if vecnorm else vecnorm_finden(zip_pfad)

        print(f"\n=== {name} ===")
        print(f"  Algorithmus : {algo_name}")
        print(f"  Beobachtung : {n_obs} Werte  ->  env_kwargs {kwargs or '{}'}")
        print(f"  VecNormalize: {pkl.name if pkl else 'keine gefunden (Modell ohne Normalisierung?)'}")

        def neue_env():
            return gym.make(env_id, render_mode="rgb_array", width=breite, height=hoehe, **kwargs)

        env = neue_env()
        if env.observation_space.shape != modell.observation_space.shape:
            raise ValueError(f"Beobachtung passt nicht: Env {env.observation_space.shape}, "
                             f"Modell {modell.observation_space.shape}. Setze --env-kwargs passend.")

        norm = None
        if pkl:
            norm = VecNormalize.load(str(pkl), DummyVecEnv([neue_env]))
            norm.training = False      # Statistik einfrieren
            norm.norm_reward = False   # echte Belohnung anzeigen

        video_fps = fps or int(round(env.metadata.get("render_fps", 20)))
        ordner = ausgabe / name
        ordner.mkdir(parents=True, exist_ok=True)

        for seed in seeds:
            modell.set_random_seed(seed)
            modus = "det" if deterministic else "stoch"
            datei = ordner / f"{name}_seed{seed}_{modus}.mp4"
            schreiber = imageio.get_writer(datei, fps=video_fps, codec="libx264", quality=8,
                                           macro_block_size=1)
            for ep in range(episoden):
                obs, info = env.reset(seed=seed + ep)
                x0 = info.get("x_position", 0.0)
                summe, schritt = 0.0, 0
                for schritt in range(1, steps + 1):
                    eingabe = norm.normalize_obs(obs) if norm else obs
                    aktion, _ = modell.predict(eingabe, deterministic=deterministic)
                    obs, r, terminated, truncated, info = env.step(aktion)
                    summe += float(r)
                    bild = env.render()
                    if overlay:
                        dx = info.get("x_position", x0) - x0
                        bild = einblenden(bild, [
                            f"{name}  ({algo_name}, {modus})",
                            f"Seed {seed}   Episode {ep + 1}/{episoden}",
                            f"Schritt {schritt:4d}   t = {schritt * env.unwrapped.dt:5.1f} s",
                            f"Return {summe:8.1f}   Distanz {dx:6.2f} m",
                        ])
                    schreiber.append_data(bild)
                    if terminated or truncated:
                        break
                dx = info.get("x_position", x0) - x0
                ergebnisse.append(dict(modell=name, algo=algo_name, seed=seed, episode=ep + 1,
                                       modus=modus, schritte=schritt, return_=round(summe, 2),
                                       distanz_m=round(dx, 3), gefallen=bool(terminated),
                                       video=str(datei)))
                print(f"  Seed {seed:>4} Ep {ep + 1}: Return {summe:8.1f} | "
                      f"{schritt:4d} Schritte | Distanz {dx:6.2f} m{'  (umgefallen)' if terminated else ''}")
            schreiber.close()
        env.close()
        if norm:
            norm.close()

    if ergebnisse:
        ausgabe.mkdir(parents=True, exist_ok=True)
        csv_pfad = ausgabe / "ergebnisse.csv"
        neu = not csv_pfad.exists()
        with open(csv_pfad, "a", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(ergebnisse[0].keys()), delimiter=";")
            if neu:
                w.writeheader()
            w.writerows(ergebnisse)
        print(f"\nFertig: {len(ergebnisse)} Episoden. Videos in {ausgabe}, Tabelle in {csv_pfad.name}")
    return ergebnisse


def liste_zeigen():
    print(f"Modelle in {MODELL_ORDNER}:")
    for z in sorted(MODELL_ORDNER.glob("*.zip")):
        try:
            info = f"{algo_erkennen(z):4s} | obs {obs_groesse(z)}"
        except Exception as e:  # beschädigte zip o.ä.
            info = f"nicht lesbar ({e})"
        pkl = vecnorm_finden(z)
        print(f"  {z.stem:35s} {info} | VecNormalize: {pkl.name if pkl else '-'}")


# --------------------------------------------------------------------------- Terminal
def main(argv=None):
    p = argparse.ArgumentParser(description="Videos von SB3-Modellen aufnehmen")
    p.add_argument("--model", default="all", help="'all', Modellname oder Pfad zur .zip")
    p.add_argument("--seeds", default="0", help="z.B. 0  |  0-4  |  0,7,42")
    p.add_argument("--env", default="Ant-v5", help="Gymnasium-Umgebung")
    p.add_argument("--steps", type=int, default=1000, help="max. Schritte pro Episode")
    p.add_argument("--episodes", type=int, default=1, help="Episoden pro Seed (im selben Video)")
    p.add_argument("--stochastic", action="store_true", help="mit Zufall (Training-Verhalten)")
    p.add_argument("--vecnorm", default=None, help="Pfad zur VecNormalize-pkl (sonst automatisch)")
    p.add_argument("--algo", default=None, help="PPO/SAC/TD3/A2C/DDPG erzwingen")
    p.add_argument("--env-kwargs", default=None, help='JSON, z.B. \'{"ctrl_cost_weight": 0.1}\'')
    p.add_argument("--width", type=int, default=1280)
    p.add_argument("--height", type=int, default=720)
    p.add_argument("--fps", type=int, default=None)
    p.add_argument("--no-overlay", action="store_true", help="ohne Texteinblendung")
    p.add_argument("--out", default=None, help="Ausgabeordner (Standard: videos/)")
    p.add_argument("--list", action="store_true", help="nur Modelle und pkl-Dateien anzeigen")
    a = p.parse_args(argv)

    if a.list:
        liste_zeigen()
        return
    aufnehmen(model=a.model, seeds=a.seeds, env_id=a.env, steps=a.steps, episoden=a.episodes,
              deterministic=not a.stochastic, vecnorm=a.vecnorm, algo=a.algo,
              env_kwargs=json.loads(a.env_kwargs) if a.env_kwargs else None,
              breite=a.width, hoehe=a.height, fps=a.fps, overlay=not a.no_overlay, ausgabe=a.out)


if __name__ == "__main__":
    main()
