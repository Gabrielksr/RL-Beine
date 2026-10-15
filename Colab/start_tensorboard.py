#Tensorboard öffnen von Claude
import time
import webbrowser
from tensorboard import program

LOGS = "./logs"   # Ordner mit deinen Trainings-Logs (tensorboard_log=... beim PPO)


def main():
    tb = program.TensorBoard()
    tb.configure(argv=[None, "--logdir", LOGS, "--reload_interval", "15"])
    url = tb.launch()
    print("TensorBoard läuft auf", url)
    print("Beenden mit Ctrl+C")
    webbrowser.open(url)

    # Skript am Leben halten, sonst schliesst sich TensorBoard sofort wieder
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nTensorBoard beendet.")


if __name__ == "__main__":
    main()
