import argparse
import optuna


parser = argparse.ArgumentParser()
parser.add_argument("--study_name", help="Name of study.", type=str, default="optuna_chime7")
parser.add_argument("--storage", help="Shared storage (i.e sqlite:///optuna.db).", type=str, default="sqlite:///optuna_ngc.db")

args = parser.parse_args()

study = optuna.load_study(study_name=args.study_name, storage=args.storage)
print(f"Best SA-WER {study.best_value}")
print(f"Best Parameter Set: {study.best_params}")
