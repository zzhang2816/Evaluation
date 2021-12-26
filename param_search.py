from src.utils import sweep_train
import yaml
import wandb
import argparse

if __name__ =="__main__":
    parser = argparse.ArgumentParser(description='model and training confg')
    parser.add_argument('--model', type=str, choices=["model_v1","model_v2"])
    args = parser.parse_args()
    with open(f"confgs/{args.model}.yaml", "r") as stream:
        try:
            my_confg=yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)
    wandb.login()
    sweep_id = wandb.sweep(my_confg, project="trial")
    wandb.agent(sweep_id, sweep_train,count=1)
    wandb.finish()
