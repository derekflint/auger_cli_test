import time

from auger_cli.config import AugerConfig
from auger_cli.client import AugerClient

from auger_cli.api import auth
from auger_cli.api import experiments


def run_iris_train():
    # To read login information from experiment dir:
    #config_settings={'login_config_path': "./iris_train"}

    # To use root user dir to read login information
    config_settings={}

    # Read experiment setting from iris_train\auger_experiment.yml
    client = AugerClient(AugerConfig(config_dir="./iris_train",
        config_settings=config_settings))

    # To login to Auger:
    # url is optional parameter, hub_url may be specified in config_settings
    url = "https://app.auger.ai"
    auth.login(client, "username", "password", url)

    # Experiment run, after finish, save experiment session parameters to .auger_experiment_session.yml
    experiments.run(client)

    while True:
        leaderboard, info = experiments.read_leaderboard(client)
        print(info.get("Status"))
        if info.get("Status") == 'error':
            raise Exception("dataset train failed: %s"%info.get("Error"))

        if info.get("Status") != 'completed':
            time.sleep(5)
            continue

        break

    print("leaderborad: {}".format(leaderboard))
    # Create pipeline based on best trial
    # Only downloaded version currently supported
    result = experiments.predict_by_file_locally(client, file='./iris_train/files/iris_predict.csv', trial_id=leaderboard[0]['id'],save_to_file=False,pull_docker=True)
    print(result[0])

run_iris_train()
