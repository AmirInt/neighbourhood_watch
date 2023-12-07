import yaml

def load_config() -> dict:
    with open("conf.yaml") as conf_file:
        try:
            config = yaml.safe_load(conf_file)
        except yaml.YAMLError as exc:
            print(exc)
            exit()

    return config
