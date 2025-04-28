import hydra

def return_config(args):

    with hydra.initialize(config_path="./configs-v2"):
        config = hydra.compose(config_name=args.config, overrides=args.overrides)
    print("Running with config:\n{}".format(config))

    return config