import wandb
run = wandb.init()
artifact = run.use_artifact('hakanonal/geodashml/run-f2mo639v-history:v0', type='wandb-history')
artifact_dir = artifact.download()