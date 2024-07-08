from environment import environment
import wandb

def train():
    e = environment()
    try:
        e.start()
    finally:
        e.end()


wandb.agent('hakanonal/geodashml/t9y4f52z',function=train)    