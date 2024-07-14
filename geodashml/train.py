from environment import environment
import wandb

def train():
    e = environment()
    try:
        e.start()
    finally:
        e.end()


wandb.agent('bestkatavn8-FFilm/geodashml/5kyomeqj',function=train)    