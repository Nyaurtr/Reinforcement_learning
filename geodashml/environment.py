from selenium import webdriver
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.common.keys import Keys
from PIL import Image
import io
import numpy as np
import time
import wandb
from agent import agent

class environment:
    def __init__(self, config=None):

        if(config is None):
            wandb.init(project="geodashml")
            self.config = wandb.config
        else:
            wandb.init(project="geodashml",config=config)
            self.config = config
        self.agent = agent(
            discount=self.config['discount'],
            exploration_rate=self.config['exploration_rate'],
            decay_factor=self.config['decay_factor'],
            learning_rate=self.config['learning_rate']
            )
        self.reward = 0
        self.max_reward = 0
        self.tot_reward = 0
        self.tot_penalty = 0
        self.tot_valid = 0
        self.tot_duration = 0
        
        self.browser_options = webdriver.ChromeOptions()
        if self.config['hide_browser']:
            self.browser_options.add_argument('headless')
        self.browser_driver = webdriver.Chrome(executable_path="/usr/local/bin/chromedriver", chrome_options=self.browser_options)
        self.browser_wait = WebDriverWait(self.browser_driver,timeout=20)

        self.browser_driver.get('https://scratch.mit.edu/projects/105500895/embed')
        self.browser_wait.until(lambda d: d.find_element_by_css_selector(".green-flag_green-flag_1kiAo"))
        flag_element = self.browser_driver.find_element_by_css_selector(".green-flag_green-flag_1kiAo")
        time.sleep(1)
        flag_element.click()
        time.sleep(5)
        self.browser_html_element = self.browser_driver.find_element_by_tag_name("html")
        self.browser_pause_down = webdriver.ActionChains(self.browser_driver).move_to_element(self.browser_html_element).key_down("p")
        self.browser_pause_up = webdriver.ActionChains(self.browser_driver).move_to_element(self.browser_html_element).key_up("p")
        self.browser_space_down = webdriver.ActionChains(self.browser_driver).move_to_element(self.browser_html_element).key_down(Keys.SPACE)
        self.browser_space_up = webdriver.ActionChains(self.browser_driver).move_to_element(self.browser_html_element).key_up(Keys.SPACE)
        self.game_pause_state = False
        self.browser_monitor_element = None


    def start(self):
        self.pressSpace() #to start the game
        self.browser_wait.until(lambda d: d.find_element_by_css_selector(".monitor_value_3Yexa"))        
        self.pauseGame()
        
        for episode in range(1,self.config['episode']+1):
            self.reward = 0
            past_reward = 0
            old_state = self.readState()
            while True:
                time_start = time.time()
                action_to_play = self.agent.get_next_action(old_state)
                self.playAction(action_to_play)
                new_state = self.readState()
                self.reward = int(self.readScore())
                self.agent.update(old_state,new_state,self.reward)
                self.tot_reward += self.reward
                self.max_reward = max(self.max_reward,self.reward)
                if(self.reward == past_reward):
                    self.tot_penalty += 1
                    break
                else:
                    self.tot_valid += 1
                past_reward = self.reward
                old_state = new_state
                time_end = time.time()
                self.duration = time_end - time_start
                self.tot_duration += self.duration
            
            metrics = {
                'tot_valid' : self.tot_valid,
                'avg_valid' : self.tot_valid/episode,
                'tot_penalty' : self.tot_penalty,
                'avg_penalty' : self.tot_penalty/episode,
                'exploration_rate' : self.agent.exploration_rate,
                'max_reward' : self.max_reward,
                'tot_reward' : self.tot_reward,
                'avg_reward' : self.tot_reward/episode,
                'reward':self.reward,
                'duration' : self.duration,
                'tot_duration' : self.tot_duration,
                'avg_duration' : self.tot_duration/episode,
            }
            wandb.log(metrics,step=episode)

    def readScore(self):
        if(self.browser_monitor_element is None):
            self.browser_monitor_element = self.browser_driver.find_element_by_css_selector('.monitor-list_monitor-list-scaler_143tA .monitor_monitor-container_2J9gl[style="touch-action: none; transform: translate(0px, 0px); top: 6px; left: 5px;"] .monitor_value_3Yexa')
        return self.browser_monitor_element.text

    def readState(self):
        image_png = self.browser_driver.find_element_by_css_selector(".stage_stage_1fD7k canvas").screenshot_as_png
        img = Image.open(io.BytesIO(image_png))
        if(self.config['hide_browser'] == 0):
            img = img.resize((725,544), Image.ANTIALIAS)
        return np.asarray(img)

    def playAction(self,action):
        self.unpauseGame()
        if action:
            self.pressSpace()
        self.pauseGame()

    def pressSpace(self):
        self.browser_space_down.perform()
        self.browser_space_up.perform()

    def pressPause(self):
        self.browser_pause_down.perform()
        self.browser_pause_up.perform()
    
    def pauseGame(self):
        if(self.game_pause_state == False):
            self.pressPause()
            self.game_pause_state = True

    def unpauseGame(self):
        if(self.game_pause_state == True):
            self.pressPause()
            self.game_pause_state = False

    def end(self):
        self.browser_driver.quit()
        self.agent.saveModel()