#!/usr/bin/env python
# coding: utf-8

# In[2]:


import retro
import numpy as np
import cv2
import neat
import pickle

env = retro.make(game='SuperMarioBros-Nes', state='Level1-1')
flattened_img = []

def eval_genomes(genomes, config):
    inx = 30 #SNES width is 240, so 30 is shrunken width
    iny = 28 # SNES height is 224, so 28 is shruken height
    
    for genome_id, genome in genomes:
        img =  env.reset() #initial img
        ac = env.action_space.sample()
        
        neural_net = neat.nn.feed_forward.FeedForwardNetwork.create(genome, config)
    
        max_fitness = 0
        fitness = 0
        counter = 0
        done = False
        
        while not done:
            env.render()
            
            img = cv2.resize(img, (inx,iny))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            img = np.reshape(img, (iny,inx))
            
            flattened_img = np.ndarray.flatten(img)
            action = neural_net.activate(flattened_img)
            for num in range(12):
                if action[num] >= 0.5:
                    action[num] = 1
                
            #print(action)
            img, reward, done, info = env.step(action)
            
            fitness += reward
            
            if fitness > max_fitness:
                max_fitness = fitness
                counter = 0
            else:
                counter += 1
                
            if done or counter == 75:
                done = True
                print(genome_id, fitness)
            else:
                genome.fitness = fitness
            
            #add implementation to switch to old checkpoint if fitness doesnt improve for long time
            #remove legacy code
            #perhaps lower counter even more while increasing population a lot
            #add slight timer to reward system
            #put negative fitness if xpos is too small or if dead
            
config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction, neat.DefaultSpeciesSet, neat.DefaultStagnation, 'config-feedforward.txt')

p = neat.Population(config)

p.add_reporter(neat.StdOutReporter(True))
stats = neat.StatisticsReporter()
p.add_reporter(stats)
p.add_reporter(neat.Checkpointer(10))

winner = p.run(eval_genomes)

with open('winner.pkl', 'wb') as output:
    pickle.dump(winner, output, 1)


# In[1]:


env.close()


# In[ ]:




