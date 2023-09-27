import pygame
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import math
import random
import os
from collections import namedtuple, deque
import matplotlib.pyplot as plt
from IPython import display

class CarApp:
    def __init__(self, draw = 1):
        self.draw = draw
        
        self.x_train = []
        self.y_train = []
        self.angle = 0
        self.speed = 0
        self.x = 1280/2
        self.y = 720/2
        self.screen_height = 720
        
        self.distances = []
        self.track_changed = 0
        self.running = True
        self.score = 0
        self.turn_angle = 0.25
        self.generate_tracks()
        
        
        if self.draw:
            pygame.init()
            self.screen = pygame.display.set_mode((1280, 720))
            self.speed_font = pygame.font.Font(None, 14)
            self.speed_text = self.speed_font.render("Speed: 0Km/h", True, (0, 0, 0))
            self.score_text = self.speed_font.render("Score: 0", True, (0, 0, 0))
            self.clock = pygame.time.Clock()
            self.clock.tick(60)
        self.update_position()
            
    def generate_tracks(self):
        self.tracks = [
            [Track(self, self.y-500, self.x-500, type = 0), Track(self, self.y-500, self.x, type = 0), Track(self, self.y-500, self.x+500, type = 0)],
            [Track(self, self.y, self.x-500, type = 5), Track(self, self.y, self.x, type = 0), Track(self, self.y, self.x+500, type = 0)],
            [Track(self, self.y+500, self.x-500, type = 0), Track(self, self.y+500, self.x, type = 0), Track(self, self.y+500, self.x+500, type = 0)]
        ]
        
        for tracks in self.tracks:
            for track in tracks:
                track.set_type()
        self.current_track = self.tracks[1][1]
        self.set_adjacent()
        
    def draw_car(self):
        car_rect_standard = pygame.Surface((10, 10) , pygame.SRCALPHA)
        car_rect_standard.fill((255, 0, 0))
        car_rect = pygame.transform.rotate(car_rect_standard, self.angle)
        new_rect = car_rect.get_rect(center = car_rect_standard.get_rect(center = (self.x, self.y)).center)
        self.screen.blit(car_rect, new_rect)

    def turn_left(self):
        if(self.angle - self.turn_angle < 360):
            self.angle += self.turn_angle
        else:
            self.angle = 360-self.angle+self.turn_angle
    
    def turn_right(self):
        if(self.angle - self.turn_angle >= 0):
            self.angle += -self.turn_angle
        else:
            self.angle = 360+self.angle-self.turn_angle
    
    def update_position(self, move = -1):
        self.track_changed = 0
        if move == 0:
            self.move_forward()

        if move == 1:
            self.turn_right()

        if move == 2:
            self.turn_left()

        if move == 3:
            self.slow_down()
        
        if move == 4:
            self.slow_down()
            self.turn_left()
        
        if move == 5:
            self.slow_down()
            self.turn_right()
            
        if move == 6:
            self.turn_right()
            self.move_forward()
            
        if move == 7:
            self.turn_left()
            self.move_forward()
            
        if self.draw:
            self.screen.fill((255, 255, 255))
            
        self.check_tracks()
        
        if self.draw:
            self.current_track.draw_track()
            self.draw_car()
            
        self.draw_tracks()
        self.draw_rays()
        
        if self.draw:
            self.speed_text = self.speed_font.render("Speed: {} Km/h".format(int(self.speed * 3.6)), True, (0, 0, 0))
            self.screen.blit(self.speed_text, (10, 10))
        
        self.score += (self.speed*self.speed)/1000
        
        if self.draw:
            self.score_text = self.speed_font.render("Score: {}".format(int(self.score)), True, (0, 0, 0))
            self.screen.blit(self.score_text, (10, 30))
            pygame.display.update()
            
    def draw_tracks(self):
        tracky = self.speed*0.3*math.cos(math.radians(self.angle))
        trackx = self.speed*0.3*math.sin(math.radians(self.angle))
        
        for tracklist in self.tracks:
            for track in tracklist:
                track.track_y += tracky
                track.track_x += trackx
                track.color = (128, 128, 128)
                if track.adjacent:
                    if self.draw:
                        track.draw_track()      
    
    def check_tracks(self):
        
        for tracks in self.tracks:
            for track in tracks:
                track.adjacent = 0
        
        if self.current_track.track_y > self.y+250:
            self.tracks[2] = self.tracks[1]
            self.tracks[1] = self.tracks[0]
            self.tracks[0] = [Track(self, self.current_track.track_y-1000, self.current_track.track_x-500), 
                                Track(self, self.current_track.track_y-1000, self.current_track.track_x), 
                                Track(self, self.current_track.track_y-1000, self.current_track.track_x+500)]
            
            self.current_track = self.tracks[1][1]
            
            self.tracks[0][1].type = random.choice(self.tracks[1][1].possible_tracks[0])
            self.tracks[0][1].set_type()
            self.tracks[0][0].type = random.choice(list(set(self.tracks[0][1].possible_tracks[3]).intersection(self.tracks[1][0].possible_tracks[0])))
            self.tracks[0][0].set_type()
            self.tracks[0][2].type = random.choice(list(set(self.tracks[0][1].possible_tracks[1]).intersection(self.tracks[1][2].possible_tracks[0])))
            self.tracks[0][2].set_type()
            self.track_changed = 1
        elif self.current_track.track_y < self.y-250:
            self.tracks[0] = self.tracks[1]
            self.tracks[1] = self.tracks[2]
            self.tracks[2] = [Track(self, self.current_track.track_y+1000, self.current_track.track_x-500), 
                                Track(self, self.current_track.track_y+1000, self.current_track.track_x), 
                                Track(self, self.current_track.track_y+1000, self.current_track.track_x+500)]
            self.current_track = self.tracks[1][1]
            
            self.tracks[2][1].type = random.choice(self.tracks[1][1].possible_tracks[2])
            self.tracks[2][1].set_type()
            self.tracks[2][0].type = random.choice(list(set(self.tracks[2][1].possible_tracks[3]).intersection(self.tracks[1][0].possible_tracks[2])))
            self.tracks[2][0].set_type()
            self.tracks[2][2].type = random.choice(list(set(self.tracks[2][1].possible_tracks[1]).intersection(self.tracks[1][2].possible_tracks[2])))
            self.tracks[2][2].set_type()
            self.track_changed = 1
        if self.current_track.track_x > self.x+250:
            self.tracks[0][2] = self.tracks[0][1]
            self.tracks[0][1] = self.tracks[0][0]
            self.tracks[1][2] = self.tracks[1][1]
            self.tracks[1][1] = self.tracks[1][0]
            self.tracks[2][2] = self.tracks[2][1]
            self.tracks[2][1] = self.tracks[2][0]  
            
            self.tracks[0][0] = Track(self, self.current_track.track_y-500, self.current_track.track_x-1000)
            self.tracks[1][0] = Track(self, self.current_track.track_y, self.current_track.track_x-1000)
            self.tracks[2][0] = Track(self, self.current_track.track_y+500, self.current_track.track_x-1000)

            self.current_track = self.tracks[1][1]
            self.tracks[1][0].type = random.choice(self.tracks[1][1].possible_tracks[3])
            self.tracks[1][0].set_type()
            self.tracks[0][0].type = random.choice(list(set(self.tracks[1][0].possible_tracks[0]).intersection(self.tracks[0][1].possible_tracks[3])))
            self.tracks[0][0].set_type()
            self.tracks[2][0].type = random.choice(list(set(self.tracks[1][0].possible_tracks[2]).intersection(self.tracks[2][1].possible_tracks[3])))
            self.tracks[2][0].set_type()
            self.track_changed = 1
            
        elif self.current_track.track_x < self.x-250:
            self.tracks[0][0] = self.tracks[0][1]
            self.tracks[0][1] = self.tracks[0][2]
            self.tracks[1][0] = self.tracks[1][1]
            self.tracks[1][1] = self.tracks[1][2]
            self.tracks[2][0] = self.tracks[2][1]
            self.tracks[2][1] = self.tracks[2][2]
            
            self.tracks[0][2] = Track(self, self.current_track.track_y-500, self.current_track.track_x+1000)
            self.tracks[1][2] = Track(self, self.current_track.track_y, self.current_track.track_x+1000)
            self.tracks[2][2] = Track(self, self.current_track.track_y+500, self.current_track.track_x+1000)   

            self.current_track = self.tracks[1][1]
            
            self.tracks[1][2].type = random.choice(self.tracks[1][1].possible_tracks[1])
            self.tracks[1][2].set_type()
            self.tracks[0][2].type = random.choice(list(set(self.tracks[1][2].possible_tracks[0]).intersection(self.tracks[0][1].possible_tracks[1])))
            self.tracks[0][2].set_type()
            self.tracks[2][2].type = random.choice(list(set(self.tracks[1][2].possible_tracks[2]).intersection(self.tracks[2][1].possible_tracks[1])))
            self.tracks[2][2].set_type()
            self.track_changed = 1            
        self.set_adjacent()
        
        self.current_track.color = (0,255,0) 
        
    def draw_rays(self):
        angles = [-90, -45, -5, 0, 5, 45, 90]
        #angles = [0]
        distances = []
        for angle in angles:
            lanedrawn = 0
            if(self.angle + angle < 0):
                angle = 360 + angle
            if(self.angle + angle >= 360):
                angle = -360 + angle
                           
            
            cords = [
                (-250,-250,250,-250),
                (250, -250, 250, 250),
                (250, 250, -250, 250),
                (-250, 250, -250, -250)
            ]
            if(self.angle+angle >= 0 and self.angle+angle < 90):
                walls = [0,3]
                tracks = [self.tracks[0][1], self.tracks[1][0]]
            if(self.angle+angle >= 90):
                walls = [3,2]
                tracks = [self.tracks[2][1], self.tracks[1][0]]
            if(self.angle+angle >= 180):
                walls = [2,1]
                tracks = [self.tracks[2][1], self.tracks[1][2]]
            if(self.angle+angle >= 270):
                walls = [0,1]
                tracks = [self.tracks[1][2], self.tracks[0][1]]
                
            segments = []
                
            for wall in walls:
                x1, y1, x2, y2 = cords[wall]
                if self.current_track.walls[wall]:
                    segments.append([(self.current_track.track_x+x1-self.x, self.current_track.track_y+y1-self.y), (self.current_track.track_x+x2-self.x, self.current_track.track_y+y2-self.y)])
                else:
                    for track in tracks:
                        if track.adjacent:
                            for outerwall in walls:
                                x1, y1, x2, y2 = cords[outerwall]
                                if track.walls[outerwall]:
                                    segments.append([(track.track_x+x1-self.x, track.track_y+y1-self.y), (track.track_x+x2-self.x, track.track_y+y2-self.y)])

            
            angle = angle - 90
            
            if(self.angle + angle < 0):
                angle = 360 + angle
            if(self.angle + angle >= 360):
                angle = -360 + angle
            
            a = -math.tan(math.radians(self.angle+angle))
            
            for segment in segments:
                pointinfo = self.intersect(segment, a, 0)
                
                distance = pointinfo[1]
                
                if (distance and distance < self.speed*0.3 and angle+90 == 0) or (distance and distance < 5):
                    self.running = 0
                point = pointinfo[0]
                if point != None:
                    distances.append(distance)
                    if self.draw:
                        pygame.draw.line(self.screen, (255,0,0),(self.x, self.y),(self.x + point[0], self.y + point[1]))
                    lanedrawn = 1
                    break
                    
            if lanedrawn == 0:
                distances.append(1000)
                point = self.point_on_line(a)
                if self.angle+angle < 90 or self.angle+angle > 270 or self.angle+angle == 90:
                    point = (-point[0], -point[1])
                if self.draw:
                    pygame.draw.line(self.screen, (255,0,0),(self.x, self.y),(self.x + point[0], self.y + point[1]))    
        self.distances = distances       

    def point_on_line(self, a): 
        if a == 0:
            x = 0
            y = 500
        elif a == float('inf'):
            x = 0
            y = 500
        else:
            x = 500 / math.sqrt(1 + a ** 2)
            y = a * x
        
        return x, y

    def intersect(self, segment, a, b):
        x1, y1 = segment[0]
        x2, y2 = segment[1]
        if x1 == x2:  # vertical segment
            if a == 0:  # line is horizontal
                y = 0
                if y1 <= y <= y2 or y2 <= y <= y1:
                    return (x1, y), -x1
                return None, None
            x = x1
            y = a * x + b
        elif y1 == y2:  # horizontal segment
            if a == float('inf'):  # line is vertical
                x = 0
                if x1 <= x <= x2 or x2 <= x <= x1:
                    return (x, y1), y1
                return None, None
            if a == 0:
                return None, None
            y = y1
            x = (y - b) / a
        else:
            return None, None
        if (x1 <= x <= x2 or x2 <= x <= x1) and (y1 <= y <= y2 or y2 <= y <= y1):
            distance = math.sqrt(x ** 2 + y ** 2)
            return (x, y), distance
        else:
            return None, None

    def move_forward(self):
        self.speed += 0.025-(self.speed*0.00008)
        
    def slow_down(self):
        
        if(self.speed >= 0.07):
            self.speed -= 0.07
        else:
            self.speed = 0
    
    def set_adjacent(self):
        adjacents = {
            0: [(0, 1), (2, 1)],
            1: [(1, 0), (1, 2)],
            2: [(2, 1), (1, 0)],
            3: [(1, 2), (2, 1)],
            4: [(0, 1), (1, 2)],
            5: [(0, 1), (1, 0)]
        }
        
        for row, col in adjacents[self.current_track.type]:
            self.tracks[row][col].adjacent = 1

class Track:
    def __init__(self, master, track_y, track_x, type = -1, color=(128, 128, 128)):
        self.master = master
        self.track_y = track_y
        self.color = color
        self.type = type
        self.track_x = track_x
        self.adjacent = 0
        self.possible_tracks = [[-1], [-1], [-1], [-1]]
        self.walls = [False, False, False, False]
        
    def set_type(self):
        if self.type == 0:
            self.possible_tracks = [[0, 2, 3], [0, 3, 4], [0, 4, 5], [0, 2, 5]]
            self.walls = [False, True, False, True]  # up and down 
        elif self.type == 1:
            self.possible_tracks = [[1, 4, 5], [5, 2, 1], [1, 2, 3], [1, 3, 4]]
            self.walls = [True, False, True, False]  # left and right 
        elif self.type == 2:
            self.possible_tracks = [[1, 4, 5], [0, 3, 4], [0, 4, 5], [1, 3, 4]]
            self.walls = [True, True, False, False]  # down and left 
        elif self.type == 3:
            self.possible_tracks = [[1, 4, 5], [5, 2, 1], [0, 4, 5], [0, 2, 5]]
            self.walls = [True, False, False, True]  # down and right 
        elif self.type == 4:
            self.possible_tracks = [[0, 2, 3], [5, 2, 1], [1, 2, 3], [0, 2, 5]]
            self.walls = [False, False, True, True]  # up and right 
        elif self.type == 5:
            self.possible_tracks = [[0, 2, 3], [0, 3, 4], [1, 2, 3], [1, 3, 4]]
            self.walls = [False, True, True, False]  # up and left 
    
    def draw_track(self, type = -1):
        track_rect = pygame.Rect(self.track_x - 250, self.track_y - 250, 500, 500)
        track = pygame.Surface((500, 500) , pygame.SRCALPHA)
        track.fill(self.color)
        if self.type==0:
            pygame.draw.line(track, (128, 0, 0), (498, 0), (498, 500), 2)
            pygame.draw.line(track, (128, 0, 0), (0, 0), (0, 500), 2)
            
            #updown
        elif self.type==1:
            pygame.draw.line(track, (128, 0, 0), (0, 498), (500, 498), 2)
            pygame.draw.line(track, (128, 0, 0), (0, 0), (500, 0), 2)
            #leftright

        elif self.type==2:
            pygame.draw.line(track, (128, 0, 0), (498, 0), (498, 500), 2)
            pygame.draw.line(track, (128, 0, 0), (0, 0), (500, 0), 2)
            #downleft
        elif self.type==3:
            pygame.draw.line(track, (128, 0, 0), (0, 0), (0, 500), 2)
            pygame.draw.line(track, (128, 0, 0), (0, 0), (500, 0), 2)
            #downright
        elif self.type==4:
            pygame.draw.line(track, (128, 0, 0), (0, 0), (0, 500), 2)
            pygame.draw.line(track, (128, 0, 0), (0, 498), (500, 498), 2)
            #upright
        elif self.type==5:
            pygame.draw.line(track, (128, 0, 0), (498, 0), (498, 500), 2)
            pygame.draw.line(track, (128, 0, 0), (0, 498), (500, 498), 2)
            #upleft
        self.master.screen.blit(track, track_rect)

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))

class ReplayMemory(object):

    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        """Save a transition"""
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

class QNetwork(nn.Module):
    def __init__(self, state_size, action_size, hidden_size=128):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(state_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, hidden_size)
        self.fc4 = nn.Linear(hidden_size, action_size)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(self.device)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        return self.fc4(x)

def select_action(state):
    global steps_done
    sample = random.random()
    eps_threshold = EPS_END + (EPS_START - EPS_END) * \
        math.exp(-1. * steps_done / EPS_DECAY)
    steps_done += 1
    if sample > eps_threshold:
        with torch.no_grad():
            return policy_net(state).max(1)[1].view(1, 1)
    else:
        return torch.tensor([[random.randint(0, 3)]], dtype=torch.long)

def optimize_model():
    if len(memory) < BATCH_SIZE:
        return
    transitions = memory.sample(BATCH_SIZE)
    batch = Transition(*zip(*transitions))

    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                          batch.next_state)), dtype=torch.bool)
    non_final_next_states = torch.cat([s for s in batch.next_state
                                                if s is not None])
    state_batch = torch.cat(batch.state)
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)

    state_action_values = policy_net(state_batch).gather(1, action_batch)

    next_state_values = torch.zeros(BATCH_SIZE)
    with torch.no_grad():
        next_state_values[non_final_mask] = target_net(non_final_next_states).max(1)[0]
    expected_state_action_values = (next_state_values * GAMMA) + reward_batch

    criterion = nn.MSELoss()
    loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))
    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_value_(policy_net.parameters(), 100)
    optimizer.step()

def plot_durations(scores):
    plt.figure(1)
    plt.xlabel('Episode')
    plt.ylabel('Duration')
    plt.plot(scores)
    plt.pause(0.1)
    display.display(plt.gcf())
    display.clear_output(wait=True)

BATCH_SIZE = 128
GAMMA = 0.99
EPS_START = 0.9
EPS_END = 0.2
EPS_DECAY = 10000
TAU = 0.01
LR = 1e-4

policy_net = QNetwork(8, 8, 512)

if os.path.exists("model.pth"):
    print("loading")
    policy_net.load_state_dict(torch.load('model.pth'))
    EPS_START = 0.5

target_net = QNetwork(8, 8, 512)
target_net.load_state_dict(policy_net.state_dict())

optimizer = optim.AdamW(policy_net.parameters(), lr=LR)
memory = ReplayMemory(30000)

steps_done = 0

scores = []

plt.ion()

for i_episode in range(10000):
    app = CarApp(0)
    state = app.distances
    state = np.array(state)/1000
    state = np.append(state, app.speed/10)
    state = torch.tensor(state)
    state = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
    ereturn = 0
    steps = 3000
    timer = 0
    for t in range(steps): # also count() for inf steps
        reward = 0
        action = select_action(state)
        
        app.update_position(action.item())
        
        observation = app.distances
        observation = np.array(observation)/1000
        observation = np.append(observation, app.speed/10)
        observation = torch.tensor(observation)
        
        if app.speed > 0:
            reward += 1*app.speed/100

        terminated = not(app.running)
        
        if terminated:
            reward = -5
        
        truncated = 0
        if t == steps-1:
            truncated = 1
        
        if terminated:
            next_state = None
        else:
            next_state = torch.tensor(observation, dtype=torch.float32).unsqueeze(0)
        ereturn += reward
        reward = torch.tensor([reward])
        
        done = terminated or truncated

        memory.push(state, action, next_state, reward)

        state = next_state

        optimize_model()

        target_net_state_dict = target_net.state_dict()
        policy_net_state_dict = policy_net.state_dict()
        for key in policy_net_state_dict:
            target_net_state_dict[key] = policy_net_state_dict[key]*TAU + target_net_state_dict[key]*(1-TAU)
        target_net.load_state_dict(target_net_state_dict)
        
        if done:
            break
    scores.append(ereturn)
    #plot_durations(scores)
    #policy_net.load_state_dict(target_net.state_dict())
    torch.save(target_net.state_dict(), 'model.pth')
    print('\rEpisode: {}\tSteps: {}\tEpisode return: {}\n'.format(i_episode, t+1, ereturn), end="")
    
    

print('Complete')
