import pygame
import sys
from pygame.math import Vector2
import random
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# -----------------------------------------------------------
# 1. NEURAL NETWORK MODEL (Required to load the AI)
# -----------------------------------------------------------
class Linear_QNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = F.relu(self.linear1(x))
        x = self.linear2(x)
        return x

# -----------------------------------------------------------
# 2. PYGAME SETUP & GLOBALS
# -----------------------------------------------------------
pygame.mixer.pre_init(44100, -16, 2, 512)
pygame.init()

cell_size = 30
cell_number = 20
# We start with a 1-player screen size. We will resize it for split-screen dynamically.
screen = pygame.display.set_mode((cell_number * cell_size, cell_number * cell_size))
pygame.display.set_caption("Snake: Human vs AI")
clock = pygame.time.Clock()

try:
    game_font = pygame.font.Font('Font/PoetsenOne-Regular.ttf', 25)
    title_font = pygame.font.Font('Font/PoetsenOne-Regular.ttf', 40)
except:
    # Fallback to default font if the user is missing the font file
    game_font = pygame.font.Font(None, 25)
    title_font = pygame.font.Font(None, 40)

try:
    apple = pygame.image.load('Graphics/apple.png').convert_alpha()
except:
    # Fallback if graphic is missing: Create a red square surface
    apple = pygame.Surface((cell_size, cell_size))
    apple.fill((255, 0, 0))

SCREEN_UPDATE = pygame.USEREVENT
pygame.time.set_timer(SCREEN_UPDATE, 150) # Speed of the game

# -----------------------------------------------------------
# 3. GAME CLASSES
# -----------------------------------------------------------
class SNAKE:
    def __init__(self):
        self.body = [Vector2(5,10), Vector2(4,10), Vector2(3,10)]
        self.direction = Vector2(1,0) # Must start with a direction for AI to work
        self.new_block = False

        # Use try-except to safely load graphics, fallback to plain colors if missing
        try:
            self.head_up = pygame.image.load('Graphics/head_up.png').convert_alpha()
            self.head_down = pygame.image.load('Graphics/head_down.png').convert_alpha()
            self.head_right = pygame.image.load('Graphics/head_right.png').convert_alpha()
            self.head_left = pygame.image.load('Graphics/head_left.png').convert_alpha()
            self.tail_up = pygame.image.load('Graphics/tail_up.png').convert_alpha()
            self.tail_down = pygame.image.load('Graphics/tail_down.png').convert_alpha()
            self.tail_right = pygame.image.load('Graphics/tail_right.png').convert_alpha()
            self.tail_left = pygame.image.load('Graphics/tail_left.png').convert_alpha()
            self.body_vertical = pygame.image.load('Graphics/body_vertical.png').convert_alpha()
            self.body_horizontal = pygame.image.load('Graphics/body_horizontal.png').convert_alpha()
            self.body_tr = pygame.image.load('Graphics/body_tr.png').convert_alpha()
            self.body_tl = pygame.image.load('Graphics/body_tl.png').convert_alpha()
            self.body_br = pygame.image.load('Graphics/body_br.png').convert_alpha()
            self.body_bl = pygame.image.load('Graphics/body_bl.png').convert_alpha()
            self.crunch_sound = pygame.mixer.Sound('Sound/crunch.wav')
            self.has_graphics = True
        except:
            self.has_graphics = False

    def draw_snake(self, surface):
        if self.has_graphics:
            self.update_head_graphics()
            self.update_tail_graphics()
            for index, block in enumerate(self.body):
                block_rect = pygame.Rect(int(block.x * cell_size), int(block.y * cell_size), cell_size, cell_size)
                if index == 0:
                    surface.blit(self.head, block_rect)
                elif index == len(self.body) - 1:
                    surface.blit(self.tail, block_rect)
                else:
                    previous_block = self.body[index + 1] - block
                    next_block = self.body[index - 1] - block
                    if previous_block.x == next_block.x:
                        surface.blit(self.body_vertical, block_rect)
                    elif previous_block.y == next_block.y:
                        surface.blit(self.body_horizontal, block_rect)
                    else:
                        if (previous_block.x == -1 and next_block.y == -1) or (previous_block.y == -1 and next_block.x == -1):
                            surface.blit(self.body_tl, block_rect)
                        elif (previous_block.x == -1 and next_block.y == 1) or (previous_block.y == 1 and next_block.x == -1):
                            surface.blit(self.body_bl, block_rect)
                        elif (previous_block.x == 1 and next_block.y == -1) or (previous_block.y == -1 and next_block.x == 1):
                            surface.blit(self.body_tr, block_rect)
                        elif (previous_block.x == 1 and next_block.y == 1) or (previous_block.y == 1 and next_block.x == 1):
                            surface.blit(self.body_br, block_rect)
        else:
            # Fallback drawing if graphics folder is missing
            for index, block in enumerate(self.body):
                block_rect = pygame.Rect(int(block.x * cell_size), int(block.y * cell_size), cell_size, cell_size)
                color = (0, 100, 255) if index == 0 else (0, 0, 255)
                pygame.draw.rect(surface, color, block_rect)

    def update_head_graphics(self):
        head_relation = self.body[1] - self.body[0]
        if head_relation == Vector2(1, 0): self.head = self.head_left
        elif head_relation == Vector2(-1, 0): self.head = self.head_right
        elif head_relation == Vector2(0, 1): self.head = self.head_up
        elif head_relation == Vector2(0, -1): self.head = self.head_down

    def update_tail_graphics(self):
        tail_relation = self.body[-2] - self.body[-1]
        if tail_relation == Vector2(1, 0): self.tail = self.tail_left
        elif tail_relation == Vector2(-1, 0): self.tail = self.tail_right
        elif tail_relation == Vector2(0, 1): self.tail = self.tail_up
        elif tail_relation == Vector2(0, -1): self.tail = self.tail_down

    def move_snake(self):
        if self.new_block:
            body_copy = self.body[:]
            body_copy.insert(0, body_copy[0] + self.direction)
            self.body = body_copy[:]
            self.new_block = False
        else:
            body_copy = self.body[:-1]
            body_copy.insert(0, body_copy[0] + self.direction)
            self.body = body_copy[:]

    def add_block(self):
        self.new_block = True

    def play_crunch_sound(self):
        if self.has_graphics:
            self.crunch_sound.play()

    def reset(self):
        self.body = [Vector2(5,10), Vector2(4,10), Vector2(3,10)]
        self.direction = Vector2(1,0)

class FRUIT:
    def __init__(self):
        self.pos = Vector2(0,0)

    def draw_fruit(self, surface):
        fruit_rect = pygame.Rect(int(self.pos.x * cell_size), int(self.pos.y * cell_size), cell_size, cell_size)
        surface.blit(apple, fruit_rect)
    
    def randomize(self, snake_body):
        while True:
            self.x = random.randint(0, cell_number - 1)
            self.y = random.randint(0, cell_number - 1)
            self.pos = Vector2(self.x, self.y)
            if self.pos not in snake_body:
                break

class MAIN:
    def __init__(self, auto_reset=True):
        self.snake = SNAKE()
        self.fruit = FRUIT()
        self.fruit.randomize(self.snake.body)
        self.score = 0
        self.is_dead = False
        self.auto_reset = auto_reset
        
        # Give each game instance its own surface so we can render two side-by-side
        self.surface = pygame.Surface((cell_number * cell_size, cell_number * cell_size))
        self.background = self.create_background()

    def create_background(self):
        bg = pygame.Surface((cell_number * cell_size, cell_number * cell_size))
        bg.fill((175, 215, 70))
        grass_color = (167, 209, 61)
        for row in range(cell_number):
            for col in range(cell_number):
                if row % 2 == col % 2:
                    grass_rect = pygame.Rect(col * cell_size, row * cell_size, cell_size, cell_size)
                    pygame.draw.rect(bg, grass_color, grass_rect)
        return bg

    def update(self):
        if not self.is_dead:
            self.snake.move_snake()
            self.check_collision()
            self.check_fail()
    
    def draw_elements(self):
        self.surface.blit(self.background, (0, 0))
        self.fruit.draw_fruit(self.surface)
        self.snake.draw_snake(self.surface)
        self.draw_score()

    def check_collision(self):
        if self.fruit.pos == self.snake.body[0]:
            self.fruit.randomize(self.snake.body)
            self.snake.add_block()
            self.snake.play_crunch_sound()

    def check_fail(self):
        # Hitting boundaries
        if not 0 <= self.snake.body[0].x < cell_number or not 0 <= self.snake.body[0].y < cell_number:
            self.game_over()
        # Hitting self
        for block in self.snake.body[1:]:
            if block == self.snake.body[0]:
                self.game_over()

    def game_over(self):
        if self.auto_reset:
            self.snake.reset()
            self.fruit.randomize(self.snake.body)
            self.score = 0
        else:
            self.is_dead = True

    def draw_score(self):
        self.score = len(self.snake.body) - 3
        score_text = str(self.score)
        score_surface = game_font.render(score_text, True, (56, 74, 12))
        score_x = int(cell_size * cell_number - 60)
        score_y = int(cell_size * cell_number - 40)
        score_rect = score_surface.get_rect(center=(score_x, score_y))
        apple_rect = apple.get_rect(midright=(score_rect.left, score_rect.centery))
        bg_rect = pygame.Rect(apple_rect.left, apple_rect.top, apple_rect.width + score_rect.width + 6, apple_rect.height)
        
        pygame.draw.rect(self.surface, (167, 209, 61), bg_rect)
        self.surface.blit(score_surface, score_rect)
        self.surface.blit(apple, apple_rect)
        pygame.draw.rect(self.surface, (56, 74, 12), bg_rect, 2)

# -----------------------------------------------------------
# 4. AI HELPERS
# -----------------------------------------------------------
def load_model():
    model_path = './model/model.pth'
    if not os.path.exists(model_path):
        return None
    model = Linear_QNet(11, 256, 3)
    model.load_state_dict(torch.load(model_path))
    model.eval() # Set to evaluation mode
    return model

def is_collision_ai(game, pt):
    if pt.x < 0 or pt.x >= cell_number or pt.y < 0 or pt.y >= cell_number:
        return True
    if pt in game.snake.body[1:]:
        return True
    return False

def get_ai_action(game, model):
    head = game.snake.body[0]
    
    point_l = head + Vector2(-1, 0)
    point_r = head + Vector2(1, 0)
    point_u = head + Vector2(0, -1)
    point_d = head + Vector2(0, 1)
    
    dir_l = game.snake.direction == Vector2(-1, 0)
    dir_r = game.snake.direction == Vector2(1, 0)
    dir_u = game.snake.direction == Vector2(0, -1)
    dir_d = game.snake.direction == Vector2(0, 1)

    state = [
        # Danger straight
        (dir_r and is_collision_ai(game, point_r)) or 
        (dir_l and is_collision_ai(game, point_l)) or 
        (dir_u and is_collision_ai(game, point_u)) or 
        (dir_d and is_collision_ai(game, point_d)),

        # Danger right
        (dir_u and is_collision_ai(game, point_r)) or 
        (dir_d and is_collision_ai(game, point_l)) or 
        (dir_l and is_collision_ai(game, point_u)) or 
        (dir_r and is_collision_ai(game, point_d)),

        # Danger left
        (dir_d and is_collision_ai(game, point_r)) or 
        (dir_u and is_collision_ai(game, point_l)) or 
        (dir_r and is_collision_ai(game, point_u)) or 
        (dir_l and is_collision_ai(game, point_d)),
        
        # Move direction
        dir_l, dir_r, dir_u, dir_d,
        
        # Food location 
        game.fruit.pos.x < head.x,  
        game.fruit.pos.x > head.x,  
        game.fruit.pos.y < head.y,  
        game.fruit.pos.y > head.y   
    ]
    
    state_tensor = torch.tensor(np.array(state, dtype=int), dtype=torch.float)
    prediction = model(state_tensor)
    move_idx = torch.argmax(prediction).item()
    
    clock_wise = [Vector2(1, 0), Vector2(0, 1), Vector2(-1, 0), Vector2(0, -1)]
    idx = clock_wise.index(game.snake.direction)
    
    if move_idx == 0:
        new_dir = clock_wise[idx] # Straight
    elif move_idx == 1:
        new_dir = clock_wise[(idx + 1) % 4] # Right turn
    else:
        new_dir = clock_wise[(idx - 1) % 4] # Left turn
        
    return new_dir

# -----------------------------------------------------------
# 5. MODES & MENU
# -----------------------------------------------------------
def draw_text_center(surface, text, font, color, y_offset=0):
    text_surface = font.render(text, True, color)
    text_rect = text_surface.get_rect(center=(surface.get_width()//2, surface.get_height()//2 + y_offset))
    surface.blit(text_surface, text_rect)

def mode_1_human_only():
    pygame.display.set_mode((cell_number * cell_size, cell_number * cell_size))
    game = MAIN(auto_reset=True)
    
    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            if event.type == SCREEN_UPDATE:
                game.update()
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE: return # Go back to menu
                if event.key == pygame.K_UP and game.snake.direction.y != 1:
                    game.snake.direction = Vector2(0, -1)
                if event.key == pygame.K_DOWN and game.snake.direction.y != -1:
                    game.snake.direction = Vector2(0, 1)
                if event.key == pygame.K_LEFT and game.snake.direction.x != 1:
                    game.snake.direction = Vector2(-1, 0)
                if event.key == pygame.K_RIGHT and game.snake.direction.x != -1:
                    game.snake.direction = Vector2(1, 0)

        game.draw_elements()
        screen.blit(game.surface, (0,0))
        pygame.display.update()
        clock.tick(60)

def mode_2_agent_only():
    pygame.display.set_mode((cell_number * cell_size, cell_number * cell_size))
    model = load_model()
    if model is None:
        print("Model not found! Train the AI first using train.py.")
        return
        
    game = MAIN(auto_reset=False)
    waiting_for_start = True
    
    # Wait loop
    while waiting_for_start:
        screen.fill((40, 40, 40))
        draw_text_center(screen, "AGENT READY", title_font, (255, 255, 255), -30)
        draw_text_center(screen, "Press any key to start AI...", game_font, (200, 200, 200), 20)
        pygame.display.update()
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE: return
                waiting_for_start = False

    # AI Game loop
    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE: return
                
            if event.type == SCREEN_UPDATE and not game.is_dead:
                # Agent makes a move
                game.snake.direction = get_ai_action(game, model)
                game.update()

        game.draw_elements()
        screen.blit(game.surface, (0,0))
        
        if game.is_dead:
            overlay = pygame.Surface(screen.get_size())
            overlay.set_alpha(150)
            overlay.fill((0,0,0))
            screen.blit(overlay, (0,0))
            draw_text_center(screen, "GAME OVER", title_font, (255, 0, 0), -30)
            draw_text_center(screen, f"Final Score: {game.score}", game_font, (255, 255, 255), 10)
            draw_text_center(screen, "Press ESC to return to Menu", game_font, (200, 200, 200), 50)
            
        pygame.display.update()
        clock.tick(60)

def mode_3_human_vs_agent():
    model = load_model()
    if model is None:
        print("Model not found! Train the AI first using train.py.")
        return

    # Set up split screen
    width = cell_number * cell_size
    height = cell_number * cell_size
    pygame.display.set_mode((width * 2, height))
    
    human_game = MAIN(auto_reset=False)
    ai_game = MAIN(auto_reset=False)

    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            
            # Key presses for human
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE: return
                if not human_game.is_dead:
                    if event.key == pygame.K_UP and human_game.snake.direction.y != 1:
                        human_game.snake.direction = Vector2(0, -1)
                    if event.key == pygame.K_DOWN and human_game.snake.direction.y != -1:
                        human_game.snake.direction = Vector2(0, 1)
                    if event.key == pygame.K_LEFT and human_game.snake.direction.x != 1:
                        human_game.snake.direction = Vector2(-1, 0)
                    if event.key == pygame.K_RIGHT and human_game.snake.direction.x != -1:
                        human_game.snake.direction = Vector2(1, 0)

            # Update loop
            if event.type == SCREEN_UPDATE:
                if not human_game.is_dead:
                    human_game.update()
                
                if not ai_game.is_dead:
                    ai_game.snake.direction = get_ai_action(ai_game, model)
                    ai_game.update()

        # Draw left screen (Human)
        human_game.draw_elements()
        screen.blit(human_game.surface, (0, 0))
        
        # Draw right screen (AI)
        ai_game.draw_elements()
        screen.blit(ai_game.surface, (width, 0))

        # Divider line
        pygame.draw.line(screen, (0,0,0), (width, 0), (width, height), 5)
        
        # Top Labels
        h_label = game_font.render("HUMAN", True, (0, 0, 0))
        a_label = game_font.render("AGENT", True, (0, 0, 0))
        screen.blit(h_label, (10, 10))
        screen.blit(a_label, (width + 10, 10))

        # Check Win Condition
        if human_game.is_dead and ai_game.is_dead:
            overlay = pygame.Surface((width * 2, height))
            overlay.set_alpha(180)
            overlay.fill((0,0,0))
            screen.blit(overlay, (0,0))
            
            if human_game.score > ai_game.score:
                msg = "HUMAN WINS!"
                col = (0, 255, 0)
            elif ai_game.score > human_game.score:
                msg = "AGENT WINS!"
                col = (255, 0, 0)
            else:
                msg = "IT'S A TIE!"
                col = (255, 255, 0)

            draw_text_center(screen, msg, title_font, col, -40)
            draw_text_center(screen, f"Human: {human_game.score}  |  Agent: {ai_game.score}", game_font, (255,255,255), 10)
            draw_text_center(screen, "Press ESC to return to Menu", game_font, (200,200,200), 60)

        pygame.display.update()
        clock.tick(60)

def main_menu():
    pygame.display.set_mode((cell_number * cell_size, cell_number * cell_size))
    while True:
        screen.fill((50, 70, 90))
        
        draw_text_center(screen, "SNAKE AI", title_font, (255, 255, 255), -100)
        draw_text_center(screen, "1: Human Only", game_font, (200, 200, 200), -20)
        draw_text_center(screen, "2: Agent Only", game_font, (200, 200, 200), 20)
        draw_text_center(screen, "3: Human vs Agent", game_font, (200, 200, 200), 60)
        draw_text_center(screen, "Press 1, 2, or 3 to play", game_font, (100, 200, 100), 120)

        pygame.display.update()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_1:
                    mode_1_human_only()
                    pygame.display.set_mode((cell_number * cell_size, cell_number * cell_size))
                elif event.key == pygame.K_2:
                    mode_2_agent_only()
                    pygame.display.set_mode((cell_number * cell_size, cell_number * cell_size))
                elif event.key == pygame.K_3:
                    mode_3_human_vs_agent()
                    pygame.display.set_mode((cell_number * cell_size, cell_number * cell_size))
                elif event.key == pygame.K_ESCAPE:
                    pygame.quit()
                    sys.exit()

if __name__ == '__main__':
    main_menu()