import pygame
import sys
from pygame.math import Vector2
import random
import numpy as np

pygame.mixer.pre_init(44100, -16, 2, 512)
pygame.init()
cell_size = 30
cell_number = 20
game_font = pygame.font.Font('Font/PoetsenOne-Regular.ttf', 25)

# Game Speed: Increase this to train faster (e.g., 60, 100, 1000)
SPEED = 1000

class SNAKE:
    def __init__(self):
        self.body = [Vector2(5,10), Vector2(4,10), Vector2(3,10)]
        # IMPORTANT: AI needs an initial direction to calculate turns. 
        self.direction = Vector2(1,0) 
        self.new_block = False

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

    def draw_snake(self, screen):
        self.update_head_graphics()
        self.update_tail_graphics()
        for index, block in enumerate(self.body):
            block_rect = pygame.Rect(int(block.x * cell_size), int(block.y * cell_size), cell_size, cell_size)
            if index == 0:
                screen.blit(self.head, block_rect)
            elif index == len(self.body) - 1:
                screen.blit(self.tail, block_rect)
            else:
                previous_block = self.body[index + 1] - block
                next_block = self.body[index - 1] - block
                if previous_block.x == next_block.x:
                    screen.blit(self.body_vertical, block_rect)
                elif previous_block.y == next_block.y:
                    screen.blit(self.body_horizontal, block_rect)
                else:
                    if (previous_block.x == -1 and next_block.y == -1) or (previous_block.y == -1 and next_block.x == -1):
                        screen.blit(self.body_tl, block_rect)
                    elif (previous_block.x == -1 and next_block.y == 1) or (previous_block.y == 1 and next_block.x == -1):
                        screen.blit(self.body_bl, block_rect)
                    elif (previous_block.x == 1 and next_block.y == -1) or (previous_block.y == -1 and next_block.x == 1):
                        screen.blit(self.body_tr, block_rect)
                    elif (previous_block.x == 1 and next_block.y == 1) or (previous_block.y == 1 and next_block.x == 1):
                        screen.blit(self.body_br, block_rect)
    
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
        self.crunch_sound.play()

class FRUIT:
    def __init__(self):
        self.apple = pygame.image.load('Graphics/apple.png').convert_alpha()
        self.pos = Vector2(0,0)

    def draw_fruit(self, screen):
        fruit_rect = pygame.Rect(int(self.pos.x * cell_size), int(self.pos.y * cell_size), cell_size, cell_size)
        screen.blit(self.apple, fruit_rect)
    
    def randomize(self, snake_body):
        while True:
            self.x = random.randint(0, cell_number - 1)
            self.y = random.randint(0, cell_number - 1)
            self.pos = Vector2(self.x, self.y)
            if self.pos not in snake_body:
                break

class MAIN_AI:
    def __init__(self):
        self.screen = pygame.display.set_mode((cell_number * cell_size, cell_number * cell_size))
        self.clock = pygame.time.Clock()
        self.apple = pygame.image.load('Graphics/apple.png').convert_alpha()
        self.reset()

    def reset(self):
        self.snake = SNAKE()
        self.fruit = FRUIT()
        self.fruit.randomize(self.snake.body)
        self.score = 0
        self.frame_iteration = 0
        self.background = self.create_background()

    def create_background(self):
        background = pygame.Surface((cell_number * cell_size, cell_number * cell_size))
        background.fill((175, 215, 70))
        grass_color = (167, 209, 61)
        for row in range(cell_number):
            for col in range(cell_number):
                if row % 2 == col % 2:
                    grass_rect = pygame.Rect(col * cell_size, row * cell_size, cell_size, cell_size)
                    pygame.draw.rect(background, grass_color, grass_rect)
        return background

    def play_step(self, action):
        self.frame_iteration += 1
        
        # 1. Collect user input (just to allow closing the window)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
        
        # 2. Move
        self._move(action)
        self.snake.move_snake()
        
        # 3. Check if game over
        reward = 0
        game_over = False
        
        # AI fails if it hits a wall, itself, or starves (takes too long)
        if self.is_collision() or self.frame_iteration > 100 * len(self.snake.body):
            game_over = True
            reward = -10
            return reward, game_over, self.score
            
        # 4. Place new food or just move
        if self.snake.body[0] == self.fruit.pos:
            self.score += 1
            reward = 10
            self.snake.play_crunch_sound()
            self.fruit.randomize(self.snake.body)
            self.snake.add_block()
        
        # 5. Update ui and clock
        self.draw_elements()
        pygame.display.update()
        self.clock.tick(SPEED)
        
        # 6. Return game over and score
        return reward, game_over, self.score

    def is_collision(self, pt=None):
        if pt is None:
            pt = self.snake.body[0]
        # Hits boundary
        if pt.x < 0 or pt.x >= cell_number or pt.y < 0 or pt.y >= cell_number:
            return True
        # Hits itself
        if pt in self.snake.body[1:]:
            return True
        return False

    def _move(self, action):
        # [straight, right, left]
        clock_wise = [Vector2(1, 0), Vector2(0, 1), Vector2(-1, 0), Vector2(0, -1)]
        idx = clock_wise.index(self.snake.direction)

        if np.array_equal(action, [1, 0, 0]):
            new_dir = clock_wise[idx] # no change
        elif np.array_equal(action, [0, 1, 0]):
            new_dir = clock_wise[(idx + 1) % 4] # right turn r -> d -> l -> u
        else: # [0, 0, 1]
            new_dir = clock_wise[(idx - 1) % 4] # left turn r -> u -> l -> d

        self.snake.direction = new_dir

    def draw_elements(self):
        self.screen.blit(self.background, (0, 0))
        self.fruit.draw_fruit(self.screen)
        self.snake.draw_snake(self.screen)
        self.draw_score()

    def draw_score(self):
        score_text = str(self.score)
        score_surface = game_font.render(score_text, True, (56, 74, 12))
        score_x = int(cell_size * cell_number - 60)
        score_y = int(cell_size * cell_number - 40)
        score_rect = score_surface.get_rect(center=(score_x, score_y))
        apple_rect = self.apple.get_rect(midright=(score_rect.left, score_rect.centery))
        bg_rect = pygame.Rect(apple_rect.left, apple_rect.top, apple_rect.width + score_rect.width + 6, apple_rect.height)
        
        pygame.draw.rect(self.screen, (167, 209, 61), bg_rect)
        self.screen.blit(score_surface, score_rect)
        self.screen.blit(self.apple, apple_rect)
        pygame.draw.rect(self.screen, (56, 74, 12), bg_rect, 2)