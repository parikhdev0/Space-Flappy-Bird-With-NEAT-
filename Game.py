import random
import os
import neat
import pygame

pygame.font.init()

width = 900
height = 900
font = pygame.font.SysFont("times", 50)
draw = False

win = pygame.display.set_mode((width,height))
pygame.display.set_caption("Machine Learned Flappy Bird")

pipe_image = pygame.transform.scale2x(pygame.image.load(os.path.join("pipe.jpg")).convert_alpha())
back_image = pygame.transform.scale(pygame.image.load(os.path.join("back.jpg")).convert_alpha(), (900, 900))
bird_image = [pygame.transform.scale2x(pygame.image.load(os.path.join("raven.png")))]
base_image = pygame.transform.scale2x(pygame.image.load(os.path.join("base.png")).convert_alpha())

generation = 0


class bird_class:
    #Add More Bird Images
    image = bird_image

    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.tick_count = 0
        self.vel = 0
        self.height = self.y
        self.img = self.image[0]

    def jump(self):
        self.vel = -10.5
        self.tick_count = 0
        self.height = self.y

    def move(self):

        self.tick_count += 1
        displacement = self.vel * (self.tick_count) + 0.5 * (3) * (self.tick_count) ** 2
        if displacement >= 16:
            displacement = (displacement / abs(displacement)) * 16

        if displacement < 0:
            displacement -= 2

        self.y = self.y + displacement

    def draw(self, win):
        self.image = self.image
        win.blit(self.image[0], (self.x, self.y))

    def get_mask(self):
        return pygame.mask.from_surface(self.img)

class Pipe():
    distance = 200
    velocity = 5

    def __init__(self, x):
        self.x = x
        self.height = 0
        self.top = 0
        self.bottom = 0
        self.top_loc = pygame.transform.flip(pipe_image, False, True)
        self.bottom_loc = pipe_image
        self.passed = False
        self.set_height()

    def set_height(self):
        self.height = random.randrange(50, 450)
        self.top = self.height - self.top_loc.get_height()
        self.bottom = self.height + self.distance

    def move(self):
        self.x -= self.velocity

    def draw(self, win):
        win.blit(self.top_loc, (self.x, self.top))
        win.blit(self.bottom_loc, (self.x, self.bottom))


    def collide(self, bird, win):
        bird_mask = bird.get_mask()
        top_mask = pygame.mask.from_surface(self.top_loc)
        bottom_mask = pygame.mask.from_surface(self.bottom_loc)
        top_offset = (self.x - bird.x, self.top - round(bird.y))
        bottom_offset = (self.x - bird.x, self.bottom - round(bird.y))
        b_point = bird_mask.overlap(bottom_mask, bottom_offset)
        t_point = bird_mask.overlap(top_mask,top_offset)
        if b_point or t_point:
            return True
        return False

class Base:
    """
    Represnts the moving floor of the game
    """
    velocity = 5
    width = base_image.get_width()
    image = base_image

    def __init__(self, y):
        self.y = y
        self.x1 = 0
        self.x2 = self.width

    def move(self):
        self.x1 -= self.velocity
        self.x2 -= self.velocity
        if self.x1 + self.width < 0:
            self.x1 = self.x2 + self.width

        if self.x2 + self.width< 0:
            self.x2 = self.x1 + self.width
    def draw(self, win):
        win.blit(self.image, (self.x1, self.y))
        win.blit(self.image, (self.x2, self.y))

def draw_window(win, birds, pipes, base, score, gen, pipe_ind):
    if gen == 0:
        gen = 1
    win.blit(back_image, (0,0))

    for pipe in pipes:
        pipe.draw(win)

    base.draw(win)
    for bird in birds:
        # draw lines from bird to pipe
        if draw:
            try:
                pygame.draw.line(win, (255,0,0), (bird.x+bird.img.get_width()/2, bird.y + bird.img.get_height()/2), (pipes[pipe_ind].x + pipes[pipe_ind].top_loc.get_width()/2, pipes[pipe_ind].height), 5)
                pygame.draw.line(win, (255,0,0), (bird.x+bird.img.get_width()/2, bird.y + bird.img.get_height()/2), (pipes[pipe_ind].x + pipes[pipe_ind].bottom_loc.get_width()/2, pipes[pipe_ind].bottom), 5)
            except:
                pass
        bird_class.draw(bird, win)

    # score
    score_label = font.render("Score: " + str(score),1,(255,255,255))
    win.blit(score_label, (width - score_label.get_width() - 15, 10))

    # generations
    score_label = font.render("Gens: " + str(gen-1),1,(255,255,255))
    win.blit(score_label, (10, 10))

    # alive
    score_label = font.render("Alive: " + str(len(birds)),1,(255,255,255))
    win.blit(score_label, (10, 50))

    pygame.display.update()

def learn(genomes, config):
    global win, generation
    win = win
    generation += 1
    nets = []
    birds = []
    ge = []
    for genome_id, genome in genomes:
        genome.fitness = 0
        net = neat.nn.FeedForwardNetwork.create(genome, config)
        nets.append(net)
        birds.append(bird_class(230,350))
        ge.append(genome)

    base = Base(height-70)
    pipes = [Pipe(1000)]
    score = 0
    clock = pygame.time.Clock()
    run = True
    while run and len(birds) > 0:
        clock.tick(30)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                run = False
                pygame.quit()
                quit()
                break

        pipe_ind = 0
        if len(birds) > 0:
            if len(pipes) > 1 and birds[0].x > pipes[0].x + pipes[0].top_loc.get_width():  # determine whether to use the first or second
                pipe_ind = 1                                                                 # pipe on the screen for neural network input

        for x, bird in enumerate(birds):  # give each bird a fitness of 0.1 for each frame it stays alive
            ge[x].fitness += 0.1
            bird.move()

            output = nets[birds.index(bird)].activate((bird.y, abs(bird.y - pipes[pipe_ind].height), abs(bird.y - pipes[pipe_ind].bottom)))

            if output[0] > 0.5:
                bird.jump()

        base.move()

        rem = []
        add_pipe = False
        for pipe in pipes:
            pipe.move()
            # check for collision
            for bird in birds:
                if pipe.collide(bird, win):
                    ge[birds.index(bird)].fitness -= 1
                    nets.pop(birds.index(bird))
                    ge.pop(birds.index(bird))
                    birds.pop(birds.index(bird))

            if pipe.x + pipe.top_loc.get_width() < 0:
                rem.append(pipe)

            if not pipe.passed and pipe.x < bird.x:
                pipe.passed = True
                add_pipe = True

        if add_pipe:
            score += 1
            # can add this line to give more reward for passing through a pipe (not required)
            for genome in ge:
                genome.fitness += 5
            pipes.append(Pipe(width))

        for r in rem:
            pipes.remove(r)

        for bird in birds:
            if bird.y + bird.img.get_height() - 10 >= height-70 or bird.y < -50:
                nets.pop(birds.index(bird))
                ge.pop(birds.index(bird))
                birds.pop(birds.index(bird))

        draw_window(win, birds, pipes, base, score, generation, pipe_ind)

def run(config_file):
    config = neat.config.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_file)
    p = neat.Population(config)
    p.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)
    winner = p.run(learn, 50)
    print('\nBest genome:\n{!s}'.format(winner))
local_dir = os.path.dirname(__file__)
config_path = os.path.join(local_dir, 'settings.txt')
run(config_path)
