import pygame
import trainer
import numpy as np
import matplotlib.pyplot as plt

#color set up
BLACK = (0,0,0)
WHITE = (255, 255, 255)

#box size vars
width = 20
height = 20
margin = 5

print("Starting up!")
#train dataset
network_obj = trainer.Trainer()

#establish play variable
play_bool = True
while play_bool:
    print("Draw a number for the program to guess")
    grid = []
    for row in range(28):
        grid.append([])
        for column in range(28):
            grid[row].append(0)

    pygame.init()

    size = [710, 710]
    screen = pygame.display.set_mode(size)
    pygame.display.set_caption("MNIST Test")
    done = False
    clock = pygame.time.Clock()
    mouseclick = False

    #Program Loop
    while not done:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True
            elif event.type == pygame.MOUSEBUTTONDOWN:
                mouseclick = not mouseclick
            elif event.type == pygame.MOUSEMOTION and mouseclick:
                pos = pygame.mouse.get_pos()
                column = pos[0]//(width +margin)
                row = pos[1]//(height+margin)
                if row in range(28) and column in range(28):
                    grid[row][column] = 1
                #print("Grid Coords: ", row, column)

        screen.fill(BLACK)
        for row in range(28):
            for column in range(28):
                color = WHITE
                if grid[row][column] == 1:
                    color = BLACK
                pygame.draw.rect(screen, color, [(margin + width) * column + margin,
                                                 (margin+ height) * row + margin,
                                                 width, height])
        # fps is 60
        clock.tick(60)

        #update screen
        pygame.display.flip()

    #no hanging close

    #wrap grid as np array
    np_grid = np.array(grid)
    np_grid = np_grid.reshape(1, 28*28)

    guess = np.argmax(network_obj.network.predict(np_grid), axis=1)
    print("You're number was guessed to be: ", guess[0])
    play_check = input("Would you like to play again? (y or n): ")
    while not (play_check.lower() == "n" or play_check.lower() == "y"):
        play_check = input("Enter a valid input (y or n): ")
    if play_check.lower() == "n":
        play_bool = False






