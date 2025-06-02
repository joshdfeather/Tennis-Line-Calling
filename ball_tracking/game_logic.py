import numpy as np
#   ___________________________________
#  |                                   |
#  |                 7                 | 
#  |-----------------------------------|
#  |                 |                 |
#  |                 |                 |
#  |       5         |        4        |
#  |                 |                 |
#  |-----------------------------------| 
#  |                 |                 |
#  |                 |                 |
#  |       1         |       2         |
#  |                 |                 |
#  |-----------------------------------|
#  |                 3                 |  
#  |___________________________________|

# Constants
COURT_LENGTH = 2378
COURT_WIDTH = 823
NET_POSITION = COURT_LENGTH / 2
SERVICE_BOX_LENGTH = 640
SERVICE_BOX_WIDTH = COURT_WIDTH / 2
BACKCOURT_LENGTH = 549
DEUCE = False

# Variables
player_shot = 1
state = 1  # 1: First Serve, 2: Second Serve, 0: Ongoing Rally
points = {1: 0, 2: 0}  # Current points in the game
games = {1: 0, 2: 0}  # Current games won in the set
sets = {1: 0, 2: 0}  # Sets won
ball_landing_zones = {1: [], 2: []}  # Tracking performance
serving = 1

# Functions
def determine_landing_zone(x, y):
    if not (0 <= y <= COURT_LENGTH and 0 <= x <= COURT_WIDTH):
        return 7  # Out of bounds

    if y < NET_POSITION:  # Player 1's side
        if x < SERVICE_BOX_WIDTH and y >= BACKCOURT_LENGTH:
            return 1  # Player 1 Left Service Box
        elif x >= SERVICE_BOX_WIDTH and y >= BACKCOURT_LENGTH:
            return 2  # Player 1 Right Service Box
        else:
            return 3  # Player 1 Backcourt
    else:  # Player 2's side
        if x < SERVICE_BOX_WIDTH and y <= BACKCOURT_LENGTH + NET_POSITION:
            return 5  # Player 2 Left Service Box
        elif x >= SERVICE_BOX_WIDTH and y <= BACKCOURT_LENGTH + NET_POSITION:
            return 4  # Player 2 Right Service Box
        else:
            return 6  # Player 2 Backcourt


def update_score(winner):
    global state, player_shot, serving, DEUCE
    points[winner] += 1
    print(f"Game Score: {points[1]} - {points[2]}")
    state = 1
    player_shot = serving
    if check_win_game():
        reset_game()


def serve_fault(player, landing_zone):
    valid_zones = {1: [4, 5], 2: [1, 2]}  # Valid zones for Player 1 and Player 2
    serve_side = points[player] % 2
    required_zone = valid_zones[player][serve_side]
    return landing_zone != required_zone


def rally_valid(player, landing_zone):
    if player == 1:
        return 4 <= landing_zone <= 6  # Ball must land on Player 2's side
    else:
        return 1 <= landing_zone <= 3  # Ball must land on Player 1's side


def check_win_game():
    global DEUCE
    if DEUCE:
        if abs(points[1] - points[2]) == 2:
            win_game(1 if points[1] > points[2] else 2)
            return True
    elif points[1] >= 4 and points[1] - points[2] >= 2:
        win_game(1)
        return True
    elif points[2] >= 4 and points[2] - points[1] >= 2:
        win_game(2)
        return True
    elif points[1] == points[2] == 3:  # Deuce starts at 3-3
        DEUCE = True
    return False


def win_game(winner):
    games[winner] += 1
    print(f"Player {winner} wins the game!")
    check_win_set()


def reset_game():
    global points, DEUCE
    points = {1: 0, 2: 0}
    DEUCE = False


def check_win_set():
    if games[1] >= 6 and games[1] - games[2] >= 2:
        win_set(1)
    elif games[2] >= 6 and games[2] - games[1] >= 2:
        win_set(2)
    elif games[1] == 7 or games[2] == 7:  # Tiebreak condition
        win_set(1 if games[1] > games[2] else 2)


def win_set(winner):
    sets[winner] += 1
    print(f"Player {winner} wins the set!")
    reset_set()


def reset_set():
    global games
    games = {1: 0, 2: 0}


def shot(x, y, returned=True):
    global player_shot, state
    landing_zone = determine_landing_zone(x, y)
    print(f'Player {player_shot} lands in {landing_zone}')
    
    # Serve phase
    if state in (1, 2):  
        if serve_fault(player_shot, landing_zone):
            if state == 2:
                update_score(3 - player_shot) 
            else:
                state = 2  # Second serve
            return
        if not returned:
            update_score(player_shot)
        else:
            state = 0
            ball_landing_zones[player_shot].append(landing_zone)
            player_shot = 3 - player_shot   
        return

    # Rally phase
    if state == 0:  
        if not returned:
            update_score(player_shot)
            return
        if not rally_valid(player_shot, landing_zone):
            update_score(3 - player_shot)  # Opponent gains a point
            return
        ball_landing_zones[player_shot].append(landing_zone)
        player_shot = 3 - player_shot  # Switch players


shot(15, 5)  # Serve fault
shot(15, 5)  # Double fault, Player 2 gains a point

shot(600, 1500, returned=True)  # Valid serve, Player 2 returns
shot(300, 1000, returned=True)  # Rally continues
shot(600, 1600, returned=False)  # Player 2 fails to return, Player 1 gains a point

shot(400, 1400, returned=False)  # Player 1 serves, valid shot

shot(600, 1500, returned=True)  # Player 1 serves, valid shot
shot(30, 250, returned=True)  # Rally continues
shot(800, 1800, returned=False)  # Player 2 fails to return, Player 1 gains a point

shot(400, 1400, returned=True)  # Player 1 serves, valid shot
shot(300, 1000, returned=True)  # Rally continues
shot(200, 1200, returned=False)  # Player 2 fails to return, Player 1 gains a point