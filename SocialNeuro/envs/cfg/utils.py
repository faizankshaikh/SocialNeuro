def _get_rewards_egoistic(player1_alive, player2_alive):
    return {
        "player1": 0 if player1_alive else -1,
        "player2": 0 if player2_alive else -1,
    }

def _get_rewards_prosocial(player1_alive, player2_alive):
    return {
        "player1": 0 if player1_alive and player2_alive else -1,
        "player2": 0 if player1_alive and player2_alive else -1,
    }