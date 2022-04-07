import gym
import numpy as np
import pickle
# first-visit MC 
FILENAME = 'blackjack_policy'
def play_jack():
    env = gym.make('Blackjack-v1')
    with open(FILENAME,'rb') as f:
        policy = pickle.load(f)
    numEpisodes = 1000
    rewards = np.zeros(numEpisodes)
    totalReward = 0
    wins = 0
    losses = 0
    draws = 0
    for i in range(numEpisodes):
        observation = env.reset()
        done = False
        while not done:
            action = policy[observation]
            observation_, reward, done, info = env.step(action)            
            observation = observation_
        totalReward += reward
        rewards[i] = totalReward

        if reward >= 1:
            wins += 1
        elif reward == 0:
            draws += 1
        elif reward == -1:
            losses += 1
    
    wins /= numEpisodes
    losses /= numEpisodes
    draws /= numEpisodes
    print('win rate', wins, 'loss rate', losses, 'draw rate', draws)
    
def main(EPS = 0.05,GAMMA = 1.0,numEpisodes = 10000,EPS_DECAY=1e-7,EPS_MIN=9):
    env = gym.make('Blackjack-v1')
    Q = {}
    agentSumSpace = [i for i in range(4, 22)]
    dealerShowCardSpace = [i+1 for i in range(10)]
    agentAceSpace = [False, True]
    actionSpace = [0, 1] # stick or hit

    stateSpace = []
    returns = {}
    pairsVisited = {}
    # initializing variables to default values
    for total in agentSumSpace:
        for card in dealerShowCardSpace:
            for ace in agentAceSpace:
                for action in actionSpace:
                    Q[((total, card, ace), action)] = 0
                    returns[((total, card, ace), action)] = 0
                    pairsVisited[((total, card, ace), action)] = 0
                stateSpace.append((total, card, ace))
    
    policy = {}
    for state in stateSpace:
        policy[state] = np.random.choice(actionSpace)

    for i in range(numEpisodes):
        statesActionsReturns = []
        memory = []
        observation = env.reset()
        done = False
        while not done:
            action = policy[observation]
            observation_, reward, done, info = env.step(action)
            memory.append((observation[0], observation[1], observation[2], action, reward))
            observation = observation_
        memory.append((observation[0], observation[1], observation[2], action, reward))    

        G = 0
        last = True
        for playerSum, dealerCard, usableAce, action, reward in reversed(memory):
            if last:
                last = False
            else:
                statesActionsReturns.append((playerSum, dealerCard, usableAce, action, G))
            G = GAMMA*G + reward

        statesActionsReturns.reverse()
        statesActionsVisited = []

        for playerSum, dealerCard, usableAce, action, G in statesActionsReturns:
            sa = ((playerSum, dealerCard, usableAce), action)
            if sa not in statesActionsVisited:
                pairsVisited[sa] += 1
                # incremental implementation
                returns[(sa)] += (1 / pairsVisited[(sa)])*(G-returns[(sa)])
                Q[sa] = returns[sa]
                rand = np.random.random()
                if rand < 1 - EPS:
                    state = (playerSum, dealerCard, usableAce)
                    values = np.array([Q[(state, a)] for a in actionSpace ])
                    best = np.random.choice(np.where(values==values.max())[0])
                    policy[state] = actionSpace[best]
                else:
                    policy[state] = np.random.choice(actionSpace)
                statesActionsVisited.append(sa)
        if EPS - EPS_DECAY > EPS_MIN:
            EPS -= EPS_DECAY
        else:
            EPS = EPS_MIN
        
    # saving
    with open(FILENAME,'wb') as f:
        pickle.dump(policy,f)
    
if __name__ == '__main__':
    main()
    play_jack()
    

    