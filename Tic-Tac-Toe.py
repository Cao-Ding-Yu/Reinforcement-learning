import numpy as np
import matplotlib.pyplot as plt

class Agent:
    def __init__(self, index, epsilon, alpha):
        self.index = index
        self.epsilon = epsilon
        self.alpha = alpha
        self.value = np.zeros((3,3,3,3,3,3,3,3,3))
        self.stored_outcome = np.zeros(9).astype(np.int8)

    def reset(self):
        self.stored_outcome = np.zeros(9).astype(np.int8)

    def move(self, state):
        outcome = state.copy()
        available = np.where(outcome == 0)[0]
        if np.random.binomial(1, self.epsilon):
            outcome[np.random.choice(available)] = self.index
        else:
            temp_value = np.zeros(len(available))
            for i in range(len(available)):
                temp_outcome = outcome.copy()
                temp_outcome[available[i]] = self.index
                temp_value[i] = self.value[tuple(temp_outcome)]
            choose = np.argmax(temp_value)
            outcome[available[choose]] = self.index
        error = self.value[tuple(outcome)] - self.value[tuple(self.stored_outcome)]
        self.value[tuple(self.stored_outcome)] += self.alpha * error
        self.stored_outcome = outcome.copy()
        return outcome

def judge(outcome, index):
    triple = np.repeat(index, 3)
    winner = 0
    if 0 not in outcome:
        winner = 3
    if (outcome[0:3]==triple).all() or (outcome[3:6]==triple).all() or (outcome[6:9]==triple).all():
        winner = index
    if (outcome[0:7:3] == triple).all() or (outcome[1:8:3] == triple).all() or (outcome[2:9:3] == triple).all():
        winner = index
    if (outcome[0:9:4] == triple).all() or (outcome[2:7:2] == triple).all():
        winner = index
    return winner

def Rate(Winner, step = 250, duration = 500):
    rate1 = np.zeros(int((trial-duration)/step)+1) # Agent1 胜率
    rate2 = np.zeros(int((trial-duration)/step)+1) # Agent2 胜率
    rate3 = np.zeros(int((trial-duration)/step)+1) # 平局概率
    for i in range(len(rate1)):
        rate1[i] = np.sum(Winner[step*i:duration+step*i]==1)/duration
        rate2[i] = np.sum(Winner[step*i:duration+step*i]==2)/duration
        rate3[i] = np.sum(Winner[step*i:duration+step*i]==3)/duration
    return rate1,rate2,rate3

if __name__ == '__main__':
    Agent1 = Agent(1, 0.1, 0.1)
    Agent2 = Agent(2, 0.1, 0.1)
    trial = 30000
    Winner = np.zeros(trial)
    for i in range(trial):
        if i==20000:
            Agent1.epsilon = 0
            Agent2.epsilon = 0
        Agent1.reset()
        Agent2.reset()
        state = np.zeros(9).astype(np.int8)
        winner = 0
        while winner == 0:
            outcome = Agent1.move(state)
            winner = judge(outcome, 1)
            if winner == 1:
                Agent1.value[tuple(outcome)] = 1
                Agent2.value[tuple(state)] = -1
            elif winner == 0:
                state = Agent2.move(outcome)
                winner = judge(state, 2)
                if winner == 2:
                    Agent1.value[tuple(outcome)] = -1
                    Agent2.value[tuple(state)] = 1
        Winner[i] = winner
    rate1, rate2, rate3 = Rate(Winner)
    fig,ax=plt.subplots(figsize=(16,9))
    plt.plot(rate1,linewidth=4,marker='.',markersize=20,color="#0071B7",label="Agent1")
    plt.plot(rate2,linewidth=4,marker='.',markersize=20,color="#DB2C2C",label="Agent2")
    plt.plot(rate3,linewidth=4,marker='.',markersize=20,color="#FAB70D",label="Draw")
    plt.xticks(np.arange(0,121,40),np.arange(0,31+1,10),fontsize=30)
    plt.yticks(np.arange(0,1.1,0.2),np.round(np.arange(0,1.1,0.2),2),fontsize=30)
    plt.xlabel("Trials(x1k)",fontsize=30)
    plt.ylabel("Winning Rate",fontsize=30)
    plt.legend(loc="best",fontsize=25)
    plt.tick_params(width=4,length=10)
    ax.spines[:].set_linewidth(4)
    plt.show()

