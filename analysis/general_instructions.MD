1. Load all the experiment data - train and test. 
2. Normalize the rewards within each task - this make comparision between tasks logical
3. For the train data:
3.1. Summarize the training epochs - RL loss, average reward and distribution of actions, compute the partner's reward
3.2. For the distribution of actions - compute the entropy of the policy across the training
4. for the test set the analysis should follow the axis of generalization.
4.1. Generalization to new opponents within the task:
4.1.1 Average reward (self and opponent)
4.1.2 Changes in the policy - measure the KLD between the training policy and the test policy
4.1.3 Summarize both metrics in the following plot: for 4.1.1 generate a simple point plot: the x-axis represents the opponent's policy (bins) and the y-axis is the average self reward. For 4.1.2 make the same plot - here the y-axis is the KLD.
4.1.4. To which new opponents does the agent generalizes "best", and to which "wosrt"
4.2. Generalization to new tasks:
Same analysis as in 4.1 - reward and policy change (if any), plots and follow up analysis
4.3 Genralization to new environment and new opponents - here we want to compute again the reward and policy change.
4.3.1 To which combinations (task and opponent) the agent fails to generalize (in terms of reward), which new environments change the policy the most?
5. Summary: Now we want to show the large traing to test map. The plot looks in the following manner: on the x-axis are the opponent bins. On the y-axis are the tasks - keep the same order (say alphabetically). This creates a grid of tasks * opponents. Each cell is colored by the average self reward (the average reward gained by the trained agents). The cell corresponding to the training environment is set to zero - all other cells show the reward relative to this cell. So if the agent is trained on the PD with p=[0.5,0.7] and they got on average 0.8 reward (normalized reward!!) and when they were tested in the HD game with p=[0.1,0.3] they got on average 0.5 - the value of the (HD, [0.1,0.3]) cell is -0.3.    