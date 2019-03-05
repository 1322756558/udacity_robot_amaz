import random

class Robot(object):

    def __init__(self, maze, alpha=0.5, gamma=0.9, epsilon0=0.5):
        
        # 迷宫实体
        self.maze = maze
        # 有效动作
        self.valid_actions = self.maze.valid_actions
        # 状态
        self.state = None
        # 动作
        self.action = None

        # Set Parameters of the Learning Robot
        # 𝑞(𝑠𝑡,𝑎)=(1−𝛼)×𝑞(𝑠𝑡,𝑎)+𝛼×(𝑅𝑡+1+𝛾×max𝑎𝑞(𝑎,𝑠𝑡+1))
        self.alpha = alpha
        self.gamma = gamma
        
        # 随机选择动作的概率
        self.epsilon0 = epsilon0
        self.epsilon = epsilon0
        self.t = 0

        self.Qtable = {}
        self.reset()

    def reset(self):
        """
        Reset the robot
        重置智能体
        """
        self.state = self.sense_state()
        self.create_Qtable_line(self.state)

    def set_status(self, learning=False, testing=False):
        """
        Determine whether the robot is learning its q table, or
        exceuting the testing procedure.
        确定智能体是在学习q表，还是执行测试程序。
        """
        self.learning = learning
        self.testing = testing

    def update_parameter(self):
        """
        Some of the paramters of the q learning robot can be altered,
        update these parameters when necessary.
        Q-learn 智能体的一些参数可以改变,必要时更新这些参数
        """
        if self.testing:
            # TODO 1. No random choice when testing
#             测试时没有随机选择
            pass
        else:
            # TODO 2. Update parameters when learning
#             学习时更新学习参数
#             更新步长及随机选择的概率
            if self.epsilon < 0.01:
                self.epsilon = 0.01
            else:
                self.epsilon -= 0.05

        return self.epsilon

    def sense_state(self):
        """
        Get the current state of the robot. In this
        在这里得到智能体的当前状态
        """

        # TODO 3. Return robot's current state
#         maze,返回当前状态
        return self.maze.sense_robot()

    def create_Qtable_line(self, state):
        """
        Create the qtable with the current state
        使用当前状态创建q-table
        """
        # TODO 4. Create qtable with current state
        # 创建qtable对于当前状态
        # Our qtable should be a two level dict,
        # 我们的qtable需要有两个等级的字典
        # Qtable[state] ={'u':xx, 'd':xx, ...}
        # If Qtable[state] already exits, then do
        # 如果已经存在,则不改变他
        # not change it.
        if state in self.Qtable:
            pass
        else:
            self.Qtable.setdefault(state, {a: 0.0 for a in self.valid_actions})

    def choose_action(self):
        """
        Return an action according to given rules
        根据给定规则返回一个动作
        """
        def is_random_exploration():

            # TODO 5. Return whether do random choice
            # hint: generate a random number, and compare
            # it with epsilon
            # 返回是否随机选择，提示：生成一个随机数，并将其与epsilon进行比较
            
            return random.random() < self.epsilon

        if self.learning:
            if is_random_exploration():
                # TODO 6. Return random choose aciton
                # 返回随机选择的动作
                return random.choice(self.valid_actions)
            else:
                # TODO 7. Return action with highest q value
                # 返回q值最高的动作
                return max(self.Qtable[self.state], key=self.Qtable[self.state].get)
        elif self.testing:
            # TODO 7. choose action with highest q value
            # 选择q值最高的动作
            return max(self.Qtable[self.state], key=self.Qtable[self.state].get)
        else:
            # TODO 6. Return random choose aciton
            # 选择随机动作
            return random.choice(self.valid_actions)

    def update_Qtable(self, r, action, next_state):
        """
        Update the qtable according to the given rule.
        根据给定的规则更新qtable。
        
        """
        if self.learning:
            pass
            # TODO 8. When learning, update the q table according
            # to the given rules
            # 根据给定规则更新q表
            # 𝑞(𝑠𝑡,𝑎)=(1−𝛼)×𝑞(𝑠𝑡,𝑎)+𝛼×(𝑅𝑡+1+𝛾×max𝑎𝑞(𝑎,𝑠𝑡+1))
            accumulate = self.Qtable[self.state][action]
            new = r + self.gamma * float(max(self.Qtable[next_state].values()))
            self.Qtable[self.state][action] += self.alpha * (new - accumulate)
    def update(self):
        """
        Describle the procedure what to do when update the robot.
        描述更新智能体时要执行的操作步骤
        Called every time in every epoch in training or testing.
        在每个时间,在每个epoch 在 学习或者测试
        Return current action and reward.
        返回当前行动和奖励
        """
        # 得到当前状态
        self.state = self.sense_state() # Get the current state
        # 对于当前状态创建qtable行
        self.create_Qtable_line(self.state) # For the state, create q table line
        
        # 选择动作对于当前状态
        action = self.choose_action() # choose action for this state
        # 移动机器人到给定的动作
        reward = self.maze.move_robot(action) # move robot for given action
        
        # 获得下一个状态
        next_state = self.sense_state() # get next state
        # 创建下一个状态的qtable行
        self.create_Qtable_line(next_state) # create q table line for next state

        if self.learning and not self.testing:
            # 更新q-table
            self.update_Qtable(reward, action, next_state) # update q table
            # 更新参数
            self.update_parameter() # update parameters
        # 动作奖励
        return action, reward
