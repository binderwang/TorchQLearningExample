--[[
--          Example of Re-inforcement learning using the Q function described in this paper from deepmind.
--          https://www.cs.toronto.edu/~vmnih/docs/dqn.pdf
--
--          The agent plays a game of pong against another agent. The agent can choose between up/down/stay actions.
--          Agent is rewarded for winning a game, is penalised for losing.
]]

require 'cunn'
require 'nn'
require 'cudnn'
require 'rnn'
require 'Memory'
require 'optim'

math.randomseed(os.time())

-- Creates the TorchPongLearning class that can be instantiated to create multiple agents.
local TorchPongLearning = torch.class('TorchPongLearning')

--[[ Helper function: Chooses a random value between the two boundaries.]] --
local function randf(s, e)
    return (math.random(0, (e - s) * 9999) / 10000) + s;
end

function TorchPongLearning:__init(params)
    self.size = params.size or 80 -- The size of the screen once it has been resized (size x size).
    self.memorySize = params.memorySize or 590000 -- The size of the memory (nb of experiences kept).
    self.numberOfFrames = params.numberOfFrames or 4 -- The number of frames we want to give the network at one time.
    self.nbActions = params.nbActions or 3 -- Number of actions (up/down/stay).
    self.discount = params.discount or 0.99 -- The discount is used to force the network to choose states that lead to the reward quicker (0 to 1)
    self.batchSize = params.batchSize or 32 -- The mini-batch size for training. Samples are randomly taken from memory till mini-batch size.
    self.initialEpsilonValue = params.initialEpsilon or 1 -- The probability of choosing a random action (in training).
    self.epsilonMinimumValue = params.epsilonMinimumValue or 0.05 -- The minimum value we want epsilon to reach in training. (0 to 1)
    self.numberOfFramesToAnnealEpsilon = params.numberOfFramesToAnnealEpsilon or 8000 -- Number of frames to anneal epsilon.
    self.epsilon = self.initialEpsilonValue
    self.memory = Memory({ maxMemory = self.memorySize, discount = self.discount }) -- Where we store previous experiences.
    self.criterion = nn.MSECriterion():cuda() -- Mean Squared Error for our loss function.
    self.frameSkips = params.frameSkips or 3 -- The agent only makes a decision every frameSkips frames. (check deepmind paper for more info).
    self.framesSkipped = self.frameSkips -- Initialise the frameskip method.
    self.lastState = nil -- The last state that the agent had to decide an action upon.
    self.lastAction = 2 -- The action that the agent chose on the last state.
    self.train = params.train or true -- Do we want to train a new network.
    self.loadModel = params.loadModel or false -- Set to true to load a model.
    self.fileName = params.fileName -- The fileName we want to load/save from/to.

    -- Params for Stochastic Gradient Descent (our optimizer).
    self.sgdParams = {
        learningRate = 1e-7,
        learningRateDecay = 1e-9,
        weightDecay = 0,
        momentum = 0.9,
        dampening = 0,
        nesterov = true
    }

    if (params.loadModel) then
        self.model = torch.load(self.fileName)
    else
        -- Create the base model.
        local model = nn.Sequential()
        model:add(cudnn.SpatialConvolution(self.numberOfFrames, 32, 8, 8, 3, 3))
        model:add(cudnn.ReLU())
        model:add(cudnn.SpatialMaxPooling(2, 2))
        model:add(cudnn.SpatialConvolution(32, 64, 4, 4, 2, 2))
        model:add(cudnn.ReLU())
        model:add(cudnn.SpatialMaxPooling(2, 2))
        model:add(nn.View(-1, 256))
        model:add(nn.Linear(256, 256))
        model:add(cudnn.ReLU())
        model:add(nn.Linear(256, self.nbActions))
        model:cuda()
        self.model = model
    end
end

-- Assume that currentState is an 80x80 tensor.
function TorchPongLearning:getAction(currentState, reward, isTerminated)
    -- We apply the same actions to nbFrameSkips set. So if this was 3, the action chosen occurs to 3 consecutive frames.
    if (self.framesSkipped < self.frameSkips) then
        self.framesSkipped = self.framesSkipped + 1
        return self.lastAction
    else
        self.framesSkipped = 0
    end
    local size = self.size
    if (self.lastState == nil) then
        -- The first time (game just started).
        self.lastState = torch.CudaTensor(self.numberOfFrames, size, size):zero()
        return 2 -- The action to remain in the same position.
    end

    -- We move up the values in the tensor, removing the last remembered frame, adding the new current frame.
    local tempState = self.lastState:clone()
    for x = 2, tempState:size(1) do
        tempState[x - 1] = tempState[x]
    end
    tempState[tempState:size(1)] = currentState
    currentState = tempState

    self.memory:remember({
        inputState = self.lastState,
        action = self.lastAction,
        reward = reward,
        nextState = currentState,
        gameOver = isTerminated
    })

    self.lastState = currentState

    local action
    -- Decides if we should choose a random action, or an action from the policy network.
    if (self.train and randf(0, 1) <= self.epsilon) then
        action = math.random(1, self.nbActions)
    else
        -- Forward the last numberOfFrames states through the network.
        local q = self.model:forward(self.lastState)
        -- Find the max index (the chosen action).
        local max, index = torch.max(q, 1)
        action = index[1]
    end

    if (self.train) then
        -- Decay the epsilon by a factor depending on how many frames we want to anneal the epsilon for.
        if (self.epsilon > self.epsilonMinimumValue) then
            self.epsilon = self.epsilon - ((self.initialEpsilonValue - self.epsilonMinimumValue) / self.numberOfFramesToAnnealEpsilon)
        end

        -- We get a batch of training data to train the model.
        local inputs, targets = self.memory:getBatch(self.model,
            self.batchSize,
            self.nbActions,
            self.numberOfFrames,
            self.size)

        -- Train the network which returns the error.
        local err = self:trainNetwork(inputs, targets)
    end
    self.lastAction = action
    return action
end

--[[ Runs one gradient update using SGD returning the loss.]] --
function TorchPongLearning:trainNetwork(inputs, targets)
    local loss = 0
    local x, gradParameters = self.model:getParameters()
    local function feval(x_new)
        gradParameters:zero()
        local predictions = self.model:forward(inputs)
        local loss = self.criterion:forward(predictions, targets)
        local gradOutput = self.criterion:backward(predictions, targets)
        self.model:backward(inputs, gradOutput)
        return loss, gradParameters
    end

    local _, fs = optim.sgd(feval, x, self.sgdParams)
    loss = loss + fs[1]
    return loss
end


function TorchPongLearning:saveNetwork()
    torch.save(self.fileName, self.model)
end
