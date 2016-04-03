-- To run this code you must use iTorch. Also you can use the .ipynb file in itorch notebook mode.
-- NOTE, you have to comment out the call to Main() at the end of the TorchQLearningExample.lua file.

require 'TorchQLearningExample'
Plot = require 'itorch.Plot'
require 'nn'

local model = torch.load("TorchQLearningModel.model")
local gridSize = 10
local maxGames = 100
local env = CatchEnvironment(gridSize)

local wins = 0
local losses = 0

ground = torch.Tensor({1})
plot = Plot()

function drawState(fruitRow, fruitColumn, basket)
    basket = torch.Tensor({basket})
    local fruitX = torch.Tensor({fruitColumn}) -- column is the x axis.
    local fruitY = torch.Tensor({gridSize - fruitRow + 1}) --Invert matrix style points to coordinates.
    plot['_data'] = {} -- Empty the current plot.
    plot:quad(ground - 1, ground, ground + 10, ground + 10, 'black', '')
    plot:quad(basket - 1, ground, basket + 1, ground + 0.5, 'red', '')
    plot:quad(fruitX - 0.5, fruitY - 0.5, fruitX + 0.5, fruitY + 0.5, 'red', '')
    local title = "Wins: " .. wins .. " Torch Plays Catch Losses: " .. losses
    plot:title(title)
    plot = plot:redraw()
end

function sleep(n)
    os.execute("sleep " .. tonumber(n))
end


local numberOfGames = 0
while (numberOfGames < maxGames) do
    -- The initial state of the environment.
    local isGameOver = false
    local fruitRow, fruitColumn, basket = env.reset()
    local currentState = env.observe()
    drawState(fruitRow, fruitColumn, basket)

    while (isGameOver ~= true) do
        -- Forward the current state through the network.
        local q = model:forward(currentState)
        -- Find the max index (the chosen action).
        local max, index = torch.max(q, 1)
        local action = index[1]
        local nextState, reward, gameOver, fruitRow, fruitColumn, basket = env.act(action)
        if(reward == 1) then wins = wins + 1 elseif(reward == -1) then losses = losses +1 end
        currentState = nextState
        isGameOver = gameOver
        drawState(fruitRow, fruitColumn, basket)
        sleep(0.2)
    end
end