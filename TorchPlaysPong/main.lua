--[[
-- The main.lua used by love to create the window and update the screen. Configures the environment and agents.
]]

require 'nn'
require 'cunn'
require 'xlua'
require 'ball'
require 'paddle_1'
require 'paddle_2'
require 'constants'

require 'TorchPongLearning'


function love.load()
    -- The size that we resize the screen to for the neural network.
    resize = 80
    -- The number of frames we want to train our model on.
    numberOfFramesToPlay = 20000

    -- Create the two models to play pong.
    player1 = TorchPongLearning({ train = true, loadModel = false, fileName = 'player1.model' })
    player2 = TorchPongLearning({ train = true, loadModel = false, fileName = 'player2.model' })

    -- Initialise the rewards for both players.
    reward1 = 0
    reward2 = 0
    isTerminated = false
    numberOfFramesPlayed = 0

    -- Screen width/height loaded from the constants.lua script. Canvas is where we draw to.
    canvas = love.graphics.newCanvas(screen_width, screen_height)

    initialisePaddle1()
    initialisePaddle2()
    initialiseBall()
    love.window.setTitle('Torch plays pong!')
    love.window.setMode(screen_width, screen_height)
end

-- Called before a draw to screen to update the speed/positions and state of the game.
-- (dt is the time since love.update was last called and can be used for making the speed of things consistent.)
function love.update(dt)
    -- Exposes the agents to a set number of frames.
    if (numberOfFramesPlayed < numberOfFramesToPlay) then
        -- Gets a snapshot of the screen. The size will be resize x resize.
        local currentState = blackAndWhiteValues(resize, screen_width, screen_height)
        -- Gets action index for both AIs controlling their paddles.
        local paddle1Action, paddle2Action = getActionFromAI(currentState, reward1, reward2, isTerminated)
        updatePaddle1(dt, paddle1Action)
        updatePaddle2(dt, paddle2Action)
        bounceBallIfItHitsTopOfScreen()
        bounceBallIfItHitsBottomOfScreen()
        -- The agents get rewards for the ball hitting their paddle.
        bounceBallIfItsHitsPaddle1()
        bounceBallIfItsHitsPaddle2()
        local rewardPlayer1, rewardPlayer2, gameOver = resetBallAndGetRewards()
        updateBall(dt)
        -- For the next update, we will pass the rewards/game for their chosen action at this update to the ai.
        reward1 = rewardPlayer1
        reward2 = rewardPlayer2
        isTerminated = gameOver
        numberOfFramesPlayed = numberOfFramesPlayed + 1
        xlua.progress(numberOfFramesPlayed, numberOfFramesToPlay)
    else
        endGame()
        player1:saveNetwork()
        player2:saveNetwork()
        print("Network saved")
    end
end

function getActionFromAI(currentState, reward1, reward2, isTerminated)
    local action1 = player1:getAction(currentState, reward1, isTerminated)
    local action2 = player2:getAction(currentState, reward2, isTerminated)
    return action1, action2
end

function endGame()
    love.event.quit()
end

function blackAndWhiteValues(resize, screen_width, screen_height)
    -- We create a new canvas that is used to resize the current canvas for retrieving pixel data.
    local image = love.graphics.newImage(canvas:newImageData())
    love.graphics.clear()
    local resizedCanvas = love.graphics.newCanvas(resize, resize)
    love.graphics.setCanvas(resizedCanvas)
    love.graphics.clear()
    love.graphics.draw(image, 0, 0, 0, resize / screen_width, resize / screen_height, 0, 0, 0)
    local imageData = resizedCanvas:newImageData()
    local frame = torch.CudaTensor(resize, resize)
    for width = 0, resize - 1 do
        for height = 0, resize - 1 do
            local r = imageData:getPixel(width, height) -- Because it is black and white, we only need 1 channel.
            if (r == 255) then r = 1 end -- Since it is black and white (and objects are white), set features to 1.
            frame[width + 1][height + 1] = r
        end
    end
    return frame
end

function love.draw()
    love.graphics.setCanvas(canvas)
    love.graphics.clear()
    drawPaddle1()
    drawPaddle2()
    drawBall()
    love.graphics.setCanvas()
    love.graphics.draw(canvas)
end