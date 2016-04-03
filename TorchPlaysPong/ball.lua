function initialiseBall()
    -- The position of the ball and its dimensions.
    ball_width = 20
    ball_height = 20
    ball_x = (screen_width / 2) - (ball_width / 2)
    ball_y = (screen_height / 2) - (ball_height / 2)

    ball_color = { 255, 255, 255 }

    -- Ball speed
    ball_speed_x = -400
    ball_speed_y = 400
end


function bounceBallIfItHitsTopOfScreen()
    if ball_y < 0 then
        ball_speed_y = math.abs(ball_speed_y)
    end
end


function bounceBallIfItHitsBottomOfScreen()
    if (ball_y + ball_height) > screen_height then
        ball_speed_y = -math.abs(ball_speed_y)
    end
end

function bounceBallIfItsHitsPaddle1()
    if ball_x <= paddle_1_width and
            (ball_y + ball_height) >= paddle_1_y and
            ball_y < (paddle_1_y + paddle_1_height)
    then
        ball_speed_x = math.abs(ball_speed_x)
        reward = 1
    end
end

function bounceBallIfItsHitsPaddle2()
    if (ball_x + ball_width) >= (screen_width - paddle_2_width) and
            (ball_y + ball_height) >= paddle_2_y and
            ball_y < (paddle_2_y + paddle_2_height)
    then
        ball_speed_x = -math.abs(ball_speed_x)
        reward = 1
    end
end

-- resets ball if it goes off the screen, as well as returns if the game terminated/rewards for players.
function resetBallAndGetRewards()
    -- Resets the ball once it is out the screen.
    local function reset()
        ball_x = (screen_width / 2) - (ball_width / 2)
        ball_y = (screen_height / 2) - (ball_height / 2)
        if (math.random() < 0.5) then
            ball_speed_x = -400
        else
            ball_speed_x = 400
        end
        ball_speed_y = 400
    end

    local player1Reward = 0
    local player2Reward = 0
    local isGameOver = false
    -- Player 1 let the ball out of bounds.
    if ball_x + ball_width < 0 then
        player1Reward = -2
        player2Reward = 2
        isGameOver = true
        reset()
        -- Player 2 let the ball out of bounds.
    elseif (ball_x > screen_width) then
        player1Reward = 2
        player2Reward = -2
        isGameOver = true
        reset()
    end
    return player1Reward, player2Reward, isGameOver
end

function updateBall(dt)
    ball_x = ball_x + (ball_speed_x * dt)
    ball_y = ball_y + (ball_speed_y * dt)
end

function drawBall()
    love.graphics.setColor(ball_color)
    love.graphics.rectangle('fill', ball_x, ball_y, ball_width, ball_height)
end
