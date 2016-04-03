
function initialisePaddle1()
    -- The position of the paddle and its dimensions.
    paddle_1_width = 20
    paddle_1_height = 70
    paddle_1_x = 0
    paddle_1_y = (screen_height / 2) - (paddle_1_height / 2)
    -- How fast it moves.
    paddle_1_speed = 400
    paddle_1_color = { 255, 255, 255 }
end


function updatePaddle1(dt, action)
    if action == 1 then
        paddle_1_y = paddle_1_y - (paddle_1_speed * dt)
    end
    if action == 3 then
        paddle_1_y = paddle_1_y + (paddle_1_speed * dt)
    end

    if paddle_1_y < 0 then
        paddle_1_y = 0
    elseif (paddle_1_y + paddle_1_height) > screen_height then
        paddle_1_y = screen_height - paddle_1_height
    end
end

function drawPaddle1()
    love.graphics.setColor(paddle_1_color)
    love.graphics.rectangle('fill', paddle_1_x, paddle_1_y, paddle_1_width, paddle_1_height)
end