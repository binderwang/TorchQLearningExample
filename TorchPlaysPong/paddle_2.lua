function initialisePaddle2()
    paddle_2_width = 20
    paddle_2_height = 70
    paddle_2_x = screen_width - paddle_2_width
    paddle_2_y = (screen_height / 2) - (paddle_1_height / 2)
    paddle_2_speed = 400
    paddle_2_color = { 255, 255, 255 }
end



function updatePaddle2(dt, action)
    if action == 1 then
        paddle_2_y = paddle_2_y - (paddle_2_speed * dt)
    end
    if action == 3 then
        paddle_2_y = paddle_2_y + (paddle_2_speed * dt)
    end

    if paddle_2_y < 0 then
        paddle_2_y = 0
    elseif (paddle_2_y + paddle_2_height) > screen_height then
        paddle_2_y = screen_height - paddle_2_height
    end
end

function drawPaddle2()
    love.graphics.setColor(paddle_2_color)
    love.graphics.rectangle('fill', paddle_2_x, paddle_2_y, paddle_2_width, paddle_2_height)
end