require 'torch'
do
    local Memory = torch.class('Memory')

    function Memory:__init(params)
        self.maxMemory = params.maxMemory
        self.discount = params.discount
        self.buffer = {}
    end

    --[[ The memory: Handles the internal memory that we add experiences that occur based on agent's actions,
    --   and creates batches of experiences based on the mini-batch size for training.]] --

    -- Appends the experience to the memory.
    function Memory:remember(memoryInput)
        table.insert(self.buffer, memoryInput)
        if (#self.buffer > self.maxMemory) then
            -- Remove the earliest memory to allocate new experience to memory.
            table.remove(self.buffer, 1)
        end
    end

    function Memory:getBatch(model, batchSize, nbActions, numberOfFrames, sizeOfFrame)

        -- We check to see if we have enough memory inputs to make an entire batch, if not we create the biggest
        -- batch we can (at the beginning of training we will not have enough experience to fill a batch).
        local memoryLength = #self.buffer
        local chosenBatchSize = math.min(batchSize, memoryLength)
        local inputs = torch.CudaTensor(chosenBatchSize, numberOfFrames, sizeOfFrame, sizeOfFrame):zero()
        local targets = torch.CudaTensor(chosenBatchSize, nbActions):zero()
        --Fill the inputs and targets up.
        for i = 1, chosenBatchSize do
            -- Choose a random memory experience to add to the batch.
            local randomIndex = math.random(1, memoryLength)
            local memoryInput = self.buffer[randomIndex]
            local target = model:forward(memoryInput.inputState)

            --Gives us Q_sa, the max q for the next state.
            local nextStateMaxQ = torch.max(model:forward(memoryInput.nextState))
            if (memoryInput.gameOver) then
                target[1][memoryInput.action] = memoryInput.reward
            else
                -- reward + discount(gamma) * max_a' Q(s',a')
                -- We are setting the Q-value for the action to  r + γmax a’ Q(s’, a’). The rest stay the same
                -- to give an error of 0 for those outputs.
                target[1][memoryInput.action] = memoryInput.reward + self.discount * nextStateMaxQ
            end
            -- Update the inputs and targets.
            inputs[i] = memoryInput.inputState
            targets[i] = target
        end
        return inputs, targets
    end
end