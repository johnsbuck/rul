----------------------------------------------------------------------
-- This script demonstrates how to define a training procedure,
-- irrespective of the model/loss functions chosen.
--
-- It shows how to:
--   + construct mini-batches on the fly
--   + define a closure to estimate (a noisy) loss
--     function, as well as its derivatives wrt the parameters of the
--     model to be trained
--   + optimize the function, according to several optmization
--     methods: SGD, L-BFGS.
--
-- Clement Farabet
----------------------------------------------------------------------

require 'torch'   -- torch
require 'xlua'    -- xlua provides useful tools, like progress bars
require 'optim'   -- an optimization package, for online and batch methods

----------------------------------------------------------------------
-- parse command line arguments
if not opt then
   print '==> processing options'
   cmd = torch.CmdLine()
   cmd:text()
   cmd:text('SVHN Training/Optimization')
   cmd:text()
   cmd:text('Options:')
   cmd:option('-save', 'results', 'subdirectory to save/log experiments in')
   cmd:option('-visualize', false, 'visualize input data and weights during training')
   cmd:option('-plot', false, 'live plot')
   cmd:option('-optimization', 'SGD', 'optimization method: Adam | SGD | ASGD | CG | LBFGS')
   cmd:option('-learningRate', 1e-3, 'learning rate at t=0')
   cmd:option('-batchSize', 1, 'mini-batch size (1 = pure stochastic)')
   cmd:option('-weightDecay', 0, 'weight decay (SGD only)')
   cmd:option('-momentum', 0, 'momentum (SGD only)')
   cmd:option('-t0', 1, 'start averaging at t0 (ASGD only), in nb of epochs')
   cmd:option('-maxIter', 2, 'maximum nb of iterations for CG and LBFGS')
   cmd:text()
   opt = cmd:parse(arg or {})
end

----------------------------------------------------------------------
-- CUDA?
if opt.type == 'cuda' then
   model:cuda()
   criterion:cuda()
end

----------------------------------------------------------------------
print '==> defining some tools'

local log_file = 'train.log'

-- Log results to files
if opt.mode and opt.mode == 'validate' then
  log_file = 'train_val.log'
end
trainLogger = optim.Logger(paths.concat(opt.save, log_file))
trainMaeLogger = optim.Logger(paths.concat(opt.save, 'mae', log_file))

-- Retrieve parameters and gradients:
-- this extracts and flattens all the trainable parameters of the mode
-- into a 1-dim vector
if model and opt.models ~= 'rnn' then
   parameters,gradParameters = model:getParameters()
end

----------------------------------------------------------------------
print '==> configuring optimizer'

if opt.optimization == 'CG' then
   optimState = {
      maxIter = opt.maxIter
   }
   optimMethod = optim.cg

elseif opt.optimization == 'LBFGS' then
   optimState = {
      learningRate = opt.learningRate,
      maxIter = opt.maxIter,
      nCorrection = 10
   }
   optimMethod = optim.lbfgs

elseif opt.optimization == 'SGD' then
   optimState = {
      learningRate = opt.learningRate,
      weightDecay = opt.weightDecay,
      momentum = opt.momentum,
      learningRateDecay = 1e-7
   }
   optimMethod = optim.sgd

elseif opt.optimization == 'ASGD' then
   optimState = {
      eta0 = opt.learningRate,
      t0 = trsize * opt.t0
   }
   optimMethod = optim.asgd

elseif opt.optimization == 'Adam' then
  optimState = {
    learningRate = opt.learningRate,
    weightDecay = opt.weightDecay,
    momentum = opt.momentum,
    learningRateDecay = 1e-7
  }
  optimMethod = optim.adam

else
   error('unknown optimization method')
end

----------------------------------------------------------------------
print '==> defining training procedure'

function train()
   train_loss = 0
   train_mae_loss = 0
   
   -- epoch tracker
   epoch = epoch or 1

   -- local vars
   local time = sys.clock()

   -- set model to training mode (for modules that differ in training and testing, like Dropout)
   model:training()

   -- shuffle at each epoch
   shuffle = torch.randperm(trsize)

   -- do one epoch
   print('==> doing epoch on training data:')
   print("==> online epoch # " .. epoch .. ' [batchSize = ' .. opt.batchSize .. ']')
   for t = 1,trainData:size(),opt.batchSize do
      -- disp progress
      xlua.progress(t, trainData:size())

      -- create mini batch
      local inputs = {}
      local targets = {}
      for i = t,math.min(t+opt.batchSize-1,trainData:size()) do
         -- load new sample
         local input = trainData.data[shuffle[i]]
         local target = trainData.target[shuffle[i]]
         if opt.type == 'double' then
            input = input:double()
            if opt.loss == 'mse' then
               target = target:double()
            end
         elseif opt.type == 'cuda' then
            input = input:cuda()
            if opt.loss == 'mse' then
               target = target:cuda()
            end
         end
         table.insert(inputs, input)
         table.insert(targets, target)
      end

      -- create closure to evaluate f(X) and df/dX
      local feval = function(x)
                       -- get new parameters
                       if x ~= parameters then
                          parameters:copy(x)
                       end

                      local inputSize = #inputs
                      
                      inputs = torch.cat(inputs)
                      inputs = torch.reshape(inputs, inputSize, trainData.data:size()[2])
                      
                      targets = torch.cat(targets)
                      targets = torch.reshape(targets, inputSize, 1)

                       -- reset gradients
                       gradParameters:zero()

                       -- f is the average of all criterions
                       local f = 0

                       -- evaluate function for complete mini batch
                       local output = model:forward(inputs)

                       local f = criterion:forward(output, targets)

                       -- estimate df/dW
                       local df_do = criterion:backward(output, targets)
                       model:backward(inputs, df_do)
                       
                       train_mae_loss = torch.sum(torch.abs(output - targets)) + train_mae_loss

                       -- normalize gradients and f(X)
                       gradParameters:div(inputSize)
                       
                       inputs = {}
                       targets = {}

                       -- return f and df/dX
                       return f,gradParameters
                    end

      -- optimize on current mini-batch
      if optimMethod == optim.asgd then
         _,fx,average = optimMethod(feval, parameters, optimState)
      else
         _,fx = optimMethod(feval, parameters, optimState)
      end
      train_loss = train_loss + fx[1]
   end

   -- time taken
   time = sys.clock() - time
   time = time / trainData:size()
   print("\n==> time to learn 1 sample = " .. (time*1000) .. 'ms')

   -- report average error on epoch
   train_loss = (train_loss/trainData:size())
   train_mae_loss = (train_mae_loss/trainData:size())
   print('current loss = ' .. train_loss)
   print('current mae = ' .. train_mae_loss)

   -- update logger/plot
   trainLogger:add{['training error'] = train_loss}
   trainMaeLogger:add{['training error'] = train_mae_loss}
   if opt.plot then
      trainLogger:style{['training error'] = '-'}
      trainLogger:plot()
   end

   -- save/log current net
   local model_name = 'model.net'
   if opt.mode and opt.mode == 'validate' then
     model_name = 'model_val.net'
   end
   
   local filename = paths.concat(opt.save, model_name)
   os.execute('mkdir -p ' .. sys.dirname(filename))
   print('==> saving model to '..filename)
   torch.save(filename, model)

   -- next epoch
   epoch = epoch + 1
end
