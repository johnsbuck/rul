----------------------------------------------------------------------
-- This tutorial shows how to train different models on the street
-- view house number dataset (SVHN),
-- using multiple optimization techniques (SGD, ASGD, CG), and
-- multiple types of models.
--
-- This script demonstrates a classical example of training 
-- well-known models (convnet, MLP, logistic regression)
-- on a 10-class classification problem. 
--
-- It illustrates several points:
-- 1/ description of the model
-- 2/ choice of a loss function (criterion) to minimize
-- 3/ creation of a dataset as a simple Lua table
-- 4/ description of training and test procedures
--
-- Clement Farabet
----------------------------------------------------------------------
require 'torch'

----------------------------------------------------------------------
print '==> processing options'

cmd = torch.CmdLine()
cmd:text()
cmd:text('SVHN Loss Function')
cmd:text()
cmd:text('Options:')
-- global:
cmd:option('-seed', 1, 'fixed input seed for repeatable experiments')
cmd:option('-threads', 2, 'number of threads')
-- data:
cmd:option('-mode', 'full', 'how many samples do we load: score | full')
-- model:
cmd:option('-model', 'mlp', 'type of model to construct: linear | mlp | load')
-- loss:
cmd:option('-loss', 'mse', 'tytxtpe of loss function to minimize: nll | mse | margin')
-- training:
cmd:option('-save', 'results', 'subdirectory to save/log experiments in')
cmd:option('-plot', false, 'live plot')
cmd:option('-optimization', 'SGD', 'optimization method: Adam | SGD | ASGD | CG | LBFGS')
cmd:option('-learningRate', 1e-3, 'learning rate at t=0')
cmd:option('-batchSize', 1, 'mini-batch size (1 = pure stochastic)')
cmd:option('-weightDecay', 0, 'weight decay (Adam & SGD only)')
cmd:option('-momentum', 0, 'momentum (SGD only)')
cmd:option('-t0', 1, 'start averaging at t0 (ASGD only), in nb of epochs')
cmd:option('-maxIter', 2, 'maximum nb of iterations for CG and LBFGS')
cmd:option('-type', 'double', 'type: double | float | cuda')
cmd:option('-epoch', math.huge, 'The number of epochs (default: inf)')
cmd:text()
opt = cmd:parse(arg or {})

-- nb of threads and fixed seed (for repeatable experiments)
if opt.type == 'float' then
   print('==> switching to floats')
   torch.setdefaulttensortype('torch.FloatTensor')
elseif opt.type == 'cuda' then
   print('==> switching to CUDA')
   require 'cunn'
   torch.setdefaulttensortype('torch.FloatTensor')
end
torch.setnumthreads(opt.threads)
torch.manualSeed(opt.seed)

----------------------------------------------------------------------
-- Checking if results exists from previous run. If so, close program

local f=io.open(opt.save .. "/results.csv","r")
if f ~= nil then 
  io.close(f)
  error("Folder shouldn't exist")
end

----------------------------------------------------------------------
print '==> executing all'

dofile 'src/1_fetch_data.lua'
dofile 'src/2_model.lua'
dofile 'src/3_loss.lua'
dofile 'src/4_train.lua'
dofile 'src/5_test.lua'

----------------------------------------------------------------------
print '==> training!'

local numEpochs = opt.epoch

while numEpochs > 0 do
  train()
  if opt.mode == 'validate' then
    test()
  end
  
  numEpochs = numEpochs - 1;
end

----------------------------------------------------------------------
if opt.mode ~= 'validate' then
  print '==> Printing Results'
  
  results = {}
  for i=1,testData:size() do
    results[units[i]] = model:forward(testData.data[i]:double()) * maxmultiplier
  end
  
  file = io.open(opt.save .. "/results.csv", "w+")
  
  for i=1, #results do
    file:write(results[i][1] .. "\n")
  end
  
  file:close()
end

----------------------------------------------------------------------
print '==> Saving Details to CSV'

model_details = ""
for i=1, #model.modules do
  model_details = model_details .. tostring(model.modules[i])
  if i < #model.modules then
    model_details = model_details .. ' X '
  end
end

opt_details = ""
for key, value in pairs(opt) do
  opt_details = opt_details .. tostring(value) .. ","
end

file = io.open("model_results.csv", "a+")

if opt.mode ~= 'validate' then
  dofile 'src/6_score.lua'

  file:write(opt_details .. model_details .. "," .. train_loss .. "," .. train_mae_loss ..
      "," .. cmapss_1_score .. "," .. cmapss_2_score .. "," .. cmapss_3_score .. "," .. cmapss_4_score .. "\n")
else
  file:write(opt_details .. model_details .. "," .. train_loss .. "," .. train_mae_loss .. 
      "," .. test_loss .. "," .. test_mae_loss .. "\n")
end

file:close()
