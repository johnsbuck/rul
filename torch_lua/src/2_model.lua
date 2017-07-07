----------------------------------------------------------------------
-- This script demonstrates how to define a couple of different
-- models:
--   + linear (FNN)
--   + 2-layer neural network (MLP)
--   + recurrent neural network (RNN)
--   + neural network ensemble
--
-- It's a good idea to run this script with the interactive mode:
-- $ th -i 2_model.lua
-- this will give you a Torch interpreter at the end, that you
-- can use to play with the model.
--
-- Clement Farabet
----------------------------------------------------------------------

require 'torch'   -- torch
require 'image'   -- for image transforms
require 'nn'      -- provides all sorts of trainable modules/layers

----------------------------------------------------------------------
-- parse command line arguments
if not opt then
   print '==> processing options'
   cmd = torch.CmdLine()
   cmd:text()
   cmd:text('SVHN Model Definition')
   cmd:text()
   cmd:text('Options:')
   cmd:option('-model', 'mlp', 'type of model to construct: linear | mlp | load')
   cmd:option('-visualize', true, 'visualize input data and weights during training')
   cmd:text()
   opt = cmd:parse(arg or {})
end

----------------------------------------------------------------------
print '==> define parameters'

-- regression
noutputs = 1

-- input dimensions
ninputs = trainData.data:size()[2]

-- number of hidden units (for MLP only):
nhiddens = 13
----------------------------------------------------------------------
print '==> construct model'

if opt.model == 'load' then
   model = torch.load('model.net')
elseif opt.model == 'linear' then

   -- Simple linear model
   model = nn.Sequential()
   model:add(nn.Reshape(ninputs))
   model:add(nn.Linear(ninputs,noutputs))

elseif opt.model == 'mlp' then

   -- Simple 2-layer neural network, with tanh hidden units
   model = nn.Sequential()
   model:add(nn.Reshape(ninputs))
   model:add(nn.Linear(ninputs,13))
   model:add(nn.ReLU())
   -- model:add(nn.Linear(13,13))
   -- model:add(nn.ReLU())
   model:add(nn.Linear(13,noutputs))
   model:add(nn.ReLU())

elseif opt.model == 'rnn' then
  
  require 'rnn'
  
  rho = 10 -- Sequence Length
  
  local r = nn.Recurrent(
    nhiddens, nn.LookupTable(ninputs, nhiddens),
    nn.Linear(nhiddens, nhiddens), nn.Tanh(),
    rho)
  
  model = nn.Sequential()
    :add(nn.Recurrence(r, nhiddens, 0))
    :add(nn.Linear(nhiddens, noutputs))
    :add(nn.Tanh())
  
  model = nn.Recursor(rnn, rho)
  
else

   error('unknown -model')
   
end

----------------------------------------------------------------------
print '==> here is the model:'
print(model)

----------------------------------------------------------------------
-- Visualization is quite easy, using itorch.image().

if opt.visualize then
  if itorch then
  else
    print '==> To visualize filters, start the script in itorch notebook'
  end
end
