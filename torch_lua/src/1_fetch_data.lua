----------------------------------------------------------------------
-- This script is used to fetch csv files processed through our
-- Python scripts.
-- @author John Bucknam
-- @license MIT
-- @script Data
----------------------------------------------------------------------

require 'torch'   -- torch
require 'rutil'

if rutil == nil then
  dofile 'src/rutil.lua'
  if rutil == nil then
    error('NO RUTIL PACKAGE AVAILABLE')
  end
end

----------------------------------------------------------------------
-- parse command line arguments
if not opt then
   print '==> processing options'
   cmd = torch.CmdLine()
   cmd:text()
   cmd:text('SVHN Dataset Preprocessing')
   cmd:text()
   cmd:text('Options:')
   cmd:option('-size', 'full', 'how many samples do we load: small | full')
   cmd:option('-visualize', true, 'visualize input data and weights during training')
   cmd:text()
   opt = cmd:parse(arg or {})
end

----------------------------------------------------------------------
print '==> load datasets'

initial_index = 2 -- 3: Heimes; 2: w/ Cycles

if opt.mode == 'validate' then
  trainData = {}
  trainData.data = torch.Tensor(rutil.csvToTable('data/train_val.csv', ' ', false))
  trainData.target = torch.Tensor(rutil.csvToTable('data/train_val_target.csv', ' ', false))
  trainData.size = function() return trainData.data:size()[1] end
  
  testData = {}
  testData.data = torch.Tensor(rutil.csvToTable('data/test_val.csv', ' ', false))
  testData.target = torch.Tensor(rutil.csvToTable('data/test_val_target.csv', ' ', false))
  testData.size = function() return testData.data:size()[1] end
  
  trainData.data = trainData.data[{{}, {initial_index, trainData.data:size()[2]}}]
  testData.data = testData.data[{{}, {initial_index, testData.data:size()[2]}}]
  
  maxmultiplier = rutil.csvToTable('data/val_maxmultiplier.csv', ' ', false)[1][1]
else
  trainData = {}
  trainData.data = torch.Tensor(rutil.csvToTable('data/train.csv', ' ', false))
  trainData.target = torch.Tensor(rutil.csvToTable('data/train_target.csv', ' ', false))
  trainData.size = function() return trainData.data:size()[1] end
  
  testData = {}
  testData.data = torch.Tensor(rutil.csvToTable('data/test.csv', ' ', false))
  testData.size = function() return testData.data:size()[1] end
  
  trainData.data = trainData.data[{{}, {initial_index, trainData.data:size()[2]}}]
  testData.data = testData.data[{{}, {initial_index, testData.data:size()[2]}}]
  
  if opt.mode == 'score' then
    testData.target = torch.Tensor(rutil.csvToTable('data/test_target.csv', ' ', false))
  end
  
  units = rutil.csvToTable('data/units.csv', ' ', false)
  
  for i=1, #units do
    units[i] = units[i][1]
  end
  
  maxmultiplier = rutil.csvToTable('data/maxmultiplier.csv', ' ', false)[1][1]
end

trsize = trainData.data:size()[1]
tesize = testData.data:size()[1]
