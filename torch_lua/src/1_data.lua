----------------------------------------------------------------------
-- This script is used for the generating of data to be used for a
-- given model.
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
print '==> choosing dataset files'

---
-- Setting Dataset Files
-- We set our train and test datasets to previously constructed
-- datasets. 
-- 
-- These datasets parses the information given from their 
-- .txt counterpart to know what their expected RUL output is. This
-- is obtain by counting the number of cycles in the training dataset
-- for each unit number and by merging the RUL_*.txt file with the 
-- corresponding test file.
-- @string train_file Location of the training dataset file
-- @string test_file Location of the testing dataset file  
train_file = 'data/train.t7'
test_file = 'data/test.t7'

----------------------------------------------------------------------
-- Setting Training/Testing Sizes
-- We set the percentage of each dataset that we will be using based
-- on the opt.size parameter.
-- @string[opt="full"] opt.size Sets the number of data points we are training and testing.
-- @number trpercent The percentage of the training dataset being used.
-- @number tepercent The percentage of the testing dataset being used.
if opt.size == 'full' then
   print '==> using regular, full training data'
   trpercent = 1
   tepercent = 1
elseif opt.size == 'small' then
   print '==> using reduced training data, for fast experiments'
   trpercent = 0.25
   tepercent = 0.25
end

----------------------------------------------------------------------
print '==> load datasets'

local function loadData(file, percent)
  local loaded = torch.load(file)
  local size = 0
  local info = {}
  if loaded.x ~= nil then
    size = math.floor(loaded.x:size()[1] * percent)
    info = {
      data = loaded.x,
      target = loaded.y,
      size = function() return size end
    }
  else
    size = math.floor(loaded:size()[1] * percent)
    info = {
      data = loaded,
      size = function() return size end
    }
  end
  
  return info
end

print '==> Loading Training Dataset'

trainData = loadData(train_file, trpercent)
trsize = trainData:size()
trainData.data = trainData.data[{{},{2,trainData.data:size()[2]}}]

print '==> Loading Testing Dataset'

testData = loadData(test_file, tepercent)
--testData.target = nil
tesize = testData:size()

units = testData.data[{{},1}]
testData.data = testData.data[{{},{2,testData.data:size()[2]}}]

----------------------------------------------------------------------
print '==> Using a Guassian Kernel'

require 'image'

print '==> Setting the Data Value'

trainData.data = rutil.standardize(trainData.data:float())
maxmultiplier = trainData.target:max()
trainData.target = trainData.target/maxmultiplier
testData.data = rutil.standardize(testData.data:float())
-- testData.target = testData.target/maxmultiplier

testData.data[{{}, {3,25}}] = rutil.scale_features(testData.data[{{}, {3,25}}]):mul(2):add(-1)

local elements = testData.data[{{},1}]
local downward = (testData.data[{{},11}]+testData.data[{{},16}]+testData.data[{{},24}]+testData.data[{{},25}])/4
local upward = (testData.data[{{},6}]+testData.data[{{},7}]+testData.data[{{},8}]+
  testData.data[{{},12}]+testData.data[{{},15}]+testData.data[{{},17}]+testData.data[{{},19}]+testData.data[{{},21}])/8
  
testData.data = torch.Tensor(testData.data:size()[1],3)

testData.data[{{},1}] = rutil.standardize(elements:clone())
testData.data[{{},2}] = rutil.standardize(downward:clone())
testData.data[{{},3}] = rutil.standardize(upward:clone())

local elements = trainData.data[{{},1}]
local downward = (trainData.data[{{},11}]+trainData.data[{{},16}]+trainData.data[{{},24}]+trainData.data[{{},25}])/4
local upward = (trainData.data[{{},6}]+trainData.data[{{},7}]+trainData.data[{{},8}]+
  trainData.data[{{},12}]+trainData.data[{{},15}]+trainData.data[{{},17}]+trainData.data[{{},19}]+trainData.data[{{},21}])/8
  
trainData.data = torch.Tensor(trainData.data:size()[1],3)

trainData.data[{{},1}] = rutil.standardize(elements:clone())
trainData.data[{{},2}] = rutil.standardize(downward:clone())
trainData.data[{{},3}] = rutil.standardize(upward:clone())

--minimum = trainData.target:min()
--maximum = trainData.target:max()
--trainData.target = rutil.scale_features(trainData.target)


----------------------------------------------------------------------
print '==> verify statistics'

-- Verifying data is normalized

----------------------------------------------------------------------
print '==> visualizing data'

-- Visualization is quite easy, using itorch.image().

if opt.visualize then
   if itorch then
   else
      print("For visualization, run this script in an itorch notebook")
   end
end
