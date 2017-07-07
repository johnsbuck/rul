----------------------------------------------------------------------
-- This script implements a test procedure, to report accuracy
-- on the test data. Nothing fancy here...
--
-- Clement Farabet
----------------------------------------------------------------------

require 'torch'   -- torch
require 'xlua'    -- xlua provides useful tools, like progress bars
require 'optim'   -- an optimization package, for online and batch methods

----------------------------------------------------------------------
print '==> defining some tools'

local log_file = 'test.log'

-- Log results to files
if opt.mode and opt.mode == 'validate' then
  log_file = 'test_val.log'
end

testLogger = optim.Logger(paths.concat(opt.save, log_file))
testMaeLogger = optim.Logger(paths.concat(opt.save, 'mae', log_file))

----------------------------------------------------------------------
print '==> defining test procedure'


-- test function
function test()
   test_loss = 0
   test_mae_loss = 0

   local results = {}
   -- local vars
   local time = sys.clock()

   -- averaged param use?
   if average then
      cachedparams = parameters:clone()
      parameters:copy(average)
   end

   -- set model to evaluate mode (for modules that differ in training and testing, like Dropout)
   model:evaluate()

   local err = 0
   -- test over test data
   print('==> testing on test set:')
   for t = 1,testData:size() do
      -- disp progress
      xlua.progress(t, testData:size())

      -- get new sample
      local input = testData.data[t]
      if opt.type == 'double' then input = input:double()
      elseif opt.type == 'cuda' then input = input:cuda() end
      
      local target = torch.Tensor(1)
      target[1] = testData.target[t]
      
      -- test sample
      local pred = model:forward(input)
      
      -- fetch loss
      test_loss = math.abs(criterion:forward(pred,target)) + test_loss
      test_mae_loss = torch.sum(torch.abs(pred - target)) + test_mae_loss
   end

   -- timing
   time = sys.clock() - time
   time = time / testData:size()
   print("==> time to test 1 sample = " .. (time*1000) .. 'ms')

   -- report average error on epoch
   test_loss = test_loss/(testData:size())
   test_mae_loss = test_mae_loss/(testData:size())
   print('current loss = ' .. test_loss)
   print('current mae = ' .. test_mae_loss .. '\n')

   -- update logger/plot
   testLogger:add{['testing error'] = test_loss}
   testMaeLogger:add{['testing error'] = test_mae_loss}
   
   if opt.plot then
      testLogger:style{['testing error'] = '-'}
      testLogger:plot()
   end

   -- averaged param use?
   if average then
      -- restore parameters
      parameters:copy(cachedparams)
   end
end
