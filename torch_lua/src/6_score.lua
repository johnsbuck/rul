----------------------------------------------------------------------
-- This script is used for scoring datasets after obtaining our model
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

function rul_score(estimate, actual)
  local difference = estimate - actual
  
  if difference < 0 then
    return math.exp(-difference/13) - 1
  end
  
  return math.exp(difference/10) - 1
end

CMAPSS_1 = torch.Tensor(rutil.csvToTable('data/CMAPSS_1_TEST.csv', ' ', false))
CMAPSS_1 = CMAPSS_1[{{}, {initial_index, CMAPSS_1:size()[2]}}]
CMAPSS_2 = torch.Tensor(rutil.csvToTable('data/CMAPSS_2_TEST.csv', ' ', false))
CMAPSS_2 = CMAPSS_2[{{}, {initial_index, CMAPSS_2:size()[2]}}]
CMAPSS_3 = torch.Tensor(rutil.csvToTable('data/CMAPSS_3_TEST.csv', ' ', false))
CMAPSS_3 = CMAPSS_3[{{}, {initial_index, CMAPSS_3:size()[2]}}]
CMAPSS_4 = torch.Tensor(rutil.csvToTable('data/CMAPSS_4_TEST.csv', ' ', false))
CMAPSS_4 = CMAPSS_4[{{}, {initial_index, CMAPSS_4:size()[2]}}]

CMAPSS_1_ANS = torch.Tensor(rutil.csvToTable('/home/jsb/Git/thesis/data/C-MAPSS/RUL_FD001.txt', ' ', false))
CMAPSS_2_ANS = torch.Tensor(rutil.csvToTable('/home/jsb/Git/thesis/data/C-MAPSS/RUL_FD002.txt', ' ', false))
CMAPSS_3_ANS = torch.Tensor(rutil.csvToTable('/home/jsb/Git/thesis/data/C-MAPSS/RUL_FD003.txt', ' ', false))
CMAPSS_4_ANS = torch.Tensor(rutil.csvToTable('/home/jsb/Git/thesis/data/C-MAPSS/RUL_FD004.txt', ' ', false))

CMAPSS_1_UNITS = rutil.csvToTable('data/CMAPSS_1_TEST_units.csv', ' ', false)
CMAPSS_2_UNITS = rutil.csvToTable('data/CMAPSS_2_TEST_units.csv', ' ', false)
CMAPSS_3_UNITS = rutil.csvToTable('data/CMAPSS_3_TEST_units.csv', ' ', false)
CMAPSS_4_UNITS = rutil.csvToTable('data/CMAPSS_4_TEST_units.csv', ' ', false)

results = {}
for i=1,CMAPSS_1:size()[1] do
  results[CMAPSS_1_UNITS[i][1]] = model:forward(CMAPSS_1[i]:double()) * maxmultiplier
end

cmapss_1_score = 0
for i=1, #results do
  cmapss_1_score = cmapss_1_score + rul_score(results[i][1], CMAPSS_1_ANS[i][1])
end

results = {}
for i=1,CMAPSS_2:size()[1] do
  results[CMAPSS_2_UNITS[i][1]] = model:forward(CMAPSS_2[i]:double()) * maxmultiplier
end

cmapss_2_score = 0
for i=1, #results do
  cmapss_2_score = cmapss_2_score + rul_score(results[i][1], CMAPSS_2_ANS[i][1])
end

results = {}
for i=1,CMAPSS_3:size()[1] do
  results[CMAPSS_3_UNITS[i][1]] = model:forward(CMAPSS_3[i]:double()) * maxmultiplier
end

cmapss_3_score = 0
for i=1, #results do
  cmapss_3_score = cmapss_3_score + rul_score(results[i][1], CMAPSS_3_ANS[i][1])
end

results = {}
for i=1,CMAPSS_4:size()[1] do
  results[CMAPSS_4_UNITS[i][1]] = model:forward(CMAPSS_4[i]:double()) * maxmultiplier
end

cmapss_4_score = 0
for i=1, #results do
  cmapss_4_score = cmapss_4_score + rul_score(results[i][1], CMAPSS_4_ANS[i][1])
end
