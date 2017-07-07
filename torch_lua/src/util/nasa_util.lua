require 'rutil'

if rutil == nil then
  dofile 'rutil.lua'
  if rutil == nil then
    error('NO RUTIL PACKAGE AVAILABLE')
  end
end

function createNasaTrainSet(file)
  local data = csvToTable(file, ' ', false)
  for i=1, #data do
    data[i][27] = nil
  end
  data = torch.Tensor(data)
  
  local info = {}
  info.x = data[{3,26}]
  info.y = data[{2}]
  
  local units = {}
  
  for i=1, data:size()[1] do
    units[data[i][1]] = data[i][2]
  end
  
  for i=1, info.y:size()[1] do
    info.y[i][1] = units[data[i][1]] - data[i][2]
  end
  
  print("==> INFO: Saving '" .. file .. "' as '" .. file:match("(.+)%.") .. ".t7' for future use.")
  torch.save(file:match("(.+)%.") .. '.t7', info)
end

function createNasaTestSet(file, rulFile)
  local data = csvToTable(file, ' ', false)
  for i=1, #data do
    data[i][27] = nil
  end
  data = torch.Tensor(data)
  
  info = {}
  info.x = data[{3,26}]
  info.y = data[{2}]
  
  local rul = csvToTable(rulFile, ' ', false)
  
  
end