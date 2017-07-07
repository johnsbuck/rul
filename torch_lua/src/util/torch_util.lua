----------------------------------------------------------------------
--  A utility module containing Torch-based functions for editing
--  tensors, normalization, and other purposes.
--  @author John Bucknam
--  @license MIT
--  @module Torch-Utilities
----------------------------------------------------------------------

require 'torch'

----------------------------------------------------------------------
-- Matrix Utilities
-- @section matrix-utilities
----------------------------------------------------------------------

---
-- Measures the covariance between 2 tensors with 1-D data vectors.
-- @tensor x
-- @tensor y
-- @treturn number The covariance of x and y
function rutil.cov(x, y)
  local len = x:size()[1]
  
  local covariance = 0
  
  for i=1, len do
    local a = x[i] - x:mean()
    local b = y[i] - y:mean()
    covariance = covariance + (a*b)/len
  end
  
  return covariance
end

---
-- Creates a covariance matrix for a given tensor.
-- @tensor X
-- @treturn tensor A 2-D matrix with matrix[i][j] corresponding to cov(i,j)
function rutil.cov_matrix(X)
  local len = X:size()[2]
  local matrix = torch.Tensor(len,len)
  
  for i=1, len do
    print("I: ", i)
    for j=1, len do
      print("J: ", j)
      matrix[i][j] = rutil.cov(X[{{},i}], X[{{},j}])
    end
  end
  
  return matrix
end

---
-- Measures the correlation between 2 tensors with 1-D data vectors.
-- @tensor x
-- @tensor y
-- @treturn number The correlation of x and y
function rutil.corr(x,y)
  local correlation = 0
  return rutil.cov(x,y)/(x:std()*y:std())
end

----------------------------------------------------------------------
-- Normalization
-- @section normalization
----------------------------------------------------------------------

---
-- Subtracts each point of a given
-- tensor by the minimum and divides by
-- the difference between the maximum
-- and minimum.
-- 
-- Feature Scaling Formula: (x - Low_X)/(High_X - Low_X)
-- 
-- @tensor data Vectors in given Tensor
-- @treturn tensor Scaled tensor based on minimum and maximum
function rutil.scale_features(data)
  local info = data:clone()
  
  -- Feature Scaling
  if #info:size() > 1 then
    for i=1, info:size()[2] do
      local min = info[{{},i}]:min()
      local max = info[{{},i}]:max()
      if max ~= min then
        info[{{},i}]:add(-min)
        info[{{},i}]:div(max-min)
      else
        info[{{},i}]:add(-min+1)
      end
    end
  else
      local min = info:min()
      local max = info:max()
      if max ~= min then
        info:add(-min)
        info:div(max-min)
      else
        info[{{},i}]:add(-min+1)
      end
  end
  
  return info
end

---
-- Takes an existing tensor and subtracts each value by its column's
-- mean. Afterwards, it divides itself by its standard deviation.
-- 
-- Standardization Formula: (x - MEAN) / STD
-- 
-- @tensor data A Torch tensor containing data to be normalized.
-- @treturn tensor A normalized Torch tensor of the same type as data.
function rutil.standardize(data)
  local info = data:clone()
  
  -- Standardization
  if #info:size() > 1 then
    MEAN = {}
    STD = {}
    for i=1, info:size()[#info:size()] do
      MEAN[i] = info[{{},i}]:mean()
      STD[i] = info[{{},i}]:std()
      info[{{},i}]:add(-info[{{},i}]:mean())
      info[{{},i}]:div(info[{{},i}]:std())
    end
  else
    MEAN = info:mean()
    STD = info:std()
    info:add(-info:mean())
    info:div(info:std())
  end

  return info
end

---
-- Takes a standardized Tensor and multiples it by the STD.
-- It then adds the MEAN to each data point.
-- 
-- Destandardize Formula: (x * prevSTD) + prevMEAN
-- 
-- @tensor data A Torch Tensor containing data vectors.
-- @treturn tensor A destandardized tensor.
function rutil.destandardize(data)
  local info = data:clone()
  
  -- Standardization
  if #info:size() > 1 then
    for i=1, info:size()[#info:size()] do
      info[{{},i}]:mul(STD[i])
      info[{{},i}]:add(MEAN[i])
    end
  else
    info:mul(STD)
    info:add(MEAN)
  end
  
  return info
end

----------------------------------------------------------------------
-- Normalization based on Previous Data Vector
-- @section normalization-prev-data
----------------------------------------------------------------------

---
-- Returns all data points as the log of themselves 
-- minus the log of their previous data point.
-- 
-- Subtractive Log Formula: ln(close_price[i]) - ln(close_price[i-1])
-- 
-- @tensor data A Torch Tensor containing data vectors.
-- @treturn tensor A normalized Torch tensor of the same type as data.
function rutil.log_prev(data)
  local info = data:clone()
  
  --Previous log minus
  info = info:log()
  for i=1, info:size()[1] - 1 do
    info[i] = info[i] - info[i+1]
  end
    
  return info
end

---
-- Returns a Tensor with each point being subtracted
-- by the previous point and divided by the previous point.
-- 
-- Divisive Percent Formula: ((X[i] - X[i-1]) * 100)/X[i-1] 
-- 
-- @tensor data A Torch Tensor containing data vectors.
-- @treturn tensor A normalized Torch tensor of the same type as data.
function rutil.percent_prev(data)
  local info = data:clone()
  
  --Previous percent
  for i=1, info:size()[1] - 1 do
    info[i] = (info[i] - info[i+1])/info[i+1]
  end
  
  return info
end
