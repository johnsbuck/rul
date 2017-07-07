----------------------------------------------------------------------
--  A utility module containing generic functions
--  for multiple purposes.
--  @author John Bucknam
--  @license MIT
--  @module Basic-Utilities
----------------------------------------------------------------------

----------------------------------------------------------------------
-- Sorting
-- @section sorts
----------------------------------------------------------------------

---
-- Sorts a given table by the elements of the table, or a tree element
-- if such a tree is given.
-- @tab tab A table of comparable elements.
-- @tab tree A tree that branches to comparing element for a table of tables (Ex. {81} -> tab[n][81])
function rutil.merge_sort(tab,tree)
  local function get_item(A, tree)
    local item = A
    for i=1, #tree do
      item = item[tree[i]]
    end
    return item
  end
  
  local function set_item(A, value, tree, overwrite)
    local item = A
    for i=1, #tree-1 do
      if type(item[tree[i]]) == 'nil' then
        item[tree[i]] = {}
      elseif type(item[tree[i]]) ~= 'table' and (not overwrite) then
        error('ERROR: Overwriting non-table values')
      else
        item[tree[i]] = {}
      end
      item = item[tree[i]]
    end
    item[tree[#tree]] = value
  end

  local function merge(A,p,q,r,tree)
    local n1 = q-p+1
    local n2 = r-q
  
    local left = {}
    local right = {}
  
    for i=1, n1 do
      left[i] = A[p+i-1]
    end
  
    for i=1, n2 do
      right[i] = A[q+i]
    end
  
    if tree == nil then
      left[n1+1] = math.huge
      right[n2+1] = math.huge
    else
      left[n1+1] = {}
      right[n2+1] = {}
      set_item(left[n1+1],math.huge,tree)
      set_item(right[n2+1],math.huge,tree)
    end
  
    local i = 1
    local j = 1
  
    if tree == nil then
      for k=p, r do
        if left[i] <= right[j] then
          A[k] = left[i]
          i = i + 1
        else
          A[k] = right[j]
          j = j + 1
        end
      end
    else
      for k=p, r do
        if get_item(left[i],tree) <= get_item(right[j], tree) then
          A[k] = left[i]
          i = i + 1
        else
          A[k] = right[j]
          j = j + 1
        end
      end
    end
  end
  
  local function merge_sort_len(A,p,r,tree)
    if p < r then
      local q = math.floor((p+r)/2)
        merge_sort_len(A,p,q,tree)
        merge_sort_len(A,q+1,r,tree)
        merge(A,p,q,r,tree)
     end
  end
  
  merge_sort_len(tab,1,#tab,tree)
end

----------------------------------------------------------------------
-- Conversion
-- @section conversion
----------------------------------------------------------------------

---
--  Takes the filename of a csv and saves it as a table.
--  If there is no header, then the header parameter should be false.
--  If header is true, will return table with ["header"] and ["data"].
--  @string  csv  Filename of csv to be open and read.
--  @string[opt=comma] delim The delimeter used to split data vlaues.
--  @bool[opt=true]  header  Splits header from data.
--  @treturn table A table of formatted values
function rutil.csvToTable(csv, delim, header)
  -- If header is nil then set to true,
  -- or it is false then leave it false
  header = (header == nil) or header
  delim = delim or ','
  
  --Open file
  local file = io.open(csv, 'r')

  -- If no file exists
  if not file then
    error('invalid file')
    return nil
  end

  -- Create data table
  local info = {}

  -- Split lines in files into data entries
  if header then
    local i = 0

    for line in file:lines() do
      if i == 0 then
        info.header = line:split(delim)
      else
        info[i] = rutil.tableToNumbers(line:split(delim))
      end
      i = i + 1
    end
  else
    local i = 1
    for line in file:lines() do
      info[i] = rutil.tableToNumbers(line:split(delim))
      i = i + 1
    end
  end

  -- Close file and return table
  file:close()
  file = nil
  return info
end

---
-- Takes an existing table and attempts to
-- convert each value into a number.
-- Will recursively attempt to convert all tables within a table.
-- @tab tab Table that is converted into numbers, if possible.
-- @treturn table A table with all possible number conversions done.
function rutil.tableToNumbers(tab)
  for key, value in pairs(tab) do
    if type(tab[key]) == 'table' then
      tab[key] = rutil.tableToNumbers(tab[key])
    else
      tab[key] = tonumber(tab[key]) or tab[key]
    end
  end
  return tab
end

---
--  Takes an existing table and saves it as the filename.
--  By default the filename is 'data/output.csv'.
--  Requires the header to be in tab.header or tab[1]
--  or no header exists.
--  @tab tab Table that is converted to a csv.
--  @string[opt='data/output.csv']  filename  Name of output file.
function rutil.tableToCsv(tab, filename)
  filename = filename or 'data/output.csv'

  -- Create file in data/output.csv
  local file = io.open(filename, 'w+')

  if tab.header then  -- Header stored separate from data

    -- Write Headers
    for key,value in pairs(tab.header) do
      if key == 1 then
        file:write(value)
      else
        file:write(',' .. value)
      end
    end

    file:write('\n')

    -- Write Data
    for i=1, #tab do
      for key,value in pairs(tab[i]) do
        if key == 1 then
          file:write(tab.date[i] .. ',' .. value)
        else
          file:write(',' .. value)
        end
      end
      file:write('\n')
    end
  else  -- No tab.header (Header is stored with data in tab or no header)
    for i=1, #tab do
      -- Write all data points from single table
      for key,value in pairs(tab[i]) do
        if key == 1 then
          file:write(value)
        else
          file:write(',' .. value)
        end
      end
      file:write('\n')
    end
  end

  -- Close file
  file:close()
  file = nil
end

----------------------------------------------------------------------
-- Copying
-- @section copy
----------------------------------------------------------------------

---
-- Copies a given value and returns it. Will return all table data as well.
-- Obtained from http://lua-users.org/wiki/CopyTable
-- @param orig A given lua parameter to be copied
-- @return The copied value.
function rutil.deepcopy(orig)
    local orig_type = type(orig)
    local copy
    if orig_type == 'table' then
        copy = {}
        for orig_key, orig_value in next, orig, nil do
            copy[rutil.deepcopy(orig_key)] = rutil.deepcopy(orig_value)
        end
        setmetatable(copy, rutil.deepcopy(getmetatable(orig)))
    else -- number, string, boolean, etc
        copy = orig
    end
    return copy
end
