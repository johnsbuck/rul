require 'torch'
require 'gnuplot'
require 'itorch'

local unit = 100
local size1 = 0
local size2 = 0
local style = '+'

for i=1, unit do
  if i > 1 then
    size1 = size2
  end
  size2 = size2 + trainData.target[i+size2][1]
end

dofile '1_data.lua'

size1 = size1 + unit
size2 = size2 + unit

gnuplot.xlabel('Element Number')
gnuplot.ylabel('Value')
  
--- Upwards

gnuplot.pngfigure('figures/upwards_1_fd001.png')
gnuplot.title('Upwards Points (6,7,8,12,15,17,19,21)')
gnuplot.plot({trainData.data[{{size1,size2},6}],style},{trainData.data[{{size1,size2},7}],style},
{trainData.data[{{size1,size2},8}],style},{trainData.data[{{size1,size2},12}],style},{trainData.data[{{size1,size2},15}],style},
{trainData.data[{{size1,size2},17}],style},{trainData.data[{{size1,size2},19}],style},{trainData.data[{{size1,size2},21}],style})

gnuplot.plot({(trainData.data[{{size1,size2},6}]+trainData.data[{{size1,size2},7}]+trainData.data[{{size1,size2},8}]+
  trainData.data[{{size1,size2},12}]+trainData.data[{{size1,size2},15}]+trainData.data[{{size1,size2},17}]+trainData.data[{{size1,size2},19}]+trainData.data[{{size1,size2},21}])/8,style})

gnuplot.xlabel('Element Number')
gnuplot.ylabel('Value')
gnuplot.plotflush()

--- No pattern(?)

gnuplot.pngfigure('figures/patternless_1_fd001.png')
gnuplot.title('No Pattern (3)')
gnuplot.plot({trainData.data[{{size1,size2},3}],style})
gnuplot.xlabel('Element Number')
gnuplot.ylabel('Value')
gnuplot.plotflush()

--- Log function (downwards)

gnuplot.pngfigure('figures/downwards_1_fd001.png')
gnuplot.title('Downwards (13,18)')
gnuplot.plot({trainData.data[{{size1,size2},13}],style},{trainData.data[{{size1,size2},18}],style})

gnuplot.plot({(trainData.data[{{size1,size2},13}]+trainData.data[{{size1,size2},18}])/2,style})


gnuplot.xlabel('Element Number')
gnuplot.ylabel('Value')
gnuplot.plotflush()

--- Steeper

gnuplot.pngfigure('figures/steepdownwards_1_fd001.png')
gnuplot.title('Downwards Steeper (11,16,24,25)')
gnuplot.plot({trainData.data[{{size1,size2},11}],style},{trainData.data[{{size1,size2},16}],style},{trainData.data[{{size1,size2},24}],style},{trainData.data[{{size1,size2},25}],style})

gnuplot.plot({(trainData.data[{{size1,size2},11}]+trainData.data[{{size1,size2},16}]+trainData.data[{{size1,size2},24}]+trainData.data[{{size1,size2},25}])/4,style})
gnuplot.xlabel('Element Number')
gnuplot.ylabel('Value')
gnuplot.plotflush()

local elements = trainData.data[{{},1}]
local downward = (trainData.data[{{},11}]+trainData.data[{{},16}]+trainData.data[{{},24}]+trainData.data[{{},25}])/4
local upward = (trainData.data[{{},6}]+trainData.data[{{},7}]+trainData.data[{{},8}]+
  trainData.data[{{},12}]+trainData.data[{{},15}]+trainData.data[{{},17}]+trainData.data[{{},19}]+trainData.data[{{},21}])/8

--- Summation

gnuplot.pngfigure('figures/summation_1_fd001.png')
gnuplot.title('Sum of Downwards and Upwards (11,16,24,25)')
gnuplot.plot({downward + upward})
gnuplot.plotflush()

--- Average

gnuplot.pngfigure('figures/summation_1_fd001.png')
gnuplot.title('Sum of Downwards and Upwards (11,16,24,25)')
gnuplot.plot({downward + upward})
gnuplot.plotflush()