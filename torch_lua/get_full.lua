require 'nn'

opt = {}
-- opt.mode = 'validate'
opt.mode = 'full'
opt.size = 'full'

-- model = torch.load('model_val.net')
model = torch.load('model.net')

dofile 'src/1_fetch_data.lua'

train_pred = model:forward(trainData.data)

file = io.open('train_pred.csv', 'w+')

for i=1,train_pred:size(1) do
	file:write(train_pred[i][1])
	file:write('\n')
end
file:close()

test_pred = model:forward(testData.data)

file = io.open('test_pred.csv', 'w+')

for i=1,test_pred:size(1) do
        file:write(test_pred[i][1])
        file:write('\n')
end
file:close()

