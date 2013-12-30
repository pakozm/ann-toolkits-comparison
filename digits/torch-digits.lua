require 'torch'
require 'image'
require 'nn'

model  = nn.Sequential()
model:add(nn.Reshape(256))
model:add(nn.Linear(256, 256))
model:add(nn.Tanh())
model:add(nn.Linear(256, 128))
model:add(nn.Tanh())
model:add(nn.Linear(128, 10))
model:add(nn.LogSoftMax())

digits = image.loadPNG('digits.png')
digits = torch.reshape(digits, digits:size(2), digits:size(3))
train_ds = {}
val_ds   = {}
for i=1,digits:size(1)-1,16 do
  local dest_ds = train_ds
  if i > 1280 then
    dest_ds = val_ds
  end
  local k=1
  for j=1,digits:size(2)-1,16 do
    table.insert(dest_ds, { digits[{ {i,i+16-1}, {j,j+16-1} }], k })
    k=k+1
  end
end

train_ds.size = function() return #train_ds end
val_ds.size = function() return #val_ds end

criterion = nn.ClassNLLCriterion()  

for i = 1,10 do
  local tr_loss = 0
  local va_loss = 0
  for j=1,train_ds:size() do
    local sample_id = torch.random() % train_ds:size() + 1
    local input,output = unpack(train_ds[sample_id])
    -- feed it to the neural network and the criterion
    tr_loss = tr_loss + criterion:forward(model:forward(input), output)
    -- train over this example in 3 steps
    -- (1) zero the accumulation of the gradients
    model:zeroGradParameters()
    -- (2) accumulate gradients
    model:backward(input, criterion:backward(model.output, output))
    -- (3) update parameters with a 0.01 learning rate
    model:updateParameters(0.01)
  end
  for j=1,val_ds:size() do
    local input,output = unpack(val_ds[j])
    -- feed it to the neural network and the criterion
    va_loss = va_loss + criterion:forward(model:forward(input), output)
  end
  tr_loss = tr_loss / train_ds:size()
  va_loss = va_loss / val_ds:size()
  print(string.format("%4d  %.6f  %.6f", i, tr_loss, va_loss))
end
