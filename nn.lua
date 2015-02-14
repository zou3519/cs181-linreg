require 'torch'
require 'cutorch'
require 'cunn'
require 'csvigo'

function square(x)
    return x^2
end

-- computes the normal of tensor a
function norm(a)
    local sum = 0
    local terms = a:size()[1]
    for i=1,terms do
        sum = sum + (a[i])^2
    end
    sum = sum/terms
    return sum^(0.5)
end

-- given two tensors, compute the root mean square error
function rmse(a, b)
    return norm(a-b)
end

-- define the model
function create_nn()
    print("==> Creating nn")
    local mlp = nn.Sequential();
    inputs =1024;
    outputs = 1;
    hidden_units = 512
    hidden_layers = 1;
    mlp:add(nn.Linear(inputs, hidden_units))
    for i = 1, hidden_layers do
        mlp:add(nn.Tanh())
        mlp:add(nn.Linear(hidden_units, hidden_units))
    end
    mlp:add(nn.Tanh())
    mlp:add(nn.Linear(hidden_units, outputs))

    -- move it to the gpu
    mlp:cuda()
    return mlp
end

-- train model on dataset
function train_model(mlp, dataset)
    torch.setdefaulttensortype('torch.CudaTensor')
    function dataset:size() return table.getn(dataset) end

    criterion = nn.MSECriterion() -- mean squared error
    trainer = nn.StochasticGradient2(mlp, criterion)
    trainer.maxIteration = 25
    trainer:train(dataset)
end

-- dataset creation
function read_file(file)
    print("==> Reading " .. file)
    local fh = torch.DiskFile(file, 'r')
    return fh:readObject()
end

function features(data, row)
    return data:sub(row,row,1,1024)
end

function targets(data,row)
    return data:sub(row,row,1025,1025)
end

function evaluate_model(model, dataset)
    local n = table.getn(dataset)
    outs = torch.Tensor(n)
    expected = torch.Tensor(n)

    for i = 1, n do
        feat = dataset[i][1]
        outs[i] = model:forward(feat)
        expected[i] = dataset[i][2]
    end
    print("RMSE: "..rmse(outs, expected))    
end

function perform_test(model, testData)
    -- testData = read_file('test.t7')
    rows = testData:size()[1]
    local result = torch.Tensor(rows)
    for i=1, rows do
        local mol = testData[i]
        out = model:forward(mol:cuda())
        result[i] = out
    end
    torch.save('test_out.t7', result)
    return result
end

-- rowstart and rowend are inclusive
function build_dataset(trainData, rowstart, rowend)
    local dataset = {}
    rows = trainData:size()[1]
    total = rowend - rowstart + 1
    for i=1, total do
        rownum = rowstart + i - 1
        dataset[i] = {features(trainData,rownum):cuda(), targets(trainData,rownum):cuda()   }
    end

    return dataset
end

function do_things()
    trainData = read_file('train.t7')
    dataset = build_dataset(trainData, 1, 10000)
    mlp = create_nn()
    train_model(mlp,dataset)
end

function init_()
    dofile('StochasticGradient.lua')
    trainData = read_file('train.t7')
    ds1 = build_dataset(trainData, 1, 5000)
    ds2 = build_dataset(trainData, 5001, 10000)
end
