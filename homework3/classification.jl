using DelimitedFiles
using Flux, CUDA
using Flux.Data: DataLoader
using Flux: onehotbatch, onecold, crossentropy
using Flux: @epochs
using Statistics
using MLDatasets
using NPZ

function load_test_data(;dim=2)
  x_test = Float32.(npzread("xTest2.npy"))./255

  for μ in 1:size(x_test, 4)
    x_test[:, :, 1, μ] = x_test[:, :, 1, μ]'
  end

  if dim == 1
    x_test = reshape(x_test, 28*28, 1, 10000)
  end

  return x_test
end

function load_data(;dim=2)
  x_train, y_train = MLDatasets.MNIST.traindata()
  x_valid, y_valid = MLDatasets.MNIST.testdata()

  x_train = Float32.(Flux.unsqueeze(x_train, 3))
  x_valid = Float32.(Flux.unsqueeze(x_valid, 3))

  y_train = onehotbatch(y_train, 0:9)
  y_valid = onehotbatch(y_valid, 0:9)

  if dim == 1
    x_train = reshape(x_train, 28*28, 1, 60000)
    x_valid = reshape(x_valid, 28*28, 1, 10000)
  end

  return x_train, y_train, x_valid, y_valid
end

function train_model(loss, parameters, data_loader, optimizer)
  for batch in data_loader
    gradient = Flux.gradient(parameters) do
      training_loss = loss(batch.data, batch.label)
      return training_loss
    end
    Flux.update!(optimizer, parameters, gradient)
  end
end

function main(epochs)
  x_train, y_train, x_valid, y_valid = load_data() |> gpu
  data_loader = DataLoader((data=x_train, label=y_train), batchsize=128) |> gpu
  
  model = get_model() |> gpu

  optimizer = Descent(0.1)
  accuracy(ŷ, y) = mean(ŷ .== y)
  loss(x, y) = Flux.crossentropy(model(x), y)
  parameters = params(model)

  for it = 1:epochs
    train_model(loss, parameters, data_loader, optimizer)

    ŷ_train = onecold(model(x_train), 0:9);
    acc_train = round( accuracy(ŷ_train, onecold(y_train, 0:9)), digits=3)
    loss_train = round( loss(x_train, y_train), digits=3)

    ŷ_valid = onecold(model(x_valid), 0:9);
    acc_valid = round( accuracy(ŷ_valid, onecold(y_valid, 0:9)), digits=3)
    loss_valid = round( loss(x_valid, y_valid), digits=3)

    println("Epoch $it: train accuracy: $acc_train, train loss: $loss_train
            \t valid accuracy: $acc_valid, valid loss: $loss_valid\n")
  end


  x_test = load_test_data()
  ŷ = onecold(model(x_test), 0:9) |> cpu
  println(ŷ[1:10])
  writedlm("/home/jona/NN/homework3/classifications.csv", ŷ, ',')
end

function get_model()
  model = Chain(
    # 28x28 => 28x28x8
    Conv((3,3), 1=>8, pad=1, stride=1, relu),
    # => 14x14x8
    x -> maxpool(x, (2,2)),
    # => 14x14x16
    Conv((3,3), 8=>16, pad=1, stride=1, relu),
    # => 7x7x16
    x -> maxpool(x, (2,2)),
    # => 784
    x -> reshape(x, :, size(x,4)),
    Dense(784, 10),
    softmax
  )
end

"""
  Dense(784, 64, relu),
  Dense(64, 64, relu),
  Dense(64, 64, relu),
  Dense(64, 10),
  softmax
"""

@time begin
  #main(1e1)
end









