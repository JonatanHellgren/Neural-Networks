using BSON: @save, @load
using Flux, CUDA
using Flux: onehotbatch, onecold 
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

function get_model()
  model = Chain(
  Dense(784, 50, relu),
  Dense(50, 2, relu),
  Dense(2, 784, relu)
  )
end

function save_model(model, name)
  @save "$name.bson" model
end

function load_model(model, name)
  @load "$name.bson" model
  return model
end

function init_model()
  println("loading model")
  model = get_model()

  return model
end


function fit_model(model, epochs)
  println("Loading data")
  x_train, y_train, x_valid, y_valid = load_data(dim=1) |> gpu
  data_loader = Flux.DataLoader((data=x_train, label=x_train), batchsize=8192, shuffle=true) |> gpu

  optimizer = ADAM(0.001)
  loss(x, y) = Flux.mse(model(x), y)
  parameters = params(model)

  println("training")
  for it = 1:epochs
    train_model(loss, parameters, data_loader, optimizer)

    loss_train = round( loss(x_train, x_train), digits=3)
    loss_valid = round( loss(x_valid, x_valid), digits=3)

    println("Epoch $it: train accuracy: train loss: $loss_train valid loss: $loss_valid\n")
  end

  return model
end




