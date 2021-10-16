using BSON: @save, @load
using Flux, CUDA
using Flux: onehotbatch, onecold 
using Plots
using MLDatasets
using NPZ
using DelimitedFiles

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

function get_model()
  model = Chain(
  Dense(784, 50, relu),
  Dense(50, 2, relu),
  Dense(2, 784, relu)
  )
end

function get_encoder(latent_dim)
  model = Chain(
  Dense(784, 50, relu),
  Dense(50, latent_dim, relu)
  )
end

function get_decoder(latent_dim)
  model = Chain(
  Dense(latent_dim, 784, relu)
  )
end


function save_model(model, name)
  @save "$name.bson" model
end

function load_model(model, name)
  @load "$name.bson" model
  return model
end

function init_model(;latent_dim=2)
  println("loading model")
  encoder = get_encoder(latent_dim)
  decoder = get_decoder(latent_dim)

  return decoder, encoder
end


function fit_auto_encoder(decoder, encoder, epochs)
  println("Loading data")
  x_train, y_train, x_valid, y_valid = load_data(dim=1) |> gpu
  data_loader = Flux.DataLoader((data=x_train, label=x_train), batchsize=8192, shuffle=true) |> gpu


  optimizer = ADAM(0.001)
  loss(x, y) = Flux.mse(decoder(encoder(x)), y)
  parameters = params(encoder, decoder)

  println("training")
  for it = 1:epochs
    train_model(loss, parameters, data_loader, optimizer)

    loss_train = round( loss(x_train, x_train), digits=3)
    loss_valid = round( loss(x_valid, x_valid), digits=3)

    println("Epoch $it: train accuracy: train loss: $loss_train valid loss: $loss_valid\n")
  end

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


function encode_data(encoder, data, name)
  encoder = encoder(data) |> cpu
  encoded = reshape(encoder, 4, size(data, 3))'
  writedlm("/home/jona/NN/homework3/$name", encoded, ',')
end

function decode_data(encoder, decoder, data, name)
  decoded = decoder(encoder(data)) |> cpu
  decoded = reshape(decoded, 784, size(data, 3))'
  writedlm("/home/jona/NN/homework3/$name", decoded, ',')
end

function plot_number(encoder, decoder, x, file)
  x_autoencoded = decoder(encoder(x))
  x_cpu = x |> cpu
  x_autoencoded = x_autoencoded |> cpu
  original = 1 .- reshape(x_cpu, 28,28)[:,end:-1:1]'
  auto_encoded = 1 .- reshape(x_autoencoded, 28,28)[:,end:-1:1]'
  p1 = heatmap(original, c=:grays, legend=:none)
  p2 = heatmap(auto_encoded, c=:grays, legend=:none)
  plot(p1, p2, layout=2)
  png(file)
end

function plot_numbers(encoder, decoder, x_test, l_dim)
  plot_number(encoder, decoder, x_test[:,1,5], "zero_$l_dim.png")
  plot_number(encoder, decoder, x_test[:,1,3], "one_$l_dim.png")
  plot_number(encoder, decoder, x_test[:,1,30], "two_$l_dim.png")
  plot_number(encoder, decoder, x_test[:,1,24], "three_$l_dim.png")
  plot_number(encoder, decoder, x_test[:,1,1], "four_$l_dim.png")
  plot_number(encoder, decoder, x_test[:,1,31], "five_$l_dim.png")
  plot_number(encoder, decoder, x_test[:,1,23], "six_$l_dim.png")
  plot_number(encoder, decoder, x_test[:,1,66], "seven_$l_dim.png")
  plot_number(encoder, decoder, x_test[:,1,10], "eight_$l_dim.png")
  plot_number(encoder, decoder, x_test[:,1,2], "nine_$l_dim.png")
  #0 -> 5
  #1 -> 3
  #2 -> 30
  #3 -> 24
  #4 -> 1
  #5 -> 31
  #6 -> 23
  #7 -> 66
  #8 -> 10
  #9 -> 2
  #
end

function make_models()
  encoder2, decoder2 = init_model(latent_dim=2) |> gpu
  fit_auto_encoder(decoder2, encoder2, 1000)
  # @save "encoder2.bson" encoder2
  # @save "decoder2.bson" decoder2

  encoder4, decoder4 = init_model(latent_dim=4) |> gpu
  fit_auto_encoder(decoder4, encoder4, 1000)
  # @save "encoder4.bson" encoder4
  # @save "decoder4.bson" decoder4
end

"""
encoder2.bson decoder2.bson
at epoch 1000
Latent dim | train loss | valid loss
2           0.066        0.066

encoder4.bson decoder4.bson
at epoch 1000
Latent dim | train loss | valid loss
4           0.049        0.049

encoder8.bson decoder8.bson
at epoch 1000
Latent dim | train loss | valid loss
8           0.033        0.033

encoder16.bson decoder16.bson
at epoch 1000
Latent dim | train loss | valid loss
16          0.021        0.02
"""


