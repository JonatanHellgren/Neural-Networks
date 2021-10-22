using BSON: @save, @load
using Flux, CUDA
using Flux: onehotbatch, onecold 
using Plots
using MLDatasets
using NPZ
using DelimitedFiles
using Distributions

"""
Function used to load the numpy data for testing
"""
function load_test_data(;dim=2)
  x_test = Float32.(npzread("xTest2.npy"))./255
  # this loops just transposes the digits
  for μ in 1:size(x_test, 4)
    x_test[:, :, 1, μ] = x_test[:, :, 1, μ]'
  end
  if dim == 1
    x_test = reshape(x_test, 28*28, 1, 10000)
  end

  return x_test
end

"""
Loads the mnist data, either 2d or 1d. It also transforms the data into float32
since that is what makes my gpu go wosh.
"""
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

"""
This is how I load the encoder, just two layers with sizes 784 and 50, both with ReLu
activation.
"""
function get_encoder(latent_dim)
  model = Chain(
  Dense(784, 50, relu),
  Dense(50, latent_dim, relu)
  )
end

"""
Similar but for the decoder, here we connect the latent dimention to a dense layer of
size 784 with ReLu activations function. Finally we end this autoencoder with a regression
layer of size 784
"""
function get_decoder(latent_dim)
  model = Chain(
  Dense(latent_dim, 784, relu),
  Dense(784, 784)
  )
end

"""
This function is just to make the loading of the encoders much simpler, simply
give it the size of your desired latent dimention and it gives you the part you need
"""
function init_model(;latent_dim=2)
  println("loading model")
  encoder = get_encoder(latent_dim)
  decoder = get_decoder(latent_dim)
  return decoder, encoder
end

"""
This function fit the parameters in the auto encoder, loads the data and prints
what loss we are currently on for our epoch
"""
function fit_auto_encoder(decoder, encoder, epochs)
  println("Loading data")
  x_train, y_train, x_valid, y_valid = load_data(dim=1) |> gpu
  data_loader = Flux.DataLoader((data=x_train, label=x_train), batchsize=8192, shuffle=true) |> gpu

  optimizer = ADAM(0.001)
  loss(x, y) = Flux.mse(decoder(encoder(x)), y)
  parameters = Flux.params(encoder, decoder)

  println("training")
  for it = 1:epochs
    train_model(loss, parameters, data_loader, optimizer)

    loss_train = round( loss(x_train, x_train), digits=3)
    loss_valid = round( loss(x_valid, x_valid), digits=3)

    println("Epoch $it: train accuracy: train loss: $loss_train valid loss: $loss_valid\n")
  end

end

"""
Function to train the network, it was possible to use Flux.train! here, but it 
didn't work for me somehow, so I found this online and modified it a bit to make it 
accept my data_loader.
"""
function train_model(loss, parameters, data_loader, optimizer)
  for batch in data_loader
    gradient = Flux.gradient(parameters) do
      training_loss = loss(batch.data, batch.label)
      return training_loss
    end
    Flux.update!(optimizer, parameters, gradient)
  end
end

"""
This function is used to plot the original digits and the autoencoded version of it
for the montage asked for in the question
"""
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
end

"""
Helper function used to plot a individual input
"""
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

"""
Function used to train and save both the models, one with latent_dim=2 and the other
one with latent_dim=4. It trains both models for 1000 epochs each
"""
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
This function is to asked for in the question. It creates a scatter plot of the latent
dimention for the well defined digits.
"""
function draw_scatter_plot1(encoder, x_valid, y_valid)
  latent_dim = encoder(x_valid[:, 1, 1:1000])
  labels = onecold(y_valid[:, 1:1000], 0:9)
  zeros = latent_dim[:, labels .== 0]
  ones = latent_dim[:, labels .== 1]
  twoes = latent_dim[:, labels .== 2]
  sixes = latent_dim[:, labels .== 6]
  scatter(zeros[1,:], zeros[2,:], color = :green, label = "Zeros")
  scatter!(ones[1,:], ones[2,:], color = :red, label = "Ones")
  scatter!(twoes[1,:], twoes[2,:], color = :blue, label = "Twoes")
  scatter!(sixes[1,:], sixes[2,:], color = :orange, label = "Sixes")
  savefig("Scatter.png")
end

"""
Simlar as the previous function but this one instead plots all them digits.
"""
function draw_scatter_plot2(encoder, x_valid, y_valid)
  latent_dim = encoder(x_valid[:, 1, 1:1000])
  labels = onecold(y_valid[:, 1:1000], 0:9)
  zeros = latent_dim[:, labels .== 0]
  ones = latent_dim[:, labels .== 1]
  twoes = latent_dim[:, labels .== 2]
  threes = latent_dim[:, labels .== 3]
  fours = latent_dim[:, labels .== 4]
  fives = latent_dim[:, labels .== 5]
  sixes = latent_dim[:, labels .== 6]
  sevens = latent_dim[:, labels .== 7]
  eights = latent_dim[:, labels .== 8]
  nines = latent_dim[:, labels .== 9]
  scatter(zeros[1,:], zeros[2,:], color = :green, label = "Zeros")
  scatter!(ones[1,:], ones[2,:], color = :red, label = "Ones")
  scatter!(twoes[1,:], twoes[2,:], color = :blue, label = "Twoes")
  scatter!(threes[1,:], threes[2,:], color = :brown, label = "Threes")
  scatter!(fours[1,:], fours[2,:], color = :grey, label = "Fours")
  scatter!(fives[1,:], fives[2,:], color = :black, label = "Fives")
  scatter!(sixes[1,:], sixes[2,:], color = :orange, label = "Sixes")
  scatter!(sevens[1,:], sevens[2,:], color = :lightblue, label = "Sevens")
  scatter!(eights[1,:], eights[2,:], color = :pink, label = "Eights")
  scatter!(nines[1,:], nines[2,:], color = :lightgreen, label = "Nines")
  savefig("Scatter2.png")
end

"""
Since four dimentional plots are hard we are gonna just plot the centroid for each
cluster created by the digits in the four dimentional latent space. It works
suprisingly well.
"""
function draw_artificial_numbers(encoder, decoder, x, y)
  for i = 0:9
    draw_artifical_number(encoder, decoder, x, y, i)
  end
end

"""
Helper function to draw the artifical numbers
"""
function draw_artifical_number(encoder, decoder, x, y, n)
  labels = onecold(y, 0:9) |> cpu
  latent_dim = encoder(x) |> cpu
  mean_latent_digit = [mean(latent_dim[i, 1, labels .== n]) for i = 1:4] |> gpu
  println(mean_latent_digit)
  decoded_mean_latent_digit = decoder(mean_latent_digit) |> cpu
  ready_to_plot = 1 .- reshape(decoded_mean_latent_digit, 28,28)[:,end:-1:1]'
  heatmap(ready_to_plot, c = :grays, legend = :none)
  savefig("Artificial$n-train")
end


"""
__ Some stats for the models trained __

encoder2.bson decoder2.bson
at epoch 1000
Latent dim | train loss | valid loss
2           0.04        0.042

is able to reconstruct: 0,1,2,6 quite well

encoder4.bson decoder4.bson
at epoch 1000
Latent dim | train loss | valid loss
4           0.028        0.029

"""



