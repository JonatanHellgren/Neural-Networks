using Distributions
using MLDatasets
using StaticArrays
using TimerOutputs

"""
Defining network
"""
abstract type Layer end
mutable struct NeuralNetwork
  # create a vector containing layers and pass it to the default
  # constructor to initalize the network
  layer::Vector{Layer}
  η::Float64
  α::Float64
end

function NeuralNetwork(dim::Vector{Int64}, η, α)
  layer = Vector{Layer}(undef, length(dim))
  layer[1] = InputLayer(dim[1])
  for ind in 2:length(dim)
    layer[ind] = DenseLayer(dim[ind-1], dim[ind], load_relu())
  end
  layer[end] = OutputLayer(dim[end-1], dim[end], load_softmax())
  NeuralNetwork(layer, η, α)
end

"""
Activation functions
"""
mutable struct ActivationFunction
  fun::Function
  prime::Function
end

function load_relu()
  fun(a) = max(a,0)
  prime(a) = a > 0 ? 1 : 0
  ActivationFunction(fun, prime)
end


"""
Energy functions
"""
mutable struct EnergyFunction
  output::Function
  energy::Function
  error::Function
end

function load_softmax()
  output(v::Vector{Float64}) = exp.(v)/sum(exp.(v))
  energy(target::Vector{Float64}, output::Vector{Float64}) = - sum(target .* log.(output))
  error(target::Vector{Float64}, output::Vector{Float64}) = target .- output
  EnergyFunction(output, energy, error)
end

"""
Input layer
"""
mutable struct InputLayer <: Layer
  v::Array{Float64}
end

function InputLayer(dim::Int64)
  InputLayer(zeros(dim))
end

"""
Dense layer
"""
mutable struct DenseLayer <: Layer
  g::ActivationFunction
  w::Matrix{Float64} 
  δw::Matrix{Float64}
  θ::Vector{Float64}
  δθ::Vector{Float64}
  δ::Vector{Float64}
  b::Vector{Float64}
  v::Vector{Float64}
end

function DenseLayer(in::Int64, sz::Int64, g::ActivationFunction)
  # initalizing weights
  distribution = Normal(0, 1/sz)
  w = rand(distribution, (sz, in))
  δw = zeros((sz, in))
  # initalizing bias
  θ = zeros(sz)
  δθ = zeros(sz)
  # initalizing other vectors used
  δ = zeros(sz)
  b = zeros(sz)
  v = zeros(sz)
  # constructing the layer
  DenseLayer(g, w, δw, θ, δθ, δ, b, v)
end

function forward(l::DenseLayer, input::Vector{Float64})
  l.b = l.w * input .- l.θ
  l.v = l.g.fun.(l.b)
end

function backward(network::NeuralNetwork, layer_ind::Int64)
  @timeit "a1" current_layer = network.layer[layer_ind]
  @timeit "a2" previous_layer = network.layer[layer_ind+1]
  @timeit "δ" current_layer.δ = previous_layer.w' * previous_layer.δ .* current_layer.g.prime.(current_layer.b)
  @timeit "δw" current_layer.δw += network.η * ( current_layer.δ * network.layer[layer_ind-1].v' )
  @timeit "δθ" current_layer.δθ -= network.η * current_layer.δ
  # print(current_layer.δθ)
end

"""
Output layer
"""
mutable struct OutputLayer <: Layer
  g::EnergyFunction
  w::Matrix{Float64} 
  δw::Matrix{Float64}
  θ::Vector{Float64}
  δθ::Vector{Float64}
  δ::Vector{Float64}
  b::Vector{Float64}
  v::Vector{Float64}
  output::Vector{Float64}
  H::Float64
end

function OutputLayer(in::Int64, sz::Int64, g::EnergyFunction)
  # initalizing weights
  distribution = Normal(0, 1/sz)
  w = rand(distribution, (sz, in))
  δw = zeros((sz, in))
  # initalizing bias
  θ = zeros(sz)
  δθ = zeros(sz)
  δ = zeros(sz)
  b = zeros(sz)
  v = zeros(sz)
  output = zeros(sz)
  H = 0.0
  OutputLayer(g, w, δw, θ, δθ, δ, b, v, output, H)
end

function forward(l::OutputLayer, input::Vector{T}) where T <: Number
  l.b = l.w * input .- l.θ
  # println(input)
  l.v = l.g.output(l.b)
end

function backward(network::NeuralNetwork, target::Vector{T}) where T <: Number
  network.layer[end].δ = network.layer[end].g.error(target, network.layer[end].v)
  # println(target, '\n', round.(network.layer[end].v, digits=2))
  # println(network.layer[end].δ, '\n', target)
  network.layer[end].δw += network.η * (network.layer[end].δ * network.layer[end-1].v')
  network.layer[end].δθ -= network.η * network.layer[end].δ
end


"""
Forward propagation
"""
function forward_prop(network::NeuralNetwork, x::Vector{T}) where T <: Number
  network.layer[1].v = x
  for l in 2:length(network.layer)
    forward(network.layer[l], network.layer[l-1].v)
  end
end


"""
Backward propagation
"""
function backward_prop(network::NeuralNetwork, label::Int64) where T <: Number
  # Svector here svector @
  target = Float64.([i == label for i in 0:9])
  @assert sum(target) == 1 "$label"
  backward(network, target)
  trainable_layers = length(network.layer)-1
  for ind in trainable_layers:-1:2
    backward(network, ind)
  end
end


"""
Fit network
"""
function fit(network::NeuralNetwork, X_train, y_train, X_test, y_test,
              mini_batch::Int64, epochs)
  reset_timer!()
  best_acc = 0
  sz = size(X_train)[2]
  for epoch in 1:epochs
    for ind in 1:mini_batch:sz
      for _ in 1:mini_batch
        sample_ind = sample(1:sz)
        forward_prop(network, X_train[:,sample_ind])
        backward_prop(network, y_train[sample_ind])
      end
      update_weights(network)
    end
    predictions = predict(network, X_test, y_test)
    accuracy = score(network, y_test, predictions)
    if accuracy > best_acc
      best_acc = accuracy
    end
    energy = network.layer[end].H
    @assert energy > 0 "$energy"
    network.layer[end].H = 0.0
    println("Epoch $epoch\n energy = $energy\n accuracy = $accuracy\n best_acc = $best_acc")
  end
  print_timer()
end

function update_weights(network::NeuralNetwork)
  n_layers = length(network.layer)
  for ind in 2:n_layers
    network.layer[ind].w += network.layer[ind].δw
    network.layer[ind].θ += network.layer[ind].δθ
    # println(network.layer[ind].δw)
    network.layer[ind].δw *= network.α #-= network.layer[ind].δw
    network.layer[ind].δθ *= network.α #network.layer[ind].δθ
  end
end


"""
Predict network
"""
function predict(network::NeuralNetwork, X::Matrix{Float64}, y::Vector{Int64})
  output = Vector{Int64}(undef, size(X)[2])
  for ind in 1:size(X)[2]
    forward_prop(network, X[:,ind])
    output[ind] = argmax(network.layer[end].v) - 1
    # @Svector 
    target = Float64.([i == y[ind] for i in 0:9])
    # target = zeros(SVector{10})
    target[y[ind]+1] = 1.0
    # println(y[ind], "\n", network.layer[end].v)
    # println(network.layer[end].δ)
    network.layer[end].H += network.layer[end].g.energy(target, network.layer[end].v)
  end
  return output
end


"""
Score network
"""
function score(network::NeuralNetwork, labels::Vector{T}, predictions::Vector{T}) where T <: Number
  correct = 0
  sz = size(labels)[1] 
  for ind in 1:sz
    if labels[ind] == predictions[ind]
      correct += 1
    end
  end
  return correct/sz
end


"""
Loading the MNIST dataset
"""
function load_data()
  X_train, y_train = MNIST.traindata()
  X_test, y_test = MNIST.testdata()
  X_train = Float64.(reshape(X_train, 28*28, 60000))
  X_test = Float64.(reshape(X_test, 28*28, 10000))
  return X_train, y_train, X_test, y_test
end


function main()
  X_train, y_train, X_test, y_test = load_data()
  network = NeuralNetwork([784, 64, 64, 10], 0.001, 0.2)
  # @time fit(network, X_train[:,1:10], y_train[1:10], X_test[:,1:2], y_test[1:2], 100, 1)
  @time fit(network, X_train, y_train, X_test, y_test, 20, 100)
  # benchmark 50,50: 12.3s
end

main()

