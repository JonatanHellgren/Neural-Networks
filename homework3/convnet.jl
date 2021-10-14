# TODO
# change array{Float64} to array{T} where T <: Number
# define backproagation, fix deltas
# improve convolution, add stide and padding
# implement regularisation
#
#
using Distributions

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
_Layers_
Below we find the types of layers defined.
Each layer is defines as a struct and have three function each
- Constructor
- Forward
- Backward
"""
abstract type Layer end

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

function backward(layer::DenseLayer, activations::Vector{Float64},
                  errorTerm::Float64, η::Float64) where L <: Layer
  # do delta computation in same layer, maybe pass entire networw to the function,
  # or maybe just l+1 
  layer.δ = layer.w' * layer.δ .* prevLayer.g.prime.(prevLayer.b)
end

"""
Convolutional layer
"""
mutable struct ConvLayer <: Layer 
  g::ActivationFunction
  w::Vector{Matrix{Float64}}
  δw::Vector{Matrix{Float64}}
  θ::Vector{Float64}
  δθ::Vector{Float64}
  δ::Vector{Float64}
  b::Vector{Matrix{Float64}}
  v::Vector{Matrix{Float64}}
end

function ConvLayer(dim::Tuple{Int64, Int64, Int64}, sz::Int64, kernelSz::Tuple{Int64, Int64}, g::ActivationFunction)
  # only accept symmetric convolution dimentions
  @assert kernelSz[1] == kernelSz[2] "Convolution must have symetric dimentions"
  # initalizing weights
  distribution = Normal(0, 1/sz)
  w = [rand(distribution, kernelSz) for _ in 1:sz]
  δw = [zeros(kernelSz) for _ in 1:sz]
  # initalizing bias
  θ = zeros(sz)
  δθ = zeros(sz)
  # initalizing other vectors used
  δ = zeros(sz)
  dimentionReduction = kernelSz .- (1,1)
  b = [zeros(dim[1:2] .- dimentionReduction) for _ in 1:sz * dim[3]] # no padding allowed, hence - (2,2)
  v = [zeros(dim[1:2] .- dimentionReduction) for _ in 1:sz * dim[3]]
  # constructing the layer
  ConvLayer(g, w, δw, θ, δθ, δ, b, v)
end

function forward(l::ConvLayer, input::Vector{Matrix{T}}) where T <: Number
  nrow, ncol = size(input[1])
  dimLoss = size(l.w[1])[1] - 1
  ind = 1
  for kernel in l.w, featureMap in input
    for row in 1:nrow-dimLoss, col in 1:ncol-dimLoss
      upperRow = row+dimLoss
      upperCol = col+dimLoss
      # @ inbounds
      l.b[ind][row, col] = sum(view(featureMap, row:upperRow, col:upperCol) .* kernel) - l.θ[ind]
    end
    l.v[ind] = l.g.fun.(l.b[ind]) 
    ind += 1
  end
end


"""
Input layer
"""
mutable struct InputLayer <: Layer
  v::Array{Matrix{Float64}}
end

function InputLayer(dim::Tuple{Int64, Int64})
  InputLayer([zeros(dim)])
end

function InputLayer(dim::Int64)
  InputLayer([zeros(dim)])
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
  l.v = l.g.output(l.b)
end

function backward(layer::OutputLayer, activations::Vector{Float64},
                  target::Vector{T}, η::Float64) where T <: Number
  layer.δ = layer.g.error(target, layer.v)
  layer.δw += η*(activations * layer.δ')
  layer.δθ -= η*layer.δ
end

"""
Flatten layer
"""
mutable struct FlattenLayer <: Layer
  v::Vector{Float64}
end

function FlattenLayer(sz::Int)
  v = zeros(sz)
  FlattenLayer(v)
end

function forward(l::FlattenLayer, input::Vector{Matrix{Float64}})
  sz = length(input[1])
  for (ind, i) in enumerate(input)
    low_ind = sz*(ind-1)+1 
    high_ind = sz*ind
    l.v[low_ind:high_ind] = reshape(i, sz)
  end
end

"""
Pooling layer
"""
mutable struct PoolingLayer <: Layer
  kernel::Tuple{Int64,Int64}
  stride::Tuple{Int64,Int64}
  v::Vector{Matrix{Float64}}
  pooling::String
end

function PoolingLayer(dim::Tuple{Int64,Int64,Int64}, kernel::Tuple{Int64,Int64},
                      stride::Tuple{Int64,Int64}, pooling::String)
  @assert pooling ∈ ("max", "l2") "Unspecified pooling method"
  dimentionReduction = kernel .- (1,1)
  v = [zeros(dim[1:2] .- dimentionReduction) for _ in 1:dim[3]]
  PoolingLayer(kernel, stride, v, pooling)
end

function forward(l::PoolingLayer, input::Vector{Matrix{Float64}})
  dim = size(input[1])
  nrow = dim[1] - l.kernel[1] + 1
  ncol = dim[2] - l.kernel[2] + 1
  for ind in 1:length(input)
    for row in 1:nrow, col in 1:ncol
      l.v[ind][row, col] = max( view(input[ind],
                                     row:row+l.kernel[1]-1, 
                                     col:col+l.kernel[2]-1)...)
    end
  end
end

"""
Defining network
"""
abstract type NeuralNetwork end

mutable struct ConvNetwork <: NeuralNetwork
  # create a vector containing layers and pass it to the default
  # constructor to initalize the network
  layer::Vector{Layer}
  η::Float64
  α::Float64
end


function forwardProp(n::NeuralNetwork, x::Matrix{Float64})
  n.layer[1].v = [x]
  for l in 2:length(n.layer)
    forward(n.layer[l], n.layer[l-1].v)
  end
end

  
relu = load_relu()
softmax = load_softmax()
M = ones(5, 5)
layers = [InputLayer((5,5)),
          ConvLayer((5,5,1), 2, (3,3), relu),
          PoolingLayer((3,3,2), (2,2), (1,1), "max"),
          FlattenLayer(8),
          DenseLayer(8, 10, relu),
          OutputLayer(10, 10, softmax)]

convNet = ConvNetwork(layers, 0.1, 0.9)

forwardProp(convNet, M)
