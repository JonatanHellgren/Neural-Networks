using Distributions

abstract type NeuralNetwork end
abstract type Layer end

mutable struct ActivationFunction
  fun::Function
  prime::Function
end
  
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

mutable struct ConvLayer <: Layer 
  g::ActivationFunction
  w::Vector{Matrix{Float64}}
  δw::Vector{Matrix{Float64}}
  θ::Vector{Float64}
  δθ::Vector{Float64}
  δ::Array{Float64}
  b::Array{Float64}
  v::Array{Float64}
end

mutable struct ConvNetwork1 <: NeuralNetwork
  # create a vector containing layers and pass it to the default
  # constructor to initalize the network
  layer::Vector{Layer}
  η::Float64
  α::Float64
end


function convLayer(dim::Tuple, sz::Int, g::ActivationFunction)
  # only accept symmetric convolution dimentions
  @assert dim[1] == dim[2] "Convolution must have symetric dimentions"
  # initalizing weights
  distribution = Normal(0, 1/sz)
  w = fill(rand(distribution, dim), sz)
  δw = fill(zeros(dim), sz)
  # initalizing bias
  θ = zeros(sz)
  δθ = zeros(sz)
  # initalizing other vectors used
  δ = zeros(sz)
  b = zeros(sz)
  v = zeros(sz)
  # constructing the layer
  ConvLayer(g, w, δw, θ, δθ, δ, b, v)
end


function denseLayer(dim::Tuple, g::ActivationFunction)
  sz = dim[1]
  # initalizing weights
  distribution = Normal(0, 1/sz)
  w = rand(distribution, dim)
  δw = zeros(dim)
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


relu = ActivationFunction(function fun(a) max(a,0) end,
                          function prime(a) a > 0 ? 1 : 0 end)


function Convolution(w::Matrix{Float64})
  row, col = size(w)
  convolution(w, row)
end

function convolve_matrix(A::Matrix{T}, c::convolution) where T <:Number
  nRow, nCol = size(A)
  B = zeros(Float64, size(A) .- (2,2))
  for i in 2:nRow-1, j in 2:nCol-1
    B[i-1, j-1] = apply_convolution(A, c, i, j)
  end
  return B
end

function apply_convolution(A::Matrix{T}, c::convolution, i, j) where T <:Number
  nRow, nCol = size(c.w)
  cMid = trunc(Int, (nRow + 1)/2)
  B_ij = 0
  for ci in 1:nRow, cj in 1:nCol
    ai = i + (ci - cMid)
    aj = j + (ci - cMid)
    B_ij += c.w[ci, cj] * A[ai, aj]
  end
  return B_ij
end

