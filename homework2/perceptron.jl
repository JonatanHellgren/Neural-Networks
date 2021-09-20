using Plots

mutable struct perceptron
    W::Any # weights to output layer
    w::Any # weights connecting input to hidden layer
    θ::Any # bias for hidden layer
    Θ::Any # bias for output layer
    # V # Hidden layer
end


function init_perceptron(k, j, i)
    w = zeros(j, k)
    W = zeros(i, j)
    θ = zeros(j)
    Θ = zeros(i)

    model = perceptron(W, w, θ, Θ)
    return model
end

function fit(perceptron, X) end

function predict(perceptron, X)
    _V_ = heavy_side.(*(perceptron.w, X) - perceptron.θ)
    _O_ = heavy_side.(dot(_V_, perceptron.W) - perceptron.Θ)
    return _O_
end

function heavy_side(n)
    return n > 0 ? 1 : 0
end

function score(perceptron, X) end
