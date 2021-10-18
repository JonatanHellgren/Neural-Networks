using Flux

Wxh = randn(Float32, 5, 2)
Whh = randn(Float32, 5, 5)
b   = randn(Float32, 5)

function rnn_cell(h, x)
    h = tanh.(Wxh * x .+ Whh * h .+ b)
    return h, h
end

x = rand(Float32, 2) # dummy data
h = rand(Float32, 5)  # initial hidden state

h, y = rnn_cell(h, x)


