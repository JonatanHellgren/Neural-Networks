using Distributions
using Plots

# This struct object is all the parameters in the perceptron model, so we can
# consider it to be the model
mutable struct perceptron
    W::Any # 3-d weight vector
    δW::Any
    V::Any # 2-d neuron values matrix
    B::Any # 2-d local field matrix
    Θ::Any # 2-d bias matrix
    δΘ::Any
    δ::Any # 2-d matrix where the error term is stored
    η::Any # learning rate
    g::Any # activation function
    g_prime::Any
    dim::Any
end

# This function makes it easier to create a perceptron struct. It is even
# possible to create multi-layered perceptrons here. It takes a vector with the
# dimensions, a activation function, the activation functions derivative and a
# learning rate as input and outputs the desired struct.
function init_perceptron(dimensions, activation_function, derivative, η)
    dim = length(dimensions)
    W = [randn(dimensions[2], dimensions[1])] #./ dimensions[2]
    δW = [randn(dimensions[2], dimensions[1])] #./ dimensions[2]
    Θ = [zeros(dimensions[2])]
    δΘ = [zeros(dimensions[2])]
    δ = [zeros(dimensions[2])]
    for ind in 2:dim-1
        push!(W, randn(dimensions[ind+1], dimensions[ind])) #./ dimensions[ind+1]
        push!(δW, randn(dimensions[ind+1], dimensions[ind])) #./ dimensions[ind+1]
        push!(Θ, zeros(dimensions[ind+1]))
        push!(δΘ, zeros(dimensions[ind+1]))
        push!(δ, zeros(dimensions[ind+1]))
    end
    V = [zeros(dimensions[1])]
    B = [zeros(dimensions[1])]
    for ind in 2:dim
        push!(V, zeros(dimensions[ind]))
        push!(B, zeros(dimensions[ind]))
    end
    g = activation_function
    g_prime = derivative
    model = perceptron(W, δW, V, B, Θ, δΘ, δ, η, g, g_prime, dim)
    return model
end

# This function trains the perceptron on the data in X_train for a certain
# number of epochs and batch_size, after every epoch we evaluate our model on
# the validation data and stop the training if the accuracy is less then 0.118
function train(this, X_train, X_val, epochs, batch_size)
    for it in 1:epochs
        fit(this, X_train, batch_size)
        train_score = score(this, X_train)
        val_score = score(this, X_val)
        println(it, "; C_train = ", train_score, ",C_val = ", val_score)
        if val_score < 0.118
            break
        end
    end
end

# This function takes a perceptron struct a matrix X and an integer batch_size
# as input and fits the perceptron to the data in X.
function fit(this, X, batch_size)
    ν_max = size(X)[1]
    for batch in 1:(ν_max/batch_size)
        for _ in 1:batch_size
            μ = sample(1:ν_max, 1)[1]
            t = X[μ, 3]
            forward_propegate(this, X[μ, 1:2])
            back_propegate(this, t)
        end
        update_network(this)
    end
end

# This functions performs a forward propegation for a given coordinate, updates
# all the local fields and neurons
function forward_propegate(this, X)
    this.V[1] = X#[1:this.dim]
    for (ind, w) in enumerate(this.W)
        this.B[ind+1] = w * this.V[ind] .- this.Θ[ind]
        this.V[ind+1] = this.g.(this.B[ind+1])
    end
end

# This function performs a backwards propegation, i.e it computes all the δ for
# all layers and updates the weight increments
function back_propegate(this, t)
    this.δ[end] = (t .- this.V[end]) .* this.g_prime.(this.B[end])
    for ind in (this.dim-1):2
        this.δ[ind-1] = (this.W[ind]' * this.δ[ind]) .* this.g_prime.(this.B[ind])
    end
    for (ind, δ) in enumerate(this.δ)
        this.δW[ind] += this.η .* (δ * this.V[ind]')
        this.δΘ[ind] -= this.η .* δ
    end
end

# This function adds the weight increments to the parameter matrices and resets
# the increment matrices afterwards
function update_network(this)
    for ind in 1:this.dim-1
        this.W[ind] += this.δW[ind]
        this.Θ[ind] += this.δΘ[ind]
    end
    this.δW -= this.δW
    this.δΘ -= this.δΘ
end

# Here we take a matrix as input and let the model predict on it. We later
# return the predictions.
function predict(this, X)
    output = []
    sz = size(X)[1]
    for ind in 1:sz
        forward_propegate(this, X[ind, 1:2])
        push!(output, sign.(this.V[end][1]))   # !push appends
    end
    return output
end

# This function we use to score our model, it returns the accuracy for the
# predictions
function score(this, X)
    output = predict(this, X)
    target = X[:,end]
    p_val = length(output)
    total = 0
    for (o, t) in zip(output, target)
        total += abs(o - t)
    end
    return total / (2 * p_val)
end
