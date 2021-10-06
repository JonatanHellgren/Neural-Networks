using Plots
using Distributions
using DelimitedFiles

# The main function initalizes by loading the the data and constructing the model.
# Then it trains the model for 1000 epochs and with a mini-batch of size 10. The
# traingin will be stopped when the desired accuracy is reached. When it is done
# we will plot the results and save the matrices containing the parameters.
function main()
    X_train, X_val = load_data()
    m1 = 100
    g = tanh
    function g_prime(x) return 1 - tanh(x)^2 end
    pelle = init_perceptron([2, m1, 1], g, g_prime, 0.01)
    train(pelle, X_train, X_val, 1000, 10)
    scatter(X_val[:,1], X_val[:,2], color = heavy_side.(predict(pelle, X_val)))
    # writedlm("/home/jona/NN/homework2/w1.csv", pelle.W[1], ',')
    # writedlm("/home/jona/NN/homework2/w2.csv", pelle.W[2]', ',')
    # writedlm("/home/jona/NN/homework2/t1.csv", pelle.Θ[1], ',')
    # writedlm("/home/jona/NN/homework2/t2.csv", pelle.Θ[2], ',')
end

# We load the data here and also normalizes it before returning.
function load_data()
    training_path = "/home/jona/NN/homework2/training_set.csv"
    X_train = read_csv(training_path)
    validation_path = "/home/jona/NN/homework2/validation_set.csv"
    X_val = read_csv(validation_path)
    normalize_data(X_train, X_val)
    return X_train, X_val
end

# This is a helper function which normalizes the data, it takes as input two
# matrices and normalizes both of them acording to the the mean and standard
# deviation in the first matrix. Hard coded to only work with two matrices.
function normalize_data(df1, df2)
    dim = size(df1)[2] - 1
    for it in 1:dim
        μ = mean(df1[:,it])
        σ = std(df1[:,it])
        df1[:,it] = df1[:,it] .- μ
        df2[:,it] = df2[:,it] .- μ
        df1[:,it] = df1[:,it] / σ
        df2[:,it] = df2[:,it] / σ
    end
end

# I couldn't find a good csv reader in Julia so I did my own, it is however
# hard-coded for this problem specificly. It takes a file path as input and
# returns a matrix containging the values in that file.
function read_csv(path)
    open(path) do f
        lines = readlines(f)
        sz = length(lines)
        M = zeros(sz, 3)
        for (ind, line) in enumerate(lines)
            x1, x2, y = split(line, ',')
            M[ind, :] = [parse(Float64, x1), parse(Float64, x2), parse(Int, y)]
        end
        return M
    end
end

# A function used in homework 1 that found its place in this script since it
# made the plotting possible
function heavy_side(n)
    return n > 0 ? 1 : 0
end

main()
