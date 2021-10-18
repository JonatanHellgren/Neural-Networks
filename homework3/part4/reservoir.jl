using DelimitedFiles
using Distributions
using LinearAlgebra
using Plots

function load_data()
    X_train = readdlm("training-set.csv", ',')
    X_test = readdlm("test-set.csv", ',')
    return X_train, X_test
end

function get_weights(X ;dim=3, N=500)
    distribution_in = Normal(0, sqrt(0.002))
    distribution_r = Normal(0, sqrt(2 / N))
    W_in = rand(distribution_in, (N, dim))
    W_r = rand(distribution_r, (N, N))
    R = compute_R(W_in, W_r, X)
    W_out = ridge_regression(R[:,1:end - 1], X)
    return W_in, W_r, W_out
end

function compute_R(W_in, W_r, X)
    T = size(X, 2)
    N = size(W_r, 1)
    R = zeros(N, T + 1)
    for t = 1:T
        R[:, t + 1] = tanh.(W_r * R[:, t] + W_in * X[:, t])
    end
    return R
end

function ridge_regression(R, X; k=0.01)
    W_out = X * R' * inv(R * R' + k * I)
    return W_out
end

function predict(W_in, W_r, W_out, X; sz=500)
    R = compute_R(W_in, W_r, X)
    O = W_out * R[:,1:end - 1]

    pred = zeros(3, sz)
    R_pred = zeros(500, sz)
    R_pred[:,1] = tanh.(W_r * R[:,end] + W_in * O[:,end])
    pred[:,1] = W_out * R_pred[:,1]
    for it in 1:sz - 1
        #= println(r[1:2], '\n', input) =#
        R_pred[:,it + 1] = tanh.(W_r * R_pred[:,it] + W_in * pred[:,it])
        pred[:,it + 1] = W_out * R_pred[:,it + 1]
    end

    return pred
end

function main()
    X_train, X_test = load_data()
    W_in, W_r, W_out = get_weights(X_train)
    pred = predict(W_in, W_r, W_out, X_test)
    #= pred = zeros(3, 0) =#
    #= for _ = 1:5 =#
    #=     O = predict(W_in, W_r, W_out, O) =#
    #=     pred = hcat(pred, O) =#
    #= end =#
    writedlm("prediction.csv", pred[2,:], ',')
    plot(X_test[1,:], X_test[2,:], X_test[3,:])
    plot!(pred[1,:], pred[2,:], pred[3, :], label="pred")

end

  

  

