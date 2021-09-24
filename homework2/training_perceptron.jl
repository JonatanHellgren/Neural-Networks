using Plots
using Distributions
using DelimitedFiles


function main()
    m1 = 100
    g = tanh
    function g_prime(x) return 1 - tanh(x)^2 end
    pelle = init_perceptron([2, m1, 1], g, g_prime, 0.01)

    train(pelle, X_train, X_val, 1000, 20)

    scatter(X_val[:,1], X_val[:,2], color = heavy_side.(predict(pelle, X_val)))

    writedlm("/home/jona/NN/homework2/w1.csv", pelle.W[1], ',')
    writedlm("/home/jona/NN/homework2/w2.csv", pelle.W[2]', ',')
    writedlm("/home/jona/NN/homework2/t1.csv", pelle.Θ[1], ',')
    writedlm("/home/jona/NN/homework2/t2.csv", pelle.Θ[2], ',')
end


function load_data()
    training_path = "/home/jona/NN/homework2/training_set.csv"
    X_train = read_csv(training_path)

    validation_path = "/home/jona/NN/homework2/validation_set.csv"
    X_val = read_csv(validation_path)

    normalize_data(X_train, X_val)

end

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

function write_csv(path, matrix)
    row, col = size(matrix)
    open(path, "w") do f
        for r in 1:row
            for c in 1:col
                print(f, matrix[r, c])
                if c != col
                    print(f, ',')
                end
            end
            print(f, '\n')
        end
    end
end

main()
