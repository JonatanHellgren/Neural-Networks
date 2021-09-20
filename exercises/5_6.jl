using Plots

function main()
    pelle = init_perceptron(2, 3, 1)
    pelle.w = [-4 -5; 3 -4; 1 9]
    pelle.θ = [-15, -12, -4]
    pelle.W = [1 1 1]
    pelle.Θ = 2
    # pelle.w = [4/5 5/5; -3/4 4/4; 1/9 9/9]
    # pelle.θ = [3, 3, -4/9]

    draw_pelles_boundary(pelle)
end

function create_grid(x, y)
    N = length(x) * length(y)
    cord = zeros(N, 2)
    for i in 1:length(x)
        for j in 1:length(y)
            cord[(i-1)*length(y) + j, 1] = x[i]
            cord[(i-1)*length(y) + j, 2] = y[j]
        end
    end
    return cord
end

function draw_pelles_boundary(pelle)
    ε = 0.1
    x = [-5:ε:5;]
    y = [-5:ε:5;]
    cord = create_grid(x, y)
    col = zeros(Int, length(cord[:,1]))
    for i in 1:length(cord[:,1])
        col[i] = predict(pelle, cord[i,:])
    end

    scatter(cord[:,1], cord[:,2], color = col)
end

main()
