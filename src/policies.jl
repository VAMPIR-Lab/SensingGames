
struct HistPolicy
    subpols::Vector
end

Flux.@layer HistPolicy

function make_nn_control(agent, id_input, id_output, n_input, n_output; t_max=5)

    function control(dist, game_params)
        t = Int(dist[:t][1])

        T = min(t, t_max)
        history = dist[id_input]

        model = game_params[agent]


        # out = sum(1:T) do t
        #     in = history[:, (1:n_input) .+ n_input*(t-1)]
        #     model.subpols[t](transpose(in))'
        # end

        in = history[:, begin:(t*n_input)]
        out = model.subpols[t](transpose(in))'

        # action = tanh.(0.01 * out)
        action = out ./ sqrt.(sum(out .^ 2, dims=2))
        # action = 0.5 * (tanh.(0.01 * out[:, 1]) .+ 1.2) .* [cos.(out[:, 2]) sin.(out[:, 2])]

        alter(dist, 
            id_output => action
        )
    end

    models = HistPolicy([
        Chain(
            Dense(n_input*t => 32),
            Dense(32 => 32, relu),
            Dense(32 => 64, relu),
            Dense(64 => 32, relu),
            Dense(32 => 32, relu),
            Dense(32 => n_output)
        ) for t in 1:t_max]
    )
    models, control
end