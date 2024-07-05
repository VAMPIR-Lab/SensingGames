
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

        action = sum(1:T) do t
            in = history[:, (1:n_input) .+ n_input*(t-1)]
            model.subpols[t](transpose(in))'
        end

        in = dist[Symbol(agent, :_pos)]'

        alter(dist, 
            id_output => tanh.(0.01 * action)
        )
    end

    models = HistPolicy([
        Chain(
            Dense(n_input => 16),
            Dense(16 => 32, relu),
            Dense(32 => n_output)
        ) for _ in 1:t_max]
    )
    models, control
end