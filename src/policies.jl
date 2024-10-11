
struct HistPolicy
    subpols::Vector
end

Flux.@layer HistPolicy

function make_nn_control(agent, id_input, id_output; t_max=7)
    function control(dist, game_params)
        # t = Int(dist[:t][1])
        # T = min(t, t_max)
        history = dist[id_input]

        model = game_params[agent]

        in = history
        out = model.subpols[1](transpose(in))'

        action = out[:, 1:2]
        scale = tanh.(out[:, 3])
        scaled_action = scale .* action ./ sqrt.(sum(action .^ 2, dims=2))

        alter(dist, 
            id_output => scaled_action
        )
    end

    GameComponent(control, [id_output])
end

function make_policy(n_input, n_output; t_max=7)
    HistPolicy([
        f64(Chain(
            Dense(n_input*t => 32, relu) |> gpu,
            Dense(32 => 64, relu) |> gpu,
            Dense(64 => 64, relu) |> gpu,
            Dense(64 => 32, relu) |> gpu,
            Dense(32 => n_output) |> gpu
        )) for t in 1:t_max]
    )
end





# Idea: Map [obs] -> [(true state, prob)...]
#   then map prob, [(true state)...] -> prob, [(best response)...]
#   then map [(prob, best response)...] -> action

# The first one we get for free under our current assumptions
#   (which will probably bother the RL people, but that's OK)
#   The second one should be really easy since basically no noise
#   The third one just interpolates according to the cost


# Also these have different needs:
# (1) needs a lot of resampling (if we don't already have it)
# (2) needs branching - otherwise only one possible true state
#     but doesn't care about resampling, aside from gradient play
#     weirdnesses



function make_belief_step(agent, true_id, m; consistent=true)
    info_id = Symbol(agent, "_info")
    belief_id = Symbol(agent, "_blf")

    function belief_step(state_dist, game_params)
        
        
        partitions = if consistent 
            Zygote.ignore() do 
                map(unique(state_dist[info_id])) do infoset
                    vec(state_dist[info_id] .== infoset)
                end
            end
        else
            [[i] for i in 1:length(state_dist)]
        end

        dists = mapreduce(vcat, partitions) do partition

            info_dist = StateDist(
                state_dist.z[partition, :],
                state_dist.w[partition],
                state_dist.ids,
                state_dist.map
            )

            weight = exp.(info_dist.w) ./ sum(exp.(info_dist.w))

            belief = sum(info_dist[true_id] .*  weight, dims=1)

            # belief = vec(info_dist[true_id])


            alter(info_dist,
                belief_id => repeat(belief, length(info_dist))
            )
        end

        StateDist(
            vcat([dist.z for dist in dists]...),
            vcat([dist.w for dist in dists]...),
            dists[begin].ids,
            dists[begin].map
        )
    end

    State(belief_id => m), belief_step
end