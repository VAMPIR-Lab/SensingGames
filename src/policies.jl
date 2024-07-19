
struct HistPolicy
    subpols::Vector
end

Flux.@layer HistPolicy

function make_nn_control(agent, id_input, id_output, n_input, n_output; t_max=7)

    function control(dist, game_params)
        t = Int(dist[:t][1])
        T = min(t, t_max)
        history = dist[id_input]

        model = game_params[agent]

        # out = sum(1:T) do t
        #     in = history[:, (1:n_input) .+ n_input*(t-1)]
        #     model.subpols[t](transpose(in))'
        # end
        in = history[:, begin:(T*n_input)]
        out = model.subpols[T](transpose(in))'

        # action = tanh.(0.01 * out)
        # action = out ./ sqrt.(sum(out .^ 2, dims=2))
        θ = out[:, 2] 
        action = 0.5 * (tanh.(out[:, 1]) .+ 1.2) .* [cos.(θ) sin.(θ)]

        alter(dist, 
            id_output => action
        )
    end

    models = HistPolicy([
        f64(Chain(
            Dense(n_input*t => 32, relu),
            Dense(32 => 32, tanh),
            Dense(32 => 32, relu),
            Dense(32 => n_output)
        )) for t in 1:t_max]
    )
    models, control
end



function make_weighted_control(agent, id_input, id_output, n_input, n_output; t_max=7)
    info_id = Symbol(agent, "_info")

    br_models = HistPolicy([
        Chain(
            Dense(n_input => 32, relu),
            Dense(32 => 32, relu),
            Dense(32 => 32, relu),
            Dense(32 => n_output)
        )]
    )

    merge_model = HistPolicy([
        Chain(
            Dense(n_input => 32, relu),
            Dense(32 => 64, relu),
            Dense(64 => 32, relu),
            Dense(32 => n_output)
        )]
    )

    function control(state_dist, game_params)
        t = Int(state_dist[:t][1])
        T = min(t, t_max)
        
        infosets = Zygote.ignore() do 
            unique(state_dist[info_id])
        end

        dists = mapreduce(vcat, infosets) do infoset
            rows = vec(state_dist[info_id] .== infoset)

            info_dist = StateDist(
                state_dist.z[rows, :],
                state_dist.w[rows],
                state_dist.ids,
                state_dist.map
            )

            # br_input = transpose()
            # best_responses = game_params[agent].br.subpols[1](br_input)
            total_weight = sum(exp.(info_dist.w))

            input = expectation(info_dist) do state
                state[id_input]
            end / total_weight

            # merge_input = [d; w]

            out = game_params[agent].merge.subpols[1](input)'
            # out = best_responses[:, 1]'

            θ = out[:, 2] 
            action = 0.5 * (tanh.(out[:, 1]) .+ 1.2) .* [cos.(θ) sin.(θ)]

            alter(info_dist,
                id_output => repeat(action, length(info_dist))
            )
        end

        StateDist(
            vcat([dist.z for dist in dists]...),
            vcat([dist.w for dist in dists]...),
            dists[begin].ids,
            dists[begin].map
        )
    end

    (; br=br_models, merge=merge_model), control
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



function make_belief_step(agent, true_id, m)
    info_id = Symbol(agent, "_info")
    belief_id = Symbol(agent, "_blf")

    function belief_step(state_dist, game_params)
        infosets = Zygote.ignore() do 
            unique(state_dist[info_id])
        end

        dists = mapreduce(vcat, infosets) do infoset
            rows = vec(state_dist[info_id] .== infoset)

            info_dist = StateDist(
                state_dist.z[rows, :],
                state_dist.w[rows],
                state_dist.ids,
                state_dist.map
            )
            total_weight = sum(exp.(info_dist.w))

            belief = expectation(info_dist) do state
                state[true_id]
            end / total_weight

            alter(info_dist,
                belief_id => repeat(belief', length(info_dist))
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