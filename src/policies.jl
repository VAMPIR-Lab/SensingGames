
mutable struct LinearPolicy <: Policy
    models
    history
    n_input
    n_output
    t_max
    flux_setups
end

function LinearPolicy(n_input, n_output; optimizer=Flux.Adam(0.7), t_max=5)
    models = [Dense(n_input => n_output) for _ in 1:t_max]
    LinearPolicy(
        models, nothing, n_input, n_output, t_max,
        [Flux.setup(optimizer, model) for model in models]
    )
end

function (policy::LinearPolicy)(obs)
    if isnothing(policy.history)
        policy.history = reshape(obs, (length(obs), 1))
    end
    policy.history = cat(policy.history, obs, dims=2)
    T = min(policy.t_max, size(policy.history)[2])

    H = eachslice(policy.history[:, end-T+1:end], dims=2)

    result = zeros(policy.n_output)
    t = 1
    for (t, obs) in enumerate(H)
        result += 0.01 * policy.models[t](Float32.(obs))
    end
    result
end

# function apply_gradient(policy::LinearPolicy, grads)
#     function loss_fn(pol) 
#         pols = [other_policies[begin:pnum-1]; pol; other_policies[pnum+1:end]]
#         evaluate(game, pols, pnum; n)
#     end
#     loss, grads = Flux.withgradient(loss_fn, policy) 

#     for (i, m) in enumerate(policy.models)
#         Flux.update!(policy.flux_setups[i], m, grads[1].models[i])
#     end
#     loss
# end


function apply_gradient!(policy::LinearPolicy, grads)
    for (i, m) in enumerate(policy.models)
        Flux.update!(policy.flux_setups[i], m, grads.models[i])
    end
end

function reset!(policy::Policy)
    policy.history = nothing
end

function make_horizon_control(agent::Symbol, id_obs::Symbol, id_action::Symbol)
    function dyn!(state, game_params)
        alter(state, 
            id_action => game_params.policies[agent](state[id_obs])
        )
    end 
    # no point to returning state components here; there are none
end