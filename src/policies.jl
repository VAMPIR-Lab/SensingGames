

struct FluxPolicy <: Policy
    models::Vector{Flux.Chain}
    n_output::Int
    t_max::Int
    flux_setups::Vector
end

function FluxPolicy(make_model, n_output; lr=0.004, t_max=5)
    models = [make_model() for _ in 1:t_max]
    FluxPolicy(
        models, n_output, t_max,
        [Flux.setup(Adam(lr), model) for model in models]
    )
end

function (policy::FluxPolicy)(history)
    hist = last(history, policy.t_max)

    # It's possible that (because of branching)
    #   not every state distribution has the same
    #   number of particles
    h = size(history[end])[1]

    result = sum(enumerate(reverse(hist))) do (t, obs)
        

        # The last index for Flux models is assumed to be batch
        # In StateDists the first index is the batch
        # (because Julia is column major and we 
        # tend to pull state components more frequently
        # than we pull state particles)

        m::Flux.Chain = policy.models[t]
        r = 0.01 * m(obs')'
        # @show h/size(r)[1]
        # @show size(r)
        mapreduce(vcat, 1:(h/size(r)[1])) do _
            r
        end
    end

    # result ./ sqrt.(sum(result.^2, dims=2))
    tanh.(result)
end

function apply_gradient!(policy::FluxPolicy, grads)
    for (i, m) in enumerate(policy.models)
        Flux.update!(policy.flux_setups[i], m, grads.models[i])
    end
end


struct RandomPolicy <: Policy
    n_output
end

function (policy::RandomPolicy)(history)
    0.2 * tanh.(randn(policy.n_output))
end


struct BoundedRandomPolicy <: Policy
    n_output
end

function (policy::BoundedRandomPolicy)(history)
    # assume observation is position
    dir = (rand(2).-0.5) * 20
    dx = history[end] - dir
    -dx / sum(dx .^ 2)
end


# struct ZeroPolicy <: Policy
#     n_output
# end

# function (policy::ZeroPolicy)(history)
#     (zeros(policy.n_output))
# end


function _act(policy::Policy, obs_history)
    # type barrier for optimization
    policy(obs_history)
end

function make_horizon_control(agent::Symbol, ids_obs::Union{Symbol, Vector{Symbol}}, id_action::Symbol)
    function dyn!(state::StateDist, history::Vector{StateDist}, game_params)::StateDist
        
        current_obs = [state[ids_obs]]
        past_obs = map(s -> s[ids_obs], history)
        obs_history = [past_obs; current_obs] 

        action = _act(game_params.policies[agent], obs_history)
        alter(state, 
            id_action => action
        )
    end 
    # no point to returning state components here; there are none
end

