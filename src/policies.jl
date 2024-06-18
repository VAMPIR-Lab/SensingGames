

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
        [Flux.setup(Flux.Adam(lr), model) for model in models]
    )
end

function (policy::FluxPolicy)(history::Vector{Vector{Float32}})
    result::Vector{Float32} = zeros(policy.n_output)
    hist = last(history, policy.t_max)

    for (t, obs) in enumerate(reverse(hist))
        m::Flux.Chain = policy.models[t]
        output::Vector{Float32} = m(obs)
        result += 0.01 * output
    end

    # p = sig(result[1])
    # softif(rand() - p, tanh.(result[2:3]), tanh.(result[4:5]))
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


struct ZeroPolicy <: Policy
    n_output
end

function (policy::ZeroPolicy)(history)
    (zeros(policy.n_output))
end


function _act(policy::Policy, obs_history::Vector{Vector{Float32}})::Vector{Float32}
    # type barrier for optimization
    policy(obs_history)
end

function make_horizon_control(agent::Symbol, id_obs::Symbol, id_action::Symbol)
    function dyn!(state::State, history::Vector{State}, game_params)::State
        current_obs::Vector{Vector{Float32}} = [state[id_obs]]
        past_obs::Vector{Vector{Float32}} = map(s -> s[id_obs], history)
        obs_history::Vector{Vector{Float32}} = [past_obs; current_obs] 

        action::Vector{Float32} = _act(game_params.policies[agent], obs_history)
        alter(state, 
            id_action => action
        )
    end 
    # no point to returning state components here; there are none
end

