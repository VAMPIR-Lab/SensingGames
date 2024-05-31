

struct LinearPolicy <: Policy
    models::Vector{Flux.Chain}
    n_input::Int
    n_output::Int
    t_max::Int
    flux_setups::Vector
end

function LinearPolicy(n_input, n_output; optimizer=Flux.Adam(0.004), t_max=5)
    models = [
        Chain(
            # Dense(n_input => n_output),
            Dense(n_input => 64, relu),
            Dense(64 => 64, relu),
            Dense(64 => n_output)
        ) for _ in 1:t_max
    ]
    LinearPolicy(
        models, n_input, n_output, t_max,
        [Flux.setup(optimizer, model) for model in models]
    )
end

function (policy::LinearPolicy)(history::Vector{Vector{Float32}})
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

function apply_gradient!(policy::LinearPolicy, grads)
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

function make_score(cost_fn, particles; 
    n_samples=1, n_lookahead=2, parallel=false, mode=:sum)
    sum_fn = parallel ? ThreadsX.sum : Base.sum
    function score(θ)
        sum_fn(particles) do prt
            Base.sum(1:n_samples) do _
                states = step(prt, θ, n=n_lookahead)
                Base.sum(states) do s
                    cost_fn(s)
                end / length(states)
            end / n_samples
        end / length(particles)
    end
end