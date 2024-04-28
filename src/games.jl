struct SensingGame
    prior
    state_dynamics
    sensor_dynamics
    cost
    init_act

    t_max
    max_obs_used
end

function SensingGame(prior, state_dyn, sense_dyn, cost, init_act; t_max=10, max_obs_used=8)
    SensingGame(
        prior, state_dyn, sense_dyn, cost, init_act, t_max, max_obs_used
    )
end


function rollout(game::SensingGame, policy::Policy)
    reset!(policy)

    state, p_state = game.prior()
    states = reshape(state, size(state)..., 1)

    ob, p_ob = game.sensor_dynamics(state)
    ob_shape = size(ob)
    obs = reshape(ob, ob_shape..., 1)

    act, p_act = game.init_act, 1.0
    act_shape = size(act)
    acts = reshape(act, act_shape..., 1)

    prob = p_state * p_ob * p_act
    
    for t ∈ 1:game.t_max
        state, p_state = game.state_dynamics(state, act)
        states = cat(states, state; dims=ndims(state)+1)

        ob, p_ob = game.sensor_dynamics(state)
        obs = cat(obs, ob; dims=length(ob_shape)+1)

        history_length = min(game.max_obs_used, t)
        act_obs = selectdim(obs, ndims(obs), 
            (size(obs)[end]-history_length+1) : size(obs)[end])
        act_acts = selectdim(acts, ndims(acts), 
            (size(acts)[end]-history_length+1) : size(acts)[end])
        act, p_act = policy(act_obs)

        acts = cat(acts, act; dims=ndims(act)+1)

        prob = p_state * p_ob * p_act
    end
    (; states, obs, acts, prob)
end
