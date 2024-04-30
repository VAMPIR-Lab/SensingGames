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

    state, ll_state = game.prior()
    states = reshape(state, size(state)..., 1)

    ob, ll_ob = game.sensor_dynamics(state)
    ob_shape = size(ob)
    obs = reshape(ob, ob_shape..., 1)

    act, ll_act = game.init_act, 0.0
    act_shape = size(act)
    acts = reshape(act, act_shape..., 1)


    log_lik = ll_state + ll_ob + ll_act
    for t ∈ 1:game.t_max
        state, ll_state = game.state_dynamics(state, act)
        states = cat(states, state; dims=ndims(state)+1)

        ob, ll_ob = game.sensor_dynamics(state)
        obs = cat(obs, ob; dims=length(ob_shape)+1)

        history_length = min(game.max_obs_used, t)
        act_obs = selectdim(obs, ndims(obs), 
            (size(obs)[end]-history_length+1) : size(obs)[end])
        act_acts = selectdim(acts, ndims(acts), 
            (size(acts)[end]-history_length+1) : size(acts)[end])
        act, ll_act = policy(act_obs)

        acts = cat(acts, act; dims=ndims(act)+1)

        log_lik += ll_state + ll_ob + ll_act
        # println("=======")
        # println(ll_state)
        # println(ll_ob)
        # println(ll_act)
    end
    (; states, obs, acts, log_lik)
end

function tree_rollout(game::SensingGame, policy::Policy; hist=nothing, shapes=nothing, bf=2, t=1)
    
    
    if t > game.t_max
        return [hist]
    end

    if isnothing(hist)
        reset!(policy)
        state, ll_state = game.prior()
        states = reshape(state, size(state)..., 1)

        ob, ll_ob = game.sensor_dynamics(state)
        ob_shape = size(ob)
        obs = reshape(ob, ob_shape..., 1)

        act, ll_act = game.init_act, 0.0
        act_shape = size(act)
        acts = reshape(act, act_shape..., 1)
        log_lik = ll_state + ll_ob + ll_act

        hist = (; states, obs, acts, log_lik)
        shapes = (; act_shape, ob_shape)
    end

    return mapreduce(vcat, 1:bf) do i
        _sample_rest(game, policy, hist, shapes, t, bf)
    end
end

function _sample_rest(game, policy, hist, shapes, t, bf)
    states, obs, acts, log_lik = hist
    act_shape, ob_shape = shapes

    cur_state = selectdim(states, ndims(states), size(states)[end])
    cur_ob    = selectdim(obs,    ndims(obs),    size(obs)[end])
    cur_act   = selectdim(acts,   ndims(acts),   size(acts)[end])

    new_state, ll_state = game.state_dynamics(cur_state, cur_act)
    new_states = cat(states, new_state; dims=ndims(new_state)+1)

    new_ob, ll_ob = game.sensor_dynamics(new_state)
    new_obs = cat(obs, new_ob; dims=length(ob_shape)+1)

    history_length = min(game.max_obs_used, t)
    act_obs = selectdim(new_obs, ndims(new_obs), 
        (size(new_obs)[end]-history_length+1) : size(new_obs)[end])
    new_act, ll_act = policy(act_obs)
    new_acts = cat(acts, new_act; dims=ndims(new_act)+1)

    new_log_lik = log_lik + ll_state + ll_ob + ll_act
    tree_rollout(
        game, policy; 
        hist = (; states=new_states, obs=new_obs, acts=new_acts, log_lik = new_log_lik), 
        shapes = (; act_shape, ob_shape),
        bf, t=t+1
    )
end

function evaluate(game::SensingGame, pol::Policy; n=1, k=0.05)
    hists = map(1:n) do _
        rollout(game, pol)
    end

    probs = map(h -> exp(h.log_lik), hists)
    norm_probs = probs ./ sum(probs)

    mapreduce(+, zip(probs, norm_probs, hists)) do (p, np, h)
        # (game.cost(h) + act_reg(h.acts)) #* p
        # -k*log(p) + act_reg(h.acts) + game.cost(h)
        game.cost(h)
    end / n
end

function tree_evaluate(game::SensingGame, pol::Policy; n=1, k=0.05)
    hists = mapreduce(vcat, 1:n) do _
        tree_rollout(game, pol)
    end
    probs = map(h -> exp(h.log_lik), hists)
    norm_probs = probs ./ sum(probs)

    mapreduce(+, zip(probs, norm_probs, hists)) do (p, np, h)
        # (game.cost(h) + act_reg(h.acts)) #* p
        # -k*log(p) + act_reg(h.acts) + game.cost(h)
        game.cost(h)
    end
end




function act_reg(act_hist; α=2)
    α * mapreduce(+, eachslice(act_hist; dims=length(size(act_hist)))) do a
        sum(a .^ 2)
    end
end