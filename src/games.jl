struct SensingGame <: Game
    prior_fn::Function
    dyn_fns::Vector{Function}
    cost_fn::Function

    history::Vector{StateDist}
    history_len::Int
end

function SensingGame(prior_fn::Function, dyn_fns::Vector{Function}, cost_fn::Function; history_len=100)
    SensingGame(prior_fn, dyn_fns, cost_fn, [], history_len)
end

function restart!(g::SensingGame)
    empty!(g.history)
end

function update!(g::SensingGame, states)
    for s in states
        roll!(g.history, s, g.history_len)
    end
    states
end

function step(g::SensingGame, game_params; n=1)
    @bp

    state = isempty(g.history) ? g.prior_fn() : g.history[end]
    res::Vector{StateDist} = [state]

    for t in 1:n
        for dyn_fn in g.dyn_fns
            h = [g.history[begin:end]; res]
            state = dyn_fn(state, h, game_params)
        end
        res = [res; state]
    end
    res
end

function step!(g::SensingGame, game_params; n=1)
    update!(g, step(g, game_params; n))
end

function clone(g::SensingGame)
    SensingGame(g.prior_fn, g.dyn_fns, g.cost_fn; history_len=g.history_len)
end