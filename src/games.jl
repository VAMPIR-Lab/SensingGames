struct SensingGame <: Game
    components::Vector{Function}
    history::Vector{State}
    history_len
end

function SensingGame(components, initial_state::State; history_len=100)
    SensingGame(components, [initial_state], history_len)
end

function step(g::SensingGame, game_params; n=1)
    state::State = g.history[end]
    res::Vector{State} = []

    for t in 1:n
        for component in g.components
            h = [g.history[begin:end]; res]
            state = component(state, h, game_params)
        end
        res = [res; state]
    end
    res
end

function update!(g::SensingGame, states)
    for s in states
        roll!(g.history, s, g.history_len)
    end
end