struct SensingGame <: Game
    components
    prior
end

function step(state::State, g::SensingGame, game_params; dt=1)
    for component in g.components
        state = component(state, game_params)
    end
    state
end

function rollout(g::SensingGame, game_params; dt=1, T=10)
    state = nothing # scoping
    Zygote.ignore() do
        state = g.prior()
        for (_, p) in game_params.policies
            reset!(p)
        end
    end

    map(1:T) do t
        state = step(state, g, game_params; dt)
    end
end
