

function solve(callback, game::SensingGame, game_params, cost_fns, options)
    seed = abs(rand(Int))

    function score(θ, agent)
        hist = rollout(game, θ, n=options.n_lookahead)
        cost_fns[agent](hist)
    end

    flux_setups = map(game_params) do params
        Flux.setup(Adam(0.01), params)
    end

    for t in 1:options.n_iters

        for (agent, params) in pairs(game_params)
            Random.seed!(seed + t ÷ options.steps_per_seed)
            c, grads = Flux.withgradient(θ -> score(θ, agent), game_params)
            @show c
            Flux.Optimisers.update!(flux_setups[agent], params, grads[1][agent])
        end

        Random.seed!(seed + t ÷ options.steps_per_seed)
        callback(game_params)
    end
end