
function solve(callback, game::SensingGame, game_params, cost_fns, options)
    seed = abs(rand(Int))
    global _game_rng = MersenneTwister(seed)

    function score(θ, agent)
        hist = rollout(game, θ, n=options.n_lookahead)
        cost_fns[agent](hist)
    end

    flux_setups = (;
        p1 = Flux.setup(Adam(1e-3), game_params[:p1]),
        p2 = Flux.setup(Adam(1e-4), game_params[:p2]),
    )

    for t in 1:options.n_iters

        for (agent, params) in pairs(game_params)

            reseed!(seed + t ÷ options.steps_per_seed)
            c, grads = Flux.withgradient(θ -> score(θ, agent), game_params)
            Flux.Optimisers.update!(flux_setups[agent], params, grads[1][agent])
        end

        reseed!(seed + t ÷ options.steps_per_seed)
        if callback(game_params)
            println("DONE ==============")
            return game_params
        end
    end

    return game_params
end

# TODO: This is very sloppy
function reseed!(seed)
    Random.seed!(_game_rng, seed)
end