
function solve(callback, game::ContinuousGame, prior_belief, game_params, cost_fns, options)
    seed = abs(rand(Int))
    global _game_rng = MersenneTwister(seed)

    function score(θ, agent)
        # temp = draw(prior_belief; n=options.batch_size)
        hist = step(game, draw(prior_belief; n=options.batch_size) |> gpu, θ, n=options.n_lookahead)
        cost_fns[agent](hist)
    end

    flux_setups = (; (k => Flux.setup(Adam(5e-4), game_params[k]) for (k, v) in pairs(cost_fns))...)
    
    for t in 1:options.n_iters

        for (agent, params) in pairs(game_params)

            reseed!(seed + t ÷ options.steps_per_seed)
            c, grads = Flux.withgradient(θ -> score(θ, agent), game_params)
            Flux.Optimisers.update!(flux_setups[agent], params, grads[1][agent])
        end

        reseed!(seed + t ÷ options.steps_per_seed)
        if callback(game_params)
            return game_params
        end
    end

    return game_params
end

function solve(game::ContinuousGame, initial_dist, game_params, cost_fns, options)
    solve((_) -> false, game, initial_dist, game_params, cost_fns, options)
end

# TODO: This is very sloppy
function reseed!(seed)
    Random.seed!(_game_rng, seed)
end