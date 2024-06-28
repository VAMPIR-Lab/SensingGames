# Partially observable stochastic game solution methods under research.

function solve(callback, game::SensingGame, initial_params, options)
    game_params = initial_params

    # TODO This is misleading - particles are now under a single StateDist
    particles = [clone(game) for _ in 1:options.n_particles]

    function score(θ, ego, threaded)
        sum_fn = threaded ? ThreadsX.sum : Base.sum
        sum_fn(particles) do prt
            dist_hists = step(prt, θ, n=options.n_lookahead)
            game.cost_fn(dist_hists)[ego]
        end / options.n_particles
    end

    r = abs(rand(Int))

    for t in 1:options.n_iters
        
        # sleep(0.08)

        try
            Random.seed!(r + t ÷ options.steps_per_seed)
            c1, g1 = Flux.withgradient(θ -> score(θ, 1, false), game_params)
            Random.seed!(r + t ÷ options.steps_per_seed)
            c2, g2 = Flux.withgradient(θ -> score(θ, 2, false), game_params)
            println("$(c1)")#\t$(c2)")
            apply_gradient!(game_params.policies[:p1], g1[1].policies[:p1])
            # apply_gradient!(game_params.policies[:p2], g2[1].policies[:p2])
        catch e
            if e isa InterruptException
                @warn "Interrupted"
                return
            else
                throw(e)
                # @warn "Gradient failed with $e; attempting to continue..."
                # continue
            end
        end

        if options.live
            # TODO - step! is broken rn
            # for prt in particles
            #     step!(prt, game_params)
            # end
        else
            restart!.(particles)
        end

        Random.seed!(r + t ÷ options.steps_per_seed)
        callback((t, particles, game_params))
    end
end