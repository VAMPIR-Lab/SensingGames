# Partially observable stochastic game solution methods under research.

# "Unrelated particles" maintains a bunch of particles which are all essentially
#   unique instances of the game with unique histories. A single step consists of simulating
#   all particles forward T steps, taking a gradient over the resulting cost, and (optionally)
#   actually updating the particles forward a single step. (In some games we don't want to do this,
#   instead sampling from the prior every time.)
#
# You can supply a callback (i.e. with a `do` block); it will receive (t, particles, params)
#   on every step to be used for output display or rendering.   
function solve_unrelated_particles(game::SensingGame, initial_params, options, get_hists, vis_options, render_game!)
    game_params = initial_params
    particles = [clone(game) for _ in 1:options.n_particles]

    function score(θ, ego, threaded)
        sum_fn = threaded ? ThreadsX.sum : Base.sum
        sum_fn(particles) do prt
            states = step(prt, θ, n=options.n_lookahead)
            if options.score_mode == :sum
                Base.sum(states) do s
                    game.cost_fn(s)[ego]
                end / length(states)
            else
                game.cost_fn(states[end])[ego]
            end
        end / options.n_particles
    end
    
    fig, ax, iter = init_window(vis_options)
    
    for t in 1:options.n_iters
        
        if !Base.isopen(fig.scene)
            println("Window closed. Program terminated.")
            break
        end
        try
            c1, g1 = Flux.withgradient(θ -> score(θ, 1, false), game_params)
            c1, g2 = Flux.withgradient(θ -> score(θ, 2, false), game_params)
            apply_gradient!(game_params.policies[:p1], g1[1].policies[:p1])
            apply_gradient!(game_params.policies[:p2], g2[1].policies[:p2])
        catch e
            if isa(err, InterruptException)
                @warn "Interrpted..."
                return
            else
                @warn "Gradient failed with $e; attempting to continue..."
                continue
            end
        end

        if options.live
            for prt in particles
                step!(prt, game_params)
            end
        end
        
        # callback((t, particles, game_params))
        hists = get_hists(options, particles, game_params)
        render_game!(ax, hists)
        sleep(0.01)
        empty!(ax)
        iter[] = t
    end
    
end