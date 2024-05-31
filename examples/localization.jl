
function make_localization_sensing(agent::Symbol, targ; dt=1.0)
    id_obs = Symbol("$(agent)_obs")
    id_pos = Symbol("$(agent)_pos")

    state = State(
        id_obs => 2
    )

    function dyn!(state, history, game_params)
        pos = state[id_pos]
        σ2 = 4*(targ - pos[2])^2
        x1, _ = sample_gauss(pos[1], σ2)
        x2, _ = sample_gauss(pos[2], σ2)

        alter(state,
            id_obs => [x1; x2]
        )
    end
    
    state, dyn!
end

function make_localization_cost(agent::Symbol, targ)
    id_pos = Symbol("$(agent)_pos")

    function cost(state)
        dist2(state[id_pos], targ)
    end
end


function render_localization_game(states)
    state_x1 = [s[:p1_pos][1] for s in states]
    state_y1 = [s[:p1_pos][2] for s in states]
    state_x2 = [s[:p2_pos][1] for s in states]
    state_y2 = [s[:p2_pos][2] for s in states]

    plot!(state_x1, state_y1,
        color=:blue,
        alpha=0.3,
        label=""
    )

    plot!(state_x1, state_y1,
        seriestype=:scatter,
        color=:blue,
        alpha=0.3,
        label=""
    )

    plot!(state_x2, state_y2,
        color=:red,
        alpha=0.3,
        label=""
    )

    plot!(state_x2, state_y2,
        seriestype=:scatter,
        color=:red,
        alpha=0.3,
        label=""
    )
end

function test_localization_game()

    @info "Making dynamics"
    state1, sdyn1 = make_vel_dynamics(:p1; control_scale=1.0)
    state2, sdyn2 = make_vel_dynamics(:p2; control_scale=2.0)

    obs1, odyn1 = make_localization_sensing(:p1, 0)
    obs2, odyn2 = make_localization_sensing(:p2, 0)

    ctrl1 = make_horizon_control(:p1, :p1_obs, :p1_vel)
    ctrl2 = make_horizon_control(:p2, :p2_obs, :p2_vel)

    initial_state = merge(state1, state2, obs1, obs2)

    game_params = (; 
        policies = (;
            p1 = LinearPolicy(2, 2; t_max=5),
            p2 = LinearPolicy(2, 2, t_max=5)
        )
    )

    prior = () -> alter(initial_state,
        :p1_pos => [randn();  0.3],
        :p2_pos => [randn(); -0.3]
    ) 

    particles = []
    components = [odyn1, odyn2, ctrl1, ctrl2, sdyn1, sdyn2]
    n_particles = 1
    n_samples = 1
    n_lookahead = 5
    resample_freq=100

    @info "Generating particles and setting costs"
    particles = [SensingGame(components, prior()) for _ in 1:n_particles]

    cost_fn_1 = make_localization_cost(:p1, [0, 2])
    cost_fn_2 = make_localization_cost(:p2, [0, -2])

    costs_1 = [0.0]
    costs_2 = [0.0]

    @info "Beginning descent"
    for t in 1:10000

        # Take gradients (before we modify any particles)
        if t == 1 @info "Taking gradients" end
        score1 = make_score(cost_fn_1, particles;    
            n_samples, n_lookahead, parallel=(t!=1))
        score2 = make_score(cost_fn_2, particles; 
            n_samples, n_lookahead, parallel=(t!=1))

        c1 = c2 = g1 = g2 = nothing
        try
            c1, g1 = Flux.withgradient(score1, game_params)
            c2, g2 = Flux.withgradient(score2, game_params)
        catch e
            # throw(e)
            if e isa InterruptException
                @info "Stopping (interrupted)"
                sleep(1)
                return
            end
            @warn "Gradient failed with $e; attempting to continue..."
            continue
        end

        n = min(100, length(costs_1))
        push!(costs_1, c1)
        push!(costs_2, c2)
        println("t=$(t)\tc1=$(sum(costs_1[end-n+1:end])/n)\tc2=$(sum(costs_2[end-n+1:end])/n)")

        if t == 1 @info "Applying gradients" end
        apply_gradient!(game_params.policies[:p1], g1[1].policies[:p1])
        apply_gradient!(game_params.policies[:p2], g2[1].policies[:p2])

        # Take the actual step (we don't do the gradient on this,
        #   although technically we could I think?)
        if t == 1 @info "Resetting particles" end
        if t % resample_freq == 0
            particles = [SensingGame(components, prior()) for _ in 1:n_particles]
        else
            particles = [SensingGame(components, prior()) for _ in 1:n_particles]
        end

        if t == 1 @info "Plotting" end
        plt = plot(aspect_ratio=:equal, lims=(-3, 3))
        for particle in particles[1:((t>1000) ? 1 : n_particles)]
            render_localization_game(step(particle, game_params, n=5))
        end
        display(plt)

        if t == 1 @info "Iterating..." end
    end
end
