# The `localization` example is a simple noninteractive example where
#   two players independently try to reach a target point when the 
#   state observations are noisier as the agent departs from the x-axis.
#   In a method that is properly using observations we expect
#   the agent to move towards y=0, then move towards the target.

function make_localization_sensing(targ_1, targ_2; n=2, dt=1.0)

    state = State(
        :p1_obs => 2,
        :p2_obs => 2
    )

    function dyn!(state::StateDist, history, game_params)

        m = length(state)

        res = mapreduce(vcat, 1:n) do _
            rep1 = draw(state)
            map(1:n) do _
                rep2 = draw(state)

                μ_1 = rep1[:p1_pos]
                μ_2 = rep2[:p2_pos]
                σ_1 = (targ_1 - μ_1[2])
                σ_2 = (targ_2 - μ_2[2])

                ob_1 = repeat(sample_gauss.(μ_1, σ_1.^2)', m)
                ob_2 = repeat(sample_gauss.(μ_2, σ_2.^2)', m)


                new_dist = alter(state,
                    :p1_obs => ob_1,
                    :p2_obs => ob_2
                )

                new_ll1 = sum(SensingGames.gauss_ll.(state[:p1_pos], μ_1', σ_1.^2), dims=2)
                new_ll2 = sum(SensingGames.gauss_ll.(state[:p2_pos], μ_2', σ_2.^2), dims=2)

                reweight(new_dist, new_ll1 + new_ll2)
            end 
        end

        # For now until we do branching correctly we just use the first option
        res[1]
    end
    
    state, dyn!
end

function make_localization_cost(targs)
    function cost(state_dist)

        # TODO - this can be vectorized
        c1 = expectation(state_dist) do state
            dist2(state[:p1_pos], targs[1])
        end
        c2 = expectation(state_dist) do state
            dist2(state[:p2_pos], targs[2])
        end
        [c1; c2]
    end
end

function make_localization_prior(zero_state; n=10)
    zero_dist = StateDist(zero_state, n)
    () -> alter(zero_dist,
        :p1_pos => [randn(n)  ( 0.3*ones(n))],
        :p2_pos => [randn(n)  (-0.3*ones(n))]
    )
end

function render_localization(hists)
    for h in hists
        for agent in [:p1, :p2]
            render_traj(h, agent)
            render_obs(h, agent)
        end
    end
end

function test_localization()
    state1, sdyn1 = make_vel_dynamics(:p1; control_scale=1)
    state2, sdyn2 = make_vel_dynamics(:p2; control_scale=1)

    # Observations happen simultaneously
    obs, odyn = make_localization_sensing(0.0, 0.0)

    ctrl1 = make_horizon_control(:p1, :p1_obs, :p1_vel)
    ctrl2 = make_horizon_control(:p2, :p2_obs, :p2_vel)

    zero_state = merge(state1, state2, obs)

    dyn_fns = [odyn, ctrl1, ctrl2, sdyn1, sdyn2]
    prior_fn = make_localization_prior(zero_state)
    cost_fn =  make_localization_cost([[0.0; 2.0], [0.0; -2.0]])

    localization_game = SensingGame(prior_fn, dyn_fns, cost_fn)
    options = (;
        parallel = false,
        n_particles = 1,
        n_lookahead = 5,
        n_render = 1,
        n_iters = 500,
        live = false,
        score_mode = :last
    )



    
    make_model() = Chain(Dense(2 => 2))

    initial_params = (; 
        policies = (;
            p1 = FluxPolicy(make_model, 2; lr=0.04, t_max=options.n_lookahead),
            p2 = FluxPolicy(make_model, 2; lr=0.04, t_max=options.n_lookahead)
        )
    )

    solve_unrelated_particles(localization_game, initial_params, options) do (t, particles, params)
        plt = plot(aspect_ratio=:equal, lims=(-3, 3))
        title!("Localization: t=$t")
        hists = [step(prt, params, n=options.n_lookahead) for prt in particles]
        render_localization(hists)
        display(plt)
    end
end
