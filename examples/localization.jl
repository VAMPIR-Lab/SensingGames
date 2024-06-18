# The `localization` example is a simple noninteractive example where
#   two players independently try to reach a target point when the 
#   state observations are noisier as the agent departs from the x-axis.
#   In a method that is properly using observations we expect
#   the agent to move towards y=0, then move towards the target.

function make_localization_sensing(agent::Symbol, targ; dt=1.0)
    id_obs = Symbol("$(agent)_obs")
    id_pos = Symbol("$(agent)_pos")

    state = State(
        id_obs => 2
    )

    function dyn!(state, history, game_params)
        pos = state[id_pos]
        σ2 = 4*(targ - pos[2])^2
        x1 = sample_gauss(pos[1], σ2)
        x2 = sample_gauss(pos[2], σ2)

        alter(state,
            id_obs => [x1; x2]
        )
    end
    
    state, dyn!
end

function make_localization_cost(targs)
    function cost(state)
        [
            dist2(state[:p1_pos], targs[1])
            dist2(state[:p2_pos], targs[2])
        ]
    end
end

function make_localization_prior(zero_state)
    () -> alter(zero_state,
        :p1_pos => [randn();  0.3],
        :p2_pos => [randn(); -0.3]
    ) 
end

function render_localization(hists)
    for h in hists
        for agent in [:p1, :p2]
            render_traj(h, agent)
        end
    end
end

function test_localization()
    state1, sdyn1 = make_vel_dynamics(:p1; control_scale=1)
    state2, sdyn2 = make_vel_dynamics(:p2; control_scale=1)

    obs1, odyn1 = make_localization_sensing(:p1, 0.0)
    obs2, odyn2 = make_localization_sensing(:p2, 0.0)

    ctrl1 = make_horizon_control(:p1, :p1_obs, :p1_vel)
    ctrl2 = make_horizon_control(:p2, :p2_obs, :p2_vel)

    zero_state = merge(state1, state2, obs1, obs2)

    dyn_fns = [odyn1, odyn2, ctrl1, ctrl2, sdyn1, sdyn2]
    prior_fn = make_localization_prior(zero_state)
    cost_fn =  make_localization_cost([[0.0; 2.0], [0.0; -2.0]])

    localization_game = SensingGame(prior_fn, dyn_fns, cost_fn)
    options = (;
        parallel = true,
        n_particles = 5,
        n_lookahead = 5,
        n_render = 1,
        n_iters = 500,
        live = false,
        score_mode = :last
    )

    
    make_model() = Chain(Dense(2 => 2))

    initial_params = (; 
        policies = (;
            p1 = FluxPolicy(make_model, 2; lr=0.4, t_max=options.n_lookahead),
            p2 = FluxPolicy(make_model, 2; lr=0.4, t_max=options.n_lookahead)
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
