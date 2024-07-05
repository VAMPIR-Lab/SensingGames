# `meetup` is a pursuit-pursuit game where two players
#   both want to be near each other, but must first locate their
#   partner by going to a specific point.

# Framework has changed a lot since this was useful - 
#   to be refactored when it becomes relevant again

function make_meetup_sensing(agent, other, targs)
    
    id_obs = Symbol("$(agent)_obs")
    id_own_pos = Symbol("$(agent)_pos")
    id_other_pos = Symbol("$(other)_pos")

    state = State(
        id_obs => 4
    )
    
    function dyn!(state, history, game_params)
        d = [dist2(state[id_own_pos], beacon) for beacon in targs]
        σ2 = 5*minimum(d)
        us = state[id_own_pos]
        them = [
            sample_gauss(state[id_other_pos][1], σ2)[1]
            sample_gauss(state[id_other_pos][2], σ2)[1]
        ]

        alter(state,
            id_obs => [us; them]
        )
    end

    state, dyn!
end

function make_meetup_cost(agent::Symbol, partner::Symbol)
    id_us = Symbol("$(agent)_pos")
    id_them = Symbol("$(partner)_pos")

    function cost(state)
        [
            sqrt(dist2(state[id_us], state[id_them]))
            sqrt(dist2(state[id_us], state[id_them]))
        ]
        
    end
end

function make_meetup_prior(zero_state)
    () -> alter(zero_state,
        :p1_pos => [-1.0 + randn();  1.5 + randn()],
        :p2_pos => [ 2.0 + randn(),  0.5 + randn()]
    ) 
end

function render_meetup!(ax, hists)
    render_target!(ax, [
        [[-1; -1],],
        [[ 1;  1],]
    ])
    for h in hists
        for agent in [:p1, :p2]
            render_traj!(ax, h, agent)
        end
    end
end

function get_meetup_hists(options, particles, params)
    [step(prt, params, n=options.n_lookahead) for prt in particles]
end

function test_meetup_game()

    targs = [
        [[-1; -1],],
        [[ 1;  1],]
    ]
    
    state1, sdyn1 = make_vel_dynamics(:p1)
    state2, sdyn2 = make_vel_dynamics(:p2)

    obs1, odyn1 = make_meetup_sensing(:p1, :p2, targs[1])
    obs2, odyn2 = make_meetup_sensing(:p2, :p1, targs[2])

    ctrl1 = make_horizon_control(:p1, :p1_obs, :p1_vel)
    ctrl2 = make_horizon_control(:p2, :p2_obs, :p2_vel)

    zero_state = merge(state1, state2, obs1, obs2)

    prior_fn = make_meetup_prior(zero_state)
    dyn_fns = [odyn1, odyn2, ctrl1, ctrl2, sdyn1, sdyn2]
    cost_fn = make_meetup_cost(:p1, :p2)

    meetup_game = SensingGame(prior_fn, dyn_fns, cost_fn)
    options = (;
        parallel = false,
        n_particles = 5,
        n_lookahead = 8,
        n_render = 1,
        n_iters = 3500,
        live = false,
        score_mode = :last
    )

    make_model() = Chain(Dense(4 => 2))

    initial_params = (; 
        policies = (;
            p1 = FluxPolicy(make_model, 2; lr=0.4, t_max=options.n_lookahead),
            p2 = FluxPolicy(make_model, 2; lr=0.4, t_max=options.n_lookahead)
        )
    )

    vis_options = (;
        name = "Meetup",
        win_size = (800, 600), 
        ax_lims = ((-3, 3), (-3, 3))    
    )

    # solve_unrelated_particles(meetup_game, initial_params, options) do (t, particles, params)
    #     plt = plot(aspect_ratio=:equal, lims=(-3, 3))
    #     title!("Meetup: t=$t")
    #     hists = [step(prt, params, n=options.n_lookahead) for prt in particles]
        
    #     render_meetup(hists)
    #     display(plt)
    # end
    solve_unrelated_particles(meetup_game, initial_params, options, get_meetup_hists, vis_options, render_meetup!)
end