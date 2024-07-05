# The `localization` example is a simple noninteractive example where
#   two players independently try to reach a target point when the 
#   state observations are noisier as the agent departs from the x-axis.
#   In a method that is properly using observations we expect
#   the agent to move towards y=0, then move towards the target.

using Random

function make_localization_sensing(ego, targ, n; dt=1.0)
    pos_id = Symbol(ego, "_pos")
    obs_id = Symbol(ego, "_obs")

    state = State(
        obs_id => 2,
    )

    function sensing_dyn(state_dist::StateDist, history, game_params)
        μ = state_dist[pos_id]
        σ2 = 0.1 .+ 4*(targ .- μ[:, 2]).^2
        obs = sample_gauss.(μ, σ2)
        alter(state_dist, 
            obs_id => obs
        )
    end

    function sensing_lik(state_dist::StateDist, compare_dist::StateDist)
        μ = state_dist[pos_id]
        σ2 = 0.1 .+ 4*(targ .- μ[:, 2]).^2
        obs = reshape(compare_dist[obs_id], (1, 2, :))
        sum(SensingGames.gauss_logpdf.(obs, μ, σ2), dims=2)
    end

    state, make_cross_step(sensing_dyn, sensing_lik, obs_id, n)
end

function make_localization_cost(targs)
    function cost(dist_hists)
        sum(dist_hists) do hist
            state_dist = hist[end]
            expectation(state_dist) do state
                [dist2(state[:p1_pos], targs[1]);
                dist2(state[:p2_pos], targs[2])]
            end
        end
    end
end

function make_localization_prior(zero_state; n=2)
    zero_dist = StateDist(zero_state, n)
    dist = alter(zero_dist,
        :p1_pos => [randn(n)  ( 0.5*ones(n))],
        :p2_pos => [randn(n)  (-0.5*ones(n))]
    )
    
    function prior() 
        dist
    end
end

function render_localization(hists)
    for h in hists
        for agent in [:p1, :p2]
            render_traj(h, agent)
            # render_obs(h, agent)
        end
    end
end

function test_localization()
    state1, sdyn1 = make_acc_dynamics(:p1; control_scale=1)
    state2, sdyn2 = make_acc_dynamics(:p2; control_scale=1)

    # Observations happen simultaneously
    obs1, odyn1 = make_localization_sensing(:p1, 0.0, [2; 2; 1; 1; 1])
    obs2, odyn2 = make_localization_sensing(:p2, 0.0, 1)

    ctrl1 = make_horizon_control(:p1, :p1_obs, :p1_acc)
    ctrl2 = make_horizon_control(:p2, :p2_obs, :p2_acc)

    zero_state = merge(state1, state2, obs1, obs2)

    dyn_fns = [odyn1, odyn2, ctrl1, ctrl2, sdyn1, sdyn2]
    prior_fn = make_localization_prior(zero_state)
    cost_fn =  make_localization_cost([[0.0; 2.0], [0.0; -2.0]])

    localization_game = SensingGame(prior_fn, dyn_fns, cost_fn)
    options = (;
        parallel = false,
        n_particles = 1, # TODO - this is misleading under StateDists
        n_lookahead = 5,
        n_render = 1,
        n_iters = 500,
        live = false,
        score_mode = :last,
        steps_per_seed = 50
    )


    make_model() = Chain(
        Dense( 2 => 32, relu),
        Dense(32 => 32, relu),
        Dense(32 => 2)
    )
    
    # make_model() = Chain(Dense(2 => 2))

    initial_params = (; 
        policies = (;
            p1 = FluxPolicy(make_model, 2; lr=0.001, t_max=options.n_lookahead),
            p2 = FluxPolicy(make_model, 2; lr=0.001, t_max=options.n_lookahead)
        )
    )

    solve(localization_game, initial_params, options) do (t, particles, params)
        plt = plot(aspect_ratio=:equal, lims=(-3, 3))
        title!("Localization: t=$t")
        dists = rollout(localization_game, params, n=options.n_lookahead)
        render_localization(dists)
        display(plt)
    end
end
