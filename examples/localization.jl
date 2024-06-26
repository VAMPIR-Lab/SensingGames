# The `localization` example is a simple noninteractive example where
#   two players independently try to reach a target point when the 
#   state observations are noisier as the agent departs from the x-axis.
#   In a method that is properly using observations we expect
#   the agent to move towards y=0, then move towards the target.

using Random

function make_localization_sensing(targ_1, targ_2, n; dt=1.0)

    state = State(
        :p1_obs => 2,
        :p2_obs => 2
    )

    function dyn!(state_dist::StateDist, history, game_params)

        t = length(history)
        nd = [2; 4; 2; 2; 2]

        m = length(state_dist)
        reps_1 = draw(state_dist; n=nd[t])
        reps_2 = draw(state_dist; n=1)
        # TODO DRY this

        obs1 = map(reps_1) do rep
            μ_1_rep = rep[:p1_pos]
            σ_1_rep = 0.1 + 4*(targ_1 - μ_1_rep[2])^2
            Zygote.ignore() do
                repeat(sample_gauss.(μ_1_rep, σ_1_rep)', m)
            end
        end

        obs2 = map(reps_2) do rep
            μ_2_rep = rep[:p2_pos]
            σ_2_rep = 0.1 + 4*(targ_2 - μ_2_rep[2])^2
            Zygote.ignore() do
                repeat(sample_gauss.(μ_2_rep, σ_2_rep)', m)
            end
        end
        
        obs = Iterators.product(obs1, obs2)

        lls = map(obs) do (ob_1, ob_2)
            μ_1_true = state_dist[:p1_pos]
            μ_2_true = state_dist[:p2_pos]
            σ_1_true = 0.1 .+ 4 * (targ_1 .- μ_1_true[:, 2]).^2
            σ_2_true = 0.1 .+ 4 * (targ_2 .- μ_2_true[:, 2]).^2

            ll_1 = sum(SensingGames.gauss_logpdf.(ob_1, μ_1_true, σ_1_true), dims=2)
            ll_2 = sum(SensingGames.gauss_logpdf.(ob_2, μ_2_true, σ_2_true), dims=2)
            ll = ll_1 #.+ ll_2

            # Prevent ll from being too extreme
            ll = softclamp.(ll, -50, 50)
            vec(ll)
        end


        ll_norm = log.(sum(ll -> exp.(ll), lls))

        # println("===")
        # @show exp.(lls[1] .- ll_norm)[1:end]
        # @show exp.(lls[2] .- ll_norm)[1:end]

        res = map(zip(obs, lls)) do (((ob_1, ob_2), ll))
            new_dist = alter(state_dist,
                :p1_obs => ob_1,
                :p2_obs => ob_2
            )
            new_dist = reweight(new_dist, 
                (ll - ll_norm)
            )
            new_dist
        end

        res = vec(res)
        
        # This enables / disables universally consistent observations
        # 
        # On: (o1a, u1a) -> (o2a, u2a)
        #                   (o2b, u2a)
        #     (o1b, u1a) -> (o2a, u2a)  (sampling from entire state dist at t=2)
        #                   (o2b, u2a)
        #Off: (o1a, u1a) -> (o2a, u2a)
        #                   (o2b, u2a)
        #     (o1b, u1a) -> (o2c, u2b)  (sampling from state dist GIVEN first obs)
        #                   (o2d, u2b) 

        # I think what's actually correct is NEITHER of these.
        # In this scenario we want:
        #     (o1a, u1a) -> (o2a, u2a)
        #                   (o2b, u2a)
        #     (o1b, u1a) -> (o2c, u2a)
        #                   (o2d, u2a) 
        # Universally consistent is conservative (and also happens
        # to be a lot faster implementation wise)

        z = mapreduce(dist -> dist.z, vcat, res)
        w = mapreduce(dist -> dist.w, vcat, res)
        res = StateDist(z, w, state_dist.ids, state_dist.map)


        res
    end
    state, dyn!
end

function make_localization_cost(targs)
    function cost(dist_hists)
        sum(dist_hists) do hist
            state_dist = hist[end]
            # @show exp.(state_dist.w)
            # @show state_dist[:p1_pos]
            # @show state_dist[:p2_pos]
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
    state1, sdyn1 = make_vel_dynamics(:p1; control_scale=1)
    state2, sdyn2 = make_vel_dynamics(:p2; control_scale=1)

    # Observations happen simultaneously
    obs, odyn = make_localization_sensing(0.0, 0.0, (2, 1))

    ctrl1 = make_horizon_control(:p1, :p1_obs, :p1_vel)
    ctrl2 = make_horizon_control(:p2, :p2_obs, :p2_vel)

    zero_state = merge(state1, state2, obs)

    dyn_fns = [odyn, ctrl1, ctrl2, sdyn1, sdyn2]
    prior_fn = make_localization_prior(zero_state)
    cost_fn =  make_localization_cost([[0.0; 2.0], [0.0; -2.0]])

    localization_game = SensingGame(prior_fn, dyn_fns, cost_fn)
    options = (;
        parallel = false,
        n_particles = 1, # TODO - this is misleading under StateDists
        n_lookahead = 5,
        n_render = 1,
        n_iters = 1000,
        live = false,
        score_mode = :last,
        steps_per_seed = 1
    )


    make_model() = Chain(
        Dense(2 =>  64, relu),
        Dense(64 => 64, relu),
        Dense(64 => 2)
    )
    
    # make_model() = Chain(Dense(2 => 2))

    initial_params = (; 
        policies = (;
            p1 = FluxPolicy(make_model, 2; lr=0.05, t_max=options.n_lookahead),
            p2 = FluxPolicy(make_model, 2; lr=0.05, t_max=options.n_lookahead)
        )
    )

    solve(localization_game, initial_params, options) do (t, particles, params)
        plt = plot(aspect_ratio=:equal, lims=(-3, 3))
        title!("Localization: t=$t")
        dists = step(localization_game, params, n=options.n_lookahead)
        render_localization(dists)
        display(plt)
    end
end
