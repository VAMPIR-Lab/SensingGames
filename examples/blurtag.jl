# `blurtag` is a simple imperfect information pursuit-evasion game where
#   the variance of the observation of the opponent position is 
#   related to the velocity of the observer. (Slower is better.)

function make_blurtag_sensing(agent, other; blur_coef, n)
    id_obs = Symbol("$(agent)_obs")
    id_own_pos = Symbol("$(agent)_pos")
    id_own_vel = Symbol("$(agent)_vel")
    id_other_pos = Symbol("$(other)_pos")

    function obs_dyn(state_dist::StateDist, history::Vector{StateDist}, game_params)
        μ = state_dist[id_other_pos]
        σ2 = 5 .+ blur_coef*(sum((state_dist[id_own_vel]).^2, dims=2))
        obs = sample_gauss.(μ, σ2)

        alter(state_dist, 
            id_obs => obs
        )
    end

    function sensing_lik(state_dist::StateDist, compare_dist::StateDist)
        # μ = state_dist[id_other_pos]
        # σ2 = 1 #.+ blur_coef*(sum(abs.(state_dist[id_own_vel]), dims=2))
        
        S = length(state_dist)
        C = length(compare_dist)
        map(Iterators.product(1:S, 1:C)) do (s, c)
            μ = state_dist[s][id_other_pos]
            σ2 = 5 .+ blur_coef*(sum((state_dist[s][id_own_vel]).^2))
            obs = compare_dist[c][id_obs]
            sum(SensingGames.gauss_logpdf.(obs, μ, σ2))
        end

        # obs = reshapecompare_dist[id_obs]
        # sum(SensingGames.gauss_logpdf.(obs, μ, σ2), dims=2)
    end

    State(id_obs => 2), 
    make_cross_step(obs_dyn, sensing_lik, id_obs, n)
end

function make_blurtag_cost()
    function cost(dist_hists)
        sum(dist_hists) do hist
            c = expectation(hist[end]) do state
                [ (dist2(state[:p1_pos], state[:p2_pos]));
                 -(dist2(state[:p1_pos], state[:p2_pos]))]
            end
            r = sum(hist) do state_dist
                expectation(state_dist) do state
                    @show state[:p1_acc]
                    [ cost_regularize(state[:p1_acc], α=1);
                      cost_regularize(state[:p2_acc], α=1)]
                end
            end
            c #+ r
        end
    end
end

function make_blurtag_prior(zero_state; n=10)

    p1_pos = [-15.0  15; -15 -15]
    p2_pos = [ 15.0  15;  15 -15]
    p1_vel = [4*ones(2) zeros(2)]
    p2_vel = zeros((2,2))
    # p1_pos = [fill(-10, n) 10 * randn(n)]
    # p1_vel = (randn(n, 2) .+ [2 -2])
    # p2_pos = [fill(10, n)  10 * randn(n)]
    # p2_vel = 0 * randn(n, 2) #.+ [4 -4]
    () -> begin
        states = map(Iterators.product(1:2, 1:2)) do (i, j)
            alter(zero_state,
                :p1_pos => p1_pos[j, :],
                :p1_vel => p1_vel[j, :],
                :p2_pos => p2_pos[i, :],
                :p2_vel => p2_vel[i, :]
            )
        end
        StateDist(states)
    end
end

function render_blurtag(dists)
    for h in dists
        for agent in [:p1, :p2]
            render_obs(h, agent)
            render_traj(h, agent)
            # render_heading(h, agent)
        end
    end
end

function test_blurtag()
    state1, sdyn1 = make_acc_dynamics(:p1; control_scale=3, drag=0.0)
    state2, sdyn2 = make_acc_dynamics(:p2; control_scale=3, drag=0.0)
    _, bdyn1 = make_bound_dynamics(:p1_pos, -22, 22)
    _, bdyn2 = make_bound_dynamics(:p2_pos, -22, 22)

    obs1, odyn1 = make_blurtag_sensing(:p1, :p2; blur_coef=200, n=[2; 8; 2; 1; 1])
    obs2, odyn2 = make_blurtag_sensing(:p2, :p1; blur_coef=200, n=1)

    ctrl1 = make_horizon_control(:p1, [:p1_obs; :p1_pos], :p1_acc)
    ctrl2 = make_horizon_control(:p2, [:p2_obs; :p2_pos], :p2_acc)

    zero_state = merge(state1, state2, obs1, obs2)

    dyn_fns = [odyn1, ctrl1, sdyn1, sdyn2]
    prior_fn = make_blurtag_prior(zero_state)
    cost_fn = make_blurtag_cost()

    blurtag_game = SensingGame(prior_fn, dyn_fns, cost_fn)
    options = (;
        parallel = false,
        n_particles = 1,
        n_lookahead = 5,
        n_render = 1,
        n_iters = 1000,
        live=true,
        score_mode = :last, # TODO doesn't do anything rn
        steps_per_seed = 10000
    )

    make_model() = Chain(
        Dense(4 =>  32, relu),
        Dense(32 => 64, relu),
        Dense(64 => 2)
    )

    initial_params = (; 
        policies = (;
            p1 = FluxPolicy(make_model, 2; lr=0.01, t_max=options.n_lookahead),
            p2 = FluxPolicy(make_model, 2; lr=0.01, t_max=options.n_lookahead)
        )
    )

    solve(blurtag_game, initial_params, options) do (t, particles, params)
        plt = plot(aspect_ratio=:equal, lims=(-30, 30))
        title!("Planar blurtag: t=$t")
        dists = step(blurtag_game, params, n=options.n_lookahead)
        # hists = last(blurtag_game.history, options.n_lookahead)
        render_blurtag(dists)
        display(plt)
    end
end