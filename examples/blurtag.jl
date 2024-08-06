# `blurtag` is a simple imperfect information pursuit-evasion game where
#   the variance of the observation of the opponent position is 
#   related to the velocity of the observer. (Slower is better.)

function make_blurtag_sensing(agent, other; blur_coef, n)
    id_obs = Symbol("$(agent)_obs")
    id_own_pos = Symbol("$(agent)_pos")
    id_infoset = Symbol("$(agent)_info")
    id_own_vel = Symbol("$(agent)_vel")
    id_other_pos = Symbol("$(other)_pos")

    function obs_dyn(state_dist::StateDist, game_params)
        μ = state_dist[id_other_pos]
        σ2 = 1 .+ blur_coef*(sum((state_dist[id_own_vel]).^2, dims=2))
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
            σ2 = 1 .+ blur_coef*(sum((state_dist[s][id_own_vel]).^2))
            obs = compare_dist[c][id_obs]
            sum(log.(SensingGames.gauss_pdf.(obs, μ, σ2) .+ 0.001))
        end

        # obs = reshapecompare_dist[id_obs]
        # sum(SensingGames.gauss_logpdf.(obs, μ, σ2), dims=2)
    end

    info_zero, cross_dyn = make_cross_step(obs_dyn, sensing_lik, id_obs, id_infoset, n)
    merge(State(id_obs => 2), info_zero), cross_dyn 
    
end

function make_blurtag_costs()
    function cost1(hist)
        expectation(hist[end]) do state
            dist2(state[:p1_pos], state[:p2_pos])
        end + 
        sum(hist) do dist
            expectation(dist) do state
                cost_regularize(state[:p1_acc], α=0.1)
            end 
        end
    end
    function cost2(hist)
        expectation(hist[end]) do state
            -dist2(state[:p1_pos], state[:p2_pos])
        end + 
        sum(hist) do dist
            expectation(dist) do state
               cost_regularize(state[:p2_acc], α=0.1)
            end 
        end
    end

    (; p1=cost1, p2=cost2)
end

function make_blurtag_prior(zero_state; n=10)

    p1_pos = [-15.0 20; -15 -20]
    p2_pos = [ 10.0 10;  10 -10]
    p1_vel = [4*ones(2) zeros(2)]
    p2_vel = [4*ones(2) zeros(2)]
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

function render_blurtag(hist)
    for agent in [:p1, :p2]
        render_obs(hist, agent)
        render_traj(hist, agent)
        # render_heading(h, agent)
    end
end

function test_blurtag()
    timestate, clock = make_clock_step(1.0)

    hist1, shist1 = make_hist_step(:p1, [:p1_pos; :p1_obs], 4, 5)
    hist2, shist2 = make_hist_step(:p2, [:p2_pos; :p2_obs], 4, 5)

    state1, sdyn1 = make_acc_dynamics_step(:p1; control_scale=8)
    state2, sdyn2 = make_acc_dynamics_step(:p2; control_scale=0.5)

    _, bdyn1 = make_bound_step(:p1_pos, -22, 22)
    _, bdyn2 = make_bound_step(:p2_pos, -22, 22)

    obs1, odyn1 = make_blurtag_sensing(:p1, :p2; blur_coef=20.0, n=[1; 8; 1; 1; 1])
    obs2, odyn2 = make_blurtag_sensing(:p2, :p1; blur_coef=20.0, n=[1; 1; 1; 1; 1])
    zero_state = merge(timestate, hist1, hist2, state1, state2, obs1, obs2)

    pol1, ctrl1 = make_nn_control(:p1, :p1_hist, :p1_acc, 4, 2)
    pol2, ctrl2 = make_nn_control(:p2, :p2_hist, :p2_acc, 4, 2)
    init_params = (; p1=pol1, p2=pol2)

    dyn_fns = [clock, odyn1, odyn2, shist1, shist2, ctrl1, ctrl2, sdyn1, sdyn2]
    prior_fn = make_blurtag_prior(zero_state)
    cost_fns = make_blurtag_costs()

    blurtag_game = SensingGame(prior_fn, dyn_fns)

    options = (;
        n_lookahead = 5,
        n_render = 1,
        n_iters = 400,
        steps_per_seed = 1000
    )

    iter = 1
    solve(blurtag_game, init_params, cost_fns, options) do params
        hist = rollout(blurtag_game, params, n=options.n_lookahead)

        plt = plot(aspect_ratio=:equal, lims=(-30, 50))
        title!("Planar blurtag step=$iter")
        iter += 1
        println(iter)
        render_blurtag(hist)
        display(plt)
    end
end