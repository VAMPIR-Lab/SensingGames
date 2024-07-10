# `fovtag` is an imperfect information pursuit-evasion game where
#   each player has a field of view in the direction of their velocity.
#   When the opponent is within the field of view observations are perfect;
#   otherwise observations are very noisy (with a steep but smooth transition
#   in between).

function make_fovtag_sensing(agent, other; fov, scale=20, offset=3, n)
    id_obs = Symbol("$(agent)_obs")
    id_own_pos = Symbol("$(agent)_pos")
    id_infoset = Symbol("$(agent)_info")
    id_own_vel = Symbol("$(agent)_vel")
    id_other_pos = Symbol("$(other)_pos")

    function _sense_noise(dθ)
        d = (fov/2 - abs(dθ))
        if d > 0
            offset - 0.01*d
        else
            offset - scale*d
        end
    end

    function obs_dyn(state_dist::StateDist, game_params)
        t = Int(state_dist[:t][1])

        our_pos = state_dist[id_own_pos]
        their_pos = state_dist[id_other_pos]
        our_vel = state_dist[id_own_vel]

        θ1 = atan.(our_vel[:, 2], our_vel[:, 1]) .+ π
        θ2 = atan.(our_pos[:, 2] .- their_pos[:, 2], our_pos[:, 1] .- their_pos[:, 1])
        dθ = angdiff.(θ1, θ2)
        # σ2 = offset .+ abs.(scale * dθ)
        # σ = offset .+ scale * penalty.((fov/2 .- dθ))
        σ = _sense_noise.(dθ)
        # σ = offset .+ scale*softif.(fov/2 .- dθ, 0, 1; hardness=5)

        μ = state_dist[id_other_pos]
        obs = sample_gauss.(μ, σ.^2)

        alter(state_dist, 
            id_obs => (t > 2) ? 0*obs : obs
        )
    end

    function sensing_lik(state_dist::StateDist, compare_dist::StateDist)
        # μ = state_dist[id_other_pos]
        # σ2 = 1 #.+ blur_coef*(sum(abs.(state_dist[id_own_vel]), dims=2))
        
        S = length(state_dist)
        C = length(compare_dist)
        map(Iterators.product(1:S, 1:C)) do (s, c)
            state = state_dist[s]
            μ = state[id_other_pos]

            our_pos = state[id_own_pos]
            their_pos = state[id_other_pos]
            our_vel = state[id_own_vel]    
            
            θ1 = atan(our_vel[2], our_vel[1]) + π
            θ2 = atan(our_pos[2] - their_pos[2], our_pos[1] - their_pos[1])
            dθ = abs(angdiff(θ1, θ2))
            σ = _sense_noise(dθ)
            # σ2 = offset .+ abs(scale * dθ)
            # println(sum(fov/2 - abs(dθ)))
            # σ = offset + scale * penalty((fov/2 - dθ))
            # σ = offset .+ scale*softif.(fov/2 - dθ, 0, 1; hardness=5)

            # σ2 = 1 .+ 2*(sum((state_dist[s][id_own_vel]).^2))
            obs = compare_dist[c][id_obs]
            sum(log.(SensingGames.gauss_pdf.(obs, μ, σ^2) .+ 0.001))
        end

        # obs = reshapecompare_dist[id_obs]
        # sum(SensingGames.gauss_logpdf.(obs, μ, σ2), dims=2)
    end

    info_zero, cross_dyn = make_cross_step(obs_dyn, sensing_lik, id_obs, id_infoset, n)
    merge(State(id_obs => 2), info_zero), cross_dyn 
    
end

function make_fovtag_costs()
    function cost1(hist)
        sum(hist) do dist
            expectation(dist) do state
                dist2(state[:p1_pos], state[:p2_pos])
            end 
        end + 
        sum(hist) do dist
            expectation(dist) do state
                cost_regularize(state[:p1_vel], α=0.1)
            end 
        end
    end
    function cost2(hist)
        sum(hist) do dist
            expectation(dist) do state
                -dist2(state[:p1_pos], state[:p2_pos])
            end 
        end + 
        sum(hist) do dist
            expectation(dist) do state
               cost_regularize(state[:p2_vel], α=0.1)
            end 
        end
    end

    (; p1=cost1, p2=cost2)
end

function make_fovtag_prior(zero_state; n=10)

    p1_pos = [-25.0 0; 25 0]
    p2_pos = [-1.0 5.0; 1.0 -5.0]
    p1_vel = 0.01 * ones((2, 2))
    p2_vel = 0.01 * ones((2, 2))
    # p1_pos = [fill(-10, n) 10 * randn(n)]
    # p1_vel = (randn(n, 2) .+ [2 -2])
    # p2_pos = [fill(10, n)  10 * randn(n)]
    # p2_vel = 0 * randn(n, 2) #.+ [4 -4]
    () -> begin
        states = map(Iterators.product(1:2, 1:2)) do (i, j)
            alter(zero_state,
                :p1_pos => p1_pos[i, :],
                :p1_vel => p1_vel[i, :],
                :p2_pos => p2_pos[j, :],
                :p2_vel => p2_vel[j, :],
                :p1_info => [Float64(-i)],
                :p2_info => [Float64(-j)],
            )
        end
        StateDist(states)
    end
end

function render_fovtag(hist; fov)
    for agent in [:p1, :p2]
        # prt = wsample(1:length(hist[end]), exp.(hist[end].w))
        # render_obs(hist, agent; prt)
        # render_traj(hist, agent; prt)
        # render_heading(hist, agent; fov=fov[agent], prt)
        render_obs(hist, agent)
        render_traj(hist, agent)
        render_heading(hist, agent; fov=fov[agent])
    end
end

function test_fovtag()
    T = 5
    fov = (; p1=4π, p2=1)

    timestate, clock = make_clock_step(1.0)

    hist1, shist1 = make_hist_step(:p1_hist, [:p1_pos; :p1_obs], 4, T)
    hist2, shist2 = make_hist_step(:p2_hist, [:p2_pos; :p2_obs], 4, T)

    state1, sdyn1 = make_vel_dynamics(:p1; control_scale=4.0)
    state2, sdyn2 = make_vel_dynamics(:p2; control_scale=4.0)

    _, bdyn1 = make_bound_dynamics(:p1_pos, -22, 22)
    _, bdyn2 = make_bound_dynamics(:p2_pos, -22, 22)

    obs1, odyn1 = make_fovtag_sensing(:p1, :p2; fov=fov[:p1], n=[1; 2; 1; 1; 1; 1; 1; 1])
    obs2, odyn2 = make_fovtag_sensing(:p2, :p1; fov=fov[:p2], n=[1; 4; 1; 1; 1; 1; 1; 1])
    zero_state = merge(timestate, hist1, hist2, state1, state2, obs1, obs2)

    pol1, ctrl1 = make_nn_control(:p1, :p1_hist, :p1_vel, 4, 2, t_max=T)
    pol2, ctrl2 = make_nn_control(:p2, :p2_hist, :p2_vel, 4, 2, t_max=T)
    init_params = (; p1=pol1, p2=pol2)

    dyn_fns = [clock, odyn1, odyn2, shist1, shist2, ctrl1, ctrl2, sdyn1, sdyn2]
    prior_fn = make_fovtag_prior(zero_state)
    cost_fns = make_fovtag_costs()

    train_fovtag_game = SensingGame(prior_fn, dyn_fns)



    _, todyn1 = make_fovtag_sensing(:p1, :p2; fov=fov[:p1], n=ones(Int, T))
    _, todyn2 = make_fovtag_sensing(:p2, :p1; fov=fov[:p2], n=ones(Int, T))
    test_dyn_fns = [clock, todyn1, todyn2, shist1, shist2, ctrl1, ctrl2, sdyn1, sdyn2]
    test_prior_fn() = draw(prior_fn(); n=1)
    test_fovtag_game = SensingGame(test_prior_fn, test_dyn_fns)
    

    options = (;
        n_lookahead = T,
        n_render = 1,
        n_iters = 30,
        steps_per_seed = 10
    )

    iter = 1
    solve(train_fovtag_game, init_params, cost_fns, options) do params
        hist = rollout(train_fovtag_game, params, n=options.n_lookahead)

        plt = plot(aspect_ratio=:equal, lims=(-70, 70))
        title!("Planar fovtag step=$iter")
        iter += 1
        println(iter)
        render_fovtag(hist; fov)
        display(plt)
    end
end