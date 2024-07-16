# `fovtag` is an imperfect information pursuit-evasion game where
#   each player has a field of view in the direction of their velocity.
#   When the opponent is within the field of view observations are perfect;
#   otherwise observations are very noisy (with a steep but smooth transition
#   in between).

function make_fovtag_sensing(agent, other; fov, scale=40, offset=1, branch=true, n)
    id_obs = Symbol("$(agent)_obs")
    id_own_pos = Symbol("$(agent)_pos")
    id_infoset = Symbol("$(agent)_info")
    id_own_vel = Symbol("$(agent)_vel")
    id_other_pos = Symbol("$(other)_pos")
    
    function _sense_noise(dθ)

        d = (fov/2 - abs(dθ))
        r = if d > 0
            -0.01*d
        else
            scale*-d
        end

        offset + r
    end

    function obs_dyn(state_dist::StateDist, game_params, quantile)

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

        # σ = 1 .+ σ * (q/10000)
        # @show (q/10000)
        # σ = 1
        # σ = offset .+ scale*softif.(fov/2 .- dθ, 0, 1; hardness=5)


        μ = state_dist[id_other_pos]


        obs = sample_gauss.(μ, σ.^2)
        # obs_alt = sample_trunc_gauss.(μ, σ.^2, [-40], [40], quantile)

        # obs = obs_alt

        # TODO: Ignoring all observations except the second one!
        alter(state_dist, 
            id_obs => (t == 2) ? obs : 0*obs
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
            r = sum(SensingGames.gauss_pdf.(obs, μ, σ^2))
            sum(log.(r .+ 1e-10))
        end

        # obs = reshapecompare_dist[id_obs]
        # sum(SensingGames.gauss_logpdf.(obs, μ, σ2), dims=2)
    end

    info_zero, cross_dyn = make_cross_step(obs_dyn, sensing_lik, id_obs, id_infoset, n)
    
    noncross_dyn = (state, params) -> obs_dyn(state, params, rand(SensingGames._game_rng, 1, 2))
    
    merge(State(id_obs => 2), info_zero), (branch) ? cross_dyn : noncross_dyn 
    
end

function make_fovtag_costs()

    function cost1(hist)
        # sum(hist) do distr
        distr = hist[end]
            expectation(distr) do state
                dist(state[:p1_pos], state[:p2_pos])
            end 
        # end#+ 
        # sum(hist) do dist
        #     expectation(dist) do state
        #         cost_regularize(state[:p1_vel], α=0.1)
            # end 
        # end
    end
    function cost2(hist)
        distr = hist[end]
        # sum(hist) do distr
            expectation(distr) do state
                -dist(state[:p1_pos], state[:p2_pos])
            end 
        # end#+ 
        # sum(hist) do dist
        #     expectation(dist) do state
        #        cost_regularize(state[:p2_vel], α=0.1)
        #     end 
        # end
    end

    (; p1=cost1, p2=cost2)
end

function make_fovtag_prior(zero_state; n=10)

    p1_pos = [-30.0 0; 30 0]
    p2_pos = [-1.0 15.0; 1.0 -15.0]
    p1_vel = 0.01 * ones((2, 2))
    p2_vel = [0 0.0001; 0 0.0001]
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

function test_fovtag()
    T = 7
    fov = (; p1=4π-0.01, p2=1)

    timestate, clock = make_clock_step(1.0)

    hist1, shist1 = make_hist_step(:p1_hist, [:p1_pos; :p1_obs], 4, T)
    hist2, shist2 = make_hist_step(:p2_hist, [:p2_pos; :p2_obs], 4, T)

    state1, sdyn1 = make_vel_dynamics(:p1; control_scale=3.0)
    state2, sdyn2 = make_vel_dynamics(:p2; control_scale=4.0)

    _, bdyn1 = make_bound_dynamics(:p1_pos, -22, 22)
    _, bdyn2 = make_bound_dynamics(:p2_pos, -22, 22)

    obs1, odyn1 = make_fovtag_sensing(:p1, :p2; fov=fov[:p1], branch=false, n=[1; 1; 1; 1; 1; 1; 1; 1])
    obs2, odyn2 = make_fovtag_sensing(:p2, :p1; fov=fov[:p2], branch=false, n=[1; 6; 1; 1; 1; 1; 1; 1])
    zero_state = merge(timestate, hist1, hist2, state1, state2, obs1, obs2)

    pol1, ctrl1 = make_nn_control(:p1, :p1_hist, :p1_vel, 4, 2, t_max=T)
    pol2, ctrl2 = make_nn_control(:p2, :p2_hist, :p2_vel, 4, 2, t_max=T)
    init_params = (; p1=pol1, p2=pol2)

    dyn_fns = [clock, odyn1, odyn2, shist1, shist2, ctrl1, ctrl2, sdyn1, sdyn2]
    prior_fn = make_fovtag_prior(zero_state)
    cost_fns = make_fovtag_costs()

    train_fovtag_game = SensingGame(prior_fn, dyn_fns)

    _, todyn1 = make_fovtag_sensing(:p1, :p2; fov=fov[:p1], branch=false, n=ones(Int, T))
    _, todyn2 = make_fovtag_sensing(:p2, :p1; fov=fov[:p2], branch=false, n=ones(Int, T))
    test_dyn_fns = [clock, todyn1, todyn2, shist1, shist2, ctrl1, ctrl2, sdyn1, sdyn2]
    test_prior_fn() = alter(prior_fn(), 
        :p1_info => rand(Int8, 4, 1),
        :p2_info => rand(Int8, 4, 1)
        )
    test_fovtag_game = SensingGame(test_prior_fn, test_dyn_fns)
    

    options = (;
        n_lookahead = T,
        n_render = 1,
        n_iters = 5000,
        steps_per_seed = 1,
        steps_per_render = 10
    )

    renderer = MakieRenderer()

    iter = 0
    c1_train = []
    c2_train = []
    c1_test = []
    c2_test = []
    N = 100
    solve(train_fovtag_game, init_params, cost_fns, options) do params
        train_hist = rollout(train_fovtag_game, params, n=options.n_lookahead)
        train_dist = train_hist[end]

        reseed!(abs(rand(Int32)))
        test_hist = rollout(test_fovtag_game, params, n=options.n_lookahead)
        test_dist = test_hist[end]

        roll!(c1_train, cost_fns[:p1](train_hist), N)
        roll!(c2_train, cost_fns[:p2](train_hist), N)
        roll!(c1_test, cost_fns[:p1](test_hist), N)
        roll!(c2_test, cost_fns[:p2](test_hist), N)

        println("iter: $iter\t
            P1 train: $(sum(c1_train)/length(c1_train))\t 
            P2 train: $(sum(c2_train)/length(c2_train))\t
            P1 test: $(sum(c1_test)/length(c1_test))\t 
            P2 test: $(sum(c2_test)/length(c2_test))\t
            P2 capture: $(sum(c2_test)/sum(c2_train))\t")


        # hist = rollout(test_fovtag_game, params, n=options.n_lookahead)


        iter += 1
        if (iter % options.steps_per_render == 0)
            render_dist(renderer, test_dist) do (state, rc)
                render_agents([:p1, :p2], rc) do (agent, rc)
                    id_hist = Symbol(agent, "_hist")
                    hist = @lift(reverse(Vector{State}($(state)[id_hist], :pos=>2, :obs=>2)))

                    cost = @lift(cost_fns[agent]([StateDist([$state])]))
                    agent_name = (agent == :p1) ? "pursuer" : "evader"
                    cost_text = @lift("$agent_name cost: " * @sprintf("%.1f", $cost))

                    render_traj(renderer, hist, :pos, rc, fov=fov[agent])
                    render_points(renderer, @lift([$hist[2]; $hist[3]]), :obs, rc)
                    render_info(renderer, cost_text, (agent == :p1) ? 2 : 1, rc)
                end
            end
        end
    end
end