using Dates

# `fovtag` is an imperfect information pursuit-evasion game where
#   each player has a field of view in the direction of their velocity.
#   When the opponent is within the field of view observations are perfect;
#   otherwise observations are very noisy (with a steep but smooth transition
#   in between).


# Observtions for FOVtag are as follows:
# * If a player's field-of-view is over the opponent, they get 
#   a near-perfect observation of the opponent's location
#   (Gaussian with sigma=`offset`)
# * If not, they get a random observation in bounds ([-40, 40] for both axes).
#   The observation is trimmed Gaussian with sigma=scale*d where d is the angle
#   in radians outside the field of view
#   With a large scale this looks like a uniform distribution on [-40, 40]
function make_fovtag_sensing_step(agent, other; fov, scale=100.0, offset=0.1, 
                                  blind=false, discard_first=true, counterfactuals=nothing)
    id_obs = Symbol("$(agent)_obs")
    id_own_pos = Symbol("$(agent)_pos")
    id_own_vel = Symbol("$(agent)_vel")
    id_other_pos = Symbol("$(other)_pos")
    
    function _sense_noise(dθ)
        d = (fov/2 - abs(dθ))
        r = if d > 0
            -0.01*d   # Must still have a slight r 
        else
            scale*-d
        end

        offset + r
    end

    function trim_tails(quantile, μ, σ)
        # Only worry about trimming the tails 
        #   if the variance is actually high enough
        #   for observations to frequently leave bounds
        #   Otherwise the gradient seems to have numerical problems
        if σ > 5
            prc1 = gauss_cdf.(μ, σ.^2, -40.0)
            prc2 = gauss_cdf.(μ, σ.^2, 40.0)
            prc1 .+ quantile.*(prc2 .- prc1)
        else
            quantile
        end
    end

    function obs_dyn(state_dist::StateDist, game_params, quantile)

        our_pos = state_dist[id_own_pos]
        their_pos = state_dist[id_other_pos]
        our_vel = state_dist[id_own_vel]

        t = Int(state_dist[:t][1])
        θ1 = atan.(our_vel[:, 2], our_vel[:, 1]) .+ π
        θ2 = atan.(our_pos[:, 2] .- their_pos[:, 2], our_pos[:, 1] .- their_pos[:, 1])
        dθ = angdiff.(θ1, θ2)
        σ = _sense_noise.(dθ)

        μ = state_dist[id_other_pos]

        new_quantile = trim_tails.(quantile, μ, σ)

        obs = sample_gauss.(μ, σ.^2, new_quantile)

        if blind
            obs = 100 * rand(size(obs)...) .- 50
        end

        alter(state_dist, 
            id_obs => (discard_first && t==1) ? 0*obs : obs
        )
    end

    function sensing_lik(state_dist::StateDist, compare_dist::StateDist)
        our_pos = state_dist[id_own_pos]
        their_pos = state_dist[id_other_pos]
        our_vel = state_dist[id_own_vel]
        
        θ1 = atan.(our_vel[:, 2], our_vel[:, 1]) .+ π
        θ2 = atan.(our_pos[:, 2] .- their_pos[:, 2], our_pos[:, 1] .- their_pos[:, 1])
        dθ = angdiff.(θ1, θ2)
        σ = _sense_noise.(dθ)
        μ = state_dist[id_other_pos]
        x = compare_dist[id_obs]

        
        # mu:    (|dist| x 2) => (|dist| x 1 x 2)
        # sigma: (|dist| x 1) => (|dist| x 1 x 1)
        # x:     (|info| x 2) => (1 x |info| x 2)
        μ = reshape(μ, (length(state_dist), 1, 2))
        σ = reshape(σ, (length(state_dist), 1, 1))
        x = reshape(x, (1, length(compare_dist), 2))

        # Technically the PDF is not the full Gaussian if it is truncated
        #   But the way in which it is truncated is the same in a particle. 
        #   The higher and lower tails are chopped off, which causes a 
        #   rescaling of the original PDF (and zero probability for observations
        #   in the tail, but we don't care about that because we never see those).
        #   All observations get the same rescale -> that probability falls 
        #   out when we normalize later on.

        p = prod(SensingGames.gauss_pdf.(x, μ, σ.^2), dims=3)
        log.(p .+ 1e-10)
    end

    step_fn = if isnothing(counterfactuals) 
        (state, params) -> obs_dyn(state, params, rand(SensingGames._game_rng, length(state), 2))
    else
        make_cross_step(obs_dyn, sensing_lik, counterfactuals, [id_obs])
    end

    step_fn
end

function make_fovtag_costs()

    function cost1(hist)
        sum(hist) do distr
            d = (sum((distr[:p1_pos] .- distr[:p2_pos]).^2, dims=2))
            b = cost_bound.(distr[:p2_pos], [-40 -40], [40 40])
            sum((b .+ d) .* exp.(distr.w))
        end
    end
    function cost2(hist)
        sum(hist) do distr
            d = (sum((distr[:p1_pos] .- distr[:p2_pos]).^2, dims=2))
            b = cost_bound.(distr[:p2_pos], [-40 -40], [40 40])
            sum((b .- d) .* exp.(distr.w))
        end
    end

    (; p1=cost1, p2=cost2)
end


function make_fovtag_plain_costs()
    # TODO - clean up

    function cost1(hist)
        sum(hist) do distr
            d = (sum((distr[:p1_pos] .- distr[:p2_pos]).^2, dims=2))
            sum(d .* exp.(distr.w))
        end
    end
    function cost2(hist)
        sum(hist) do distr
            d = (sum((distr[:p1_pos] .- distr[:p2_pos]).^2, dims=2))
            sum(-d .* exp.(distr.w))
        end
    end

    (; p1=cost1, p2=cost2)
end

function make_fovtag_prior(zero_state; n=16, c=0)

    rng = Random.MersenneTwister()

    p1_prior() = alter(StateDist(zero_state, n),
        :p1_pos => randn(rng, n, 2)*20,
        :p1_vel => randn(rng, n, 2)*0.01
    )

    p2_prior_ind(state, params, quantile) = alter(state,
        :p2_pos => randn(rng, n, 2)*5,
        :p2_vel => randn(rng, n, 2)*0.01
    )

    # P1 and P2 priors are independent - P(s2 | s1) = P(s2) which gets
    #   normalized out anyway
    # In this case the cross step is a little unecessarily complex
    # but that's OK
    p2_ll(state, compare) = ones(n, n)

    p2_prior = make_cross_step(p2_prior_ind, p2_ll, c, [:p2_pos, :p2_vel])

    return () -> p2_prior(p1_prior(), nothing)
end

function test_fovtag()
    T = 7
    fov = (; p1=4π, p2=2)

    options = (;
        n_lookahead = T,
        n_iters = 10,
        max_wall_time = 120000, # Not respected right now
        steps_per_seed = 1,
        steps_per_render = 20
    )

    # 1.1 - Set up steps in the game

    clock_step = make_clock_step(1.0)

    p1_pos_hist_step = make_hist_step(:p1_pos_h, :p1_pos, T)
    p2_pos_hist_step = make_hist_step(:p2_pos_h, :p2_pos, T)
    p1_obs_hist_step = make_hist_step(:p1_obs_h, :p1_obs, T)
    p2_obs_hist_step = make_hist_step(:p2_obs_h, :p2_obs, T)

    p1_dynamics_step = make_vel_dynamics_step(:p1; control_scale=0.5)
    p2_dynamics_step = make_vel_dynamics_step(:p2; control_scale=0.5)

    p1_bound_step = make_bound_step(:p1_pos, -42, 42)
    p2_bound_step = make_bound_step(:p2_pos, -42, 42)

    p1_obs_step = make_fovtag_sensing_step(:p1, :p2; fov=fov[:p1])
    p2_obs_step = make_fovtag_sensing_step(:p2, :p1; fov=fov[:p2])

    # TODO - there's some special casing for this example in policies.jl that needs to go
    p1_control_step = make_nn_control(:p1, [:p1_pos_h, :p1_obs_h], :p1_vel, t_max=T)
    p2_control_step = make_nn_control(:p2, [:p2_pos_h, :p2_obs_h], :p2_vel, t_max=T)

    # 1.1 - Set up the policies
    game_params = (;
        p1 = make_policy(4*T, 2; t_max=T),
        p2 = make_policy(4*T, 2; t_max=T)
    )

    # 1.2 - Set up the initial (zero) state
    # TODO: We used to return zero states for each step and merge
    #   Now we just give the whole thing explicitly here (which is nice when
    #   we're wondering if a particular thing is part of the state, and cleaner
    #   in the step functions)
    # The other examples still do the old way; will fix as needed
    attr_ids = [:pos, :vel, :obs, :pos_h, :obs_h]
    attr_w = [2; 2; 2; 2*T; 2*T]
    zero_state = State([
        :t => 1;
        Symbol.(:p1, :_, attr_ids) .=> attr_w;
        Symbol.(:p2, :_, attr_ids) .=> attr_w;
    ]...)


    # 2 - Set up the games themselves 

    # Training: pull new particles each rollout
    #   Use counterfactuals, maybe
    train_fovtag_game = begin
        prior_fn_train = make_fovtag_prior(zero_state; n=50, c=0)
        p2_obs_step_train = make_fovtag_sensing_step(:p2, :p1; fov=fov[:p2],
            # vvv This is to be experimented with (also c=0 in prior_fn_train)
            counterfactuals=[25; 0; 0; 0; 0; 0; 0; 0]
        )

        SensingGame(prior_fn_train, [
            clock_step,
            p1_obs_step,       p2_obs_step_train,
            p1_pos_hist_step,  p2_pos_hist_step,
            p1_obs_hist_step,  p2_obs_hist_step,
            p1_control_step,   p2_control_step,
            p1_dynamics_step,  p2_dynamics_step,
            p1_bound_step,     p2_bound_step
        ])
    end

    # Testing: Lots of unseen particles
    test_fovtag_game = begin
        prior_fn_test = make_fovtag_prior(zero_state; n=200, c=0)
        SensingGame(prior_fn_test, [
            clock_step,
            p1_obs_step,      p2_obs_step,
            p1_pos_hist_step, p2_pos_hist_step,
            p1_obs_hist_step, p2_obs_hist_step,
            p1_control_step,  p2_control_step,
            p1_dynamics_step, p2_dynamics_step,
            p1_bound_step,    p2_bound_step
        ])
    end

    # 3 - Set up a couple data / viz things
    cost_fns  = make_fovtag_costs()
    score_fns = make_fovtag_plain_costs() # competitive part of the cost only

    renderer = MakieRenderer()

    c1_train = []
    c2_train = []
    c1_test = []
    c2_test = []
    n_rolling_avg = 100

    # 4 - Actually do the solve

    game_params = solve(train_fovtag_game, game_params, cost_fns, options) do params

        train_hist = rollout(train_fovtag_game, game_params, n=options.n_lookahead)
        test_hist = rollout(test_fovtag_game,   game_params, n=options.n_lookahead)

        roll!(c1_train, score_fns[:p1](train_hist), n_rolling_avg)
        roll!(c2_train, score_fns[:p2](train_hist), n_rolling_avg)
        roll!(c1_test,  score_fns[:p1](test_hist), n_rolling_avg)
        roll!(c2_test,  score_fns[:p2](test_hist), n_rolling_avg)
        println("
            P1 train: $(sum(c1_train)/length(c1_train))\t 
            P2 train: $(sum(c2_train)/length(c2_train))\t
            P1 test: $(sum(c1_test)/length(c1_test))\t 
            P2 test: $(sum(c2_test)/length(c2_test))\t"
        )
        @show "a"

        # Render the first few test samples
        show_dist = StateDist(
            test_hist[end].z[1:4, :],
            test_hist[end].w[1:4],
            test_hist[end].ids,
            test_hist[end].map
        )

        render_dist(renderer, show_dist) do (state, rc)
            render_agents([:p1, :p2], rc) do (agent, rc)
                id_obs_hist = Symbol(agent, "_obs_h")
                id_pos_hist = Symbol(agent, "_pos_h")
                obs_hist = @lift(reverse(Vector{State}($state[id_obs_hist], :obs => 2)))
                pos_hist = @lift(reverse(Vector{State}($state[id_pos_hist], :pos => 2)))

                cost = @lift(cost_fns[agent]([StateDist([$state])]))
                agent_name = (agent == :p1) ? "pursuer" : "evader"
                cost_text = @lift("$agent_name cost: " * @sprintf("%.1f", $cost))

                render_traj(renderer, pos_hist, :pos, rc, fov=fov[agent])
                render_points(renderer, obs_hist, :obs, rc)
                render_info(renderer, cost_text, (agent == :p1) ? 2 : 1, rc)
            end
        end

        return false # not done yet - allow to use all iterations
    end
end