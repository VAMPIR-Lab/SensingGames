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



function make_fovtag_sensing_step(agent, other; fov, scale=100.0, offset=2)
    id_obs = Symbol("$(agent)_obs")
    id_own_θ = Symbol("$(agent)_θ")
    id_own_pos = Symbol("$(agent)_pos")
    id_other_pos = Symbol("$(other)_pos")

    function get_sensing_noise(dθ)
        d = (fov/2 - abs(dθ))
        offset + d * ((d > 0) ? (-0.001) : (-scale))
    end

    function get_gaussian_params(state_dist)
        our_pos = state_dist[id_own_pos]
        their_pos = state_dist[id_other_pos]

        θ1 = state_dist[id_own_θ] .+ π
        θ2 = atan.(our_pos[:, 2] .- their_pos[:, 2], our_pos[:, 1] .- their_pos[:, 1])
        dθ = angdiff.(θ1, θ2)
        σ = get_sensing_noise.(dθ)
        μ = state_dist[id_other_pos]

        return μ, σ.^2
    end

    function rollout_observation(state_dist::StateDist, game_params)
        μ, σ2 = get_gaussian_params(state_dist)   
        obs = sample_trimmed_gauss.(μ, σ2; l=-40, u=40)

        alter(state_dist, 
            id_obs => obs
        )
    end

    function lik_observation(state_dist, compare_dist, game_params) 
        μ, σ2 = get_gaussian_params(state_dist)   
        x = compare_dist[id_obs]

        # mu:    (|dist| x 2) => (|dist| x 1 x 2)
        # sigma: (|dist| x 1) => (|dist| x 1 x 1)
        # x:     (|info| x 2) => (1 x |info| x 2)
        μ = reshape(μ, (length(state_dist), 1, 2))
        σ2 = reshape(σ2, (length(state_dist), 1, 1))
        x = reshape(x, (1, length(compare_dist), 2))

        # Note that we use the true Gaussian PDF (not the trimmed one)
        p = prod(SensingGames.gauss_pdf.(x, μ, σ2), dims=3)
        log.(p .+ 1e-10)
    end

    GameComponent(rollout_observation, lik_observation, [id_obs])
end

function make_fovtag_costs()

    function cost1(hist)
        sum(hist) do distr
            d = (sum((distr[:p1_pos] .- distr[:p2_pos]).^2, dims=2))
            l = sum((distr[:p1_pos]).^2, dims=2)
            # b = cost_bound.(l, -1, 40^2)
            b = cost_bound.(distr[:p1_pos], [-40 -40], [40 40])
            r = cost_regularize.(distr[:p1_vω], α=10)
            sum((b .+ r .+ d) .* exp.(distr.w))
        end
    end
    function cost2(hist)
        sum(hist) do distr
            sum((-(sum((distr[:p1_pos] .- distr[:p2_pos]).^2, dims=2)) .+
            cost_bound.(sqrt.(sum((distr[:p2_pos]).^2, dims=2)), [0], [40]) .+
            cost_regularize.(distr[:p2_vω], α=10)) .*
            exp.(distr.w))
        end
    end

    (; p1=cost1, p2=cost2)
end

function make_fovtag_prior(zero_state; n=16)
    function prior() 
        alter(StateDist(zero_state, n),
            :p1_pos => rand(n, 2)*30 .- 15,
            :p1_vel => rand(n, 2)*0.01,
            :p1_θ => rand(n, 1)*2π,
            :p2_pos => rand(n, 2)*30 .- 15,
            :p2_vel => rand(n, 2)*0.01,
            :p2_θ => rand(n, 1)*2π
        )
    end
end

function test_fovtag()
    T = 5
    fov = (; p1=1.0, p2=1.0)

    optimization_options = (;
        n_lookahead = T,
        n_iters = 1,
        batch_size = 20,
        max_wall_time = 120000, # Not respected right now
        steps_per_seed = 1,
        steps_per_render = 10
    )

    clock_step = make_clock_step(1.0)

    p1_pos_hist_step = make_hist_step(:p1_pos_h, :p1_pos, T)
    p2_pos_hist_step = make_hist_step(:p2_pos_h, :p2_pos, T)
    p1_obs_hist_step = make_hist_step(:p1_obs_h, :p1_obs, T)
    p2_obs_hist_step = make_hist_step(:p2_obs_h, :p2_obs, T)
    p1_ang_hist_step = make_hist_step(:p1_θ_h, :p1_θ, T)
    p2_ang_hist_step = make_hist_step(:p2_θ_h, :p2_θ, T)

    p1_dynamics_step = make_vel_dynamics_step(:p1; control_scale=1.4)
    p2_dynamics_step = make_vel_dynamics_step(:p2; control_scale=2)

    p1_bound_step = make_bound_step(:p1_pos, -42, 42)
    p2_bound_step = make_bound_step(:p2_pos, -42, 42)

    p1_obs_step = make_fovtag_sensing_step(:p1, :p2; fov=fov[:p1])
    p2_obs_step = make_fovtag_sensing_step(:p2, :p1; fov=fov[:p2])

    # TODO - there's some special casing for this example in policies.jl that needs to go
    p1_control_step = make_nn_control(:p1, [:p1_pos_h, :p1_obs_h], :p1_vel, t_max=T)
    p2_control_step = make_nn_control(:p2, [:p2_pos_h, :p2_obs_h], :p2_vel, t_max=T)

    

    attr_ids = [:pos, :vel, :acc, :θ, :vω, :obs, :pos_h, :obs_h, :θ_h]
    attr_w   = [ 2;    2;    2;    1;  2;   2;    2*T;    2*T;    T]
    zero_state = State([
        :t => 1;
        Symbol.(:p1, :_, attr_ids) .=> attr_w;
        Symbol.(:p2, :_, attr_ids) .=> attr_w;
    ]...)



    fovtag_game = ContinuousGame([
        clock_step,
        p1_obs_step,       p2_obs_step,
        p1_pos_hist_step,  p2_pos_hist_step,
        p1_obs_hist_step,  p2_obs_hist_step,
        p1_ang_hist_step,  p2_ang_hist_step,
        p1_control_step,   p2_control_step,
        p1_dynamics_step,  p2_dynamics_step,
        # p1_bound_step,     p2_bound_step
    ])

    # Version of the game where players don't get new information
    #   during planning (no active information gathering)
    inactive_fovtag_game = ContinuousGame([
        clock_step,
        p1_obs_step,       p2_obs_step,
        p1_pos_hist_step,  p2_pos_hist_step,
        p1_ang_hist_step,  p2_ang_hist_step,
        p1_control_step,   p2_control_step,
        p1_dynamics_step,  p2_dynamics_step,
        # p1_bound_step,     p2_bound_step
    ])



    prior_fn = make_fovtag_prior(zero_state, n=1000)
    p1_belief = HybridParticleBelief(fovtag_game, prior_fn(), 0.01)
    p2_belief = HybridParticleBelief(fovtag_game, prior_fn(), 0.01)

    true_state = StateDist([prior_fn()[rand(1:10)]])
    p1_ids = [:t; :p1_pos; :p1_vel; :p1_θ; :p1_obs; :p1_pos_h; :p1_obs_h]
    p2_ids = [:t; :p2_pos; :p2_vel; :p2_θ; :p2_obs; :p2_pos_h; :p2_obs_h]

    cost_fns  = make_fovtag_costs()
    renderer = MakieRenderer()


    p1_params = (;
        p1 = make_policy(4*T, 3; t_max=T), # π_1^(1)
        p2 = make_policy(4*T, 3; t_max=T)  # π_2^(1)
    )
    p2_params = (;
        p1 = make_policy(4*T, 3; t_max=T), # π_1^(2)
        p2 = make_policy(4*T, 3; t_max=T)  # π_2^(2)
    )
    true_params = (;
        p1 = p1_params[:p1],
        p2 = p2_params[:p2]
    )

    for t in 1:1000

        # Each player solves their game
        iter = 0
        p1_params = solve(fovtag_game, p1_belief, true_params, cost_fns, optimization_options) do params

            print(".")
            iter += 1
            if iter % optimization_options.steps_per_render != 0
                return false
            end

            return false
        end

        p2_params = solve(fovtag_game, p2_belief, true_params, cost_fns, optimization_options)
        true_params = (;
            p1 = p1_params[:p1],
            p2 = p2_params[:p2]
        )



        true_state = step(fovtag_game, true_state, true_params)
        update(p1_belief, select(true_state, p1_ids...)[1], p1_params)
        update(p2_belief, select(true_state, p2_ids...)[1], p2_params)

        # Big assumption: p1_params = p2_params = true_params
        render_fovtag(renderer, [
            (draw(p1_belief; n=20), true_params),
            (true_state, true_params),
            (draw(p2_belief; n=20), true_params)
            ], fovtag_game, fov; T)
    end
end

function render_fovtag(renderer, dists, game, fov; T)

    unspool_scheme = [
        :p1_pos_h => :p1_pos,
        :p2_pos_h => :p2_pos,
        :p1_obs_h => :p1_obs,
        :p2_obs_h => :p2_obs,
        :p1_θ_h   => :p1_θ,
        :p2_θ_h   => :p2_θ
    ]

    p1_colormap = :linear_blue_5_95_c73_n256
    p2_colormap = :linear_kry_5_95_c72_n256
    colorrange = (-T, T)

    render(renderer) do
        for (col, (dist, params)) in enumerate(dists)

            plan = step(game, dist, params; n=T)[end]

            for i in 1:length(dist)
                current_state = dist[i]
                lik = exp(dist.w[i])

                render_location(renderer, current_state, :p1_pos; 
                    ax_idx=(1, col), color=0, alpha=1.0*lik,
                    colormap=p1_colormap, colorrange, markersize=8)
                render_location(renderer, current_state, :p2_pos; 
                    ax_idx=(1, col), color=0, alpha=1.0*lik,
                    colormap=p2_colormap, colorrange, markersize=8)


                past = unspool(current_state, unspool_scheme...)
                future_particle = plan[i]
                future = unspool(future_particle, unspool_scheme...)
                history = [past; future]
                # history = [past; future[1]]

                render_trajectory(renderer, history, :p1_pos; 
                    ax_idx=(1, col), color=0, alpha=0.5*lik,
                    colormap=p1_colormap, colorrange)
                render_trajectory(renderer, history, :p2_pos; 
                    ax_idx=(1, col), color=0, alpha=0.5*lik,
                    colormap=p2_colormap, colorrange)
                
                for (i, state) in enumerate(history)
                    t = i-T
                    # render_location(renderer, state, :p1_obs; 
                    #     ax_idx=(1, col), color=t, alpha=1.0 * ((t > 0) ? 0 : 1), marker='x',
                    #     colormap=p1_colormap, colorrange)
                    # render_location(renderer, state, :p2_obs; 
                    #     ax_idx=(1, col), color=t, alpha=1.0 * ((t > 0) ? 0 : 1), marker='x',
                    #     colormap=p2_colormap, colorrange)
                    render_fov(renderer, state, fov[:p1], :p1_pos, :p1_θ; 
                        ax_idx=(1, col), color=t, alpha=0.1*lik,
                        colormap=p1_colormap, colorrange)
                    render_fov(renderer, state, fov[:p2], :p2_pos, :p2_θ; 
                        ax_idx=(1, col), color=t, alpha=0.1*lik,
                        colormap=p2_colormap, colorrange)

                    t += 1
                end
            end
        end
    end
end