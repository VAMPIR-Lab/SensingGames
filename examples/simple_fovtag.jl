using Dates

function make_vistag_sensing_step(agent, other; fov, scale=100.0, offset=0.1)
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

        # No observation on first timestep
        if (state_dist[:t][1]) != 2
            obs *= 0
        end

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

function make_vistag_costs()

    function cost1(hist)
        sum(hist) do distr
            sum((
                sqrt.(sum((distr[:p1_pos] .- distr[:p2_pos]).^2, dims=2)) .+
                cost_bound.(sqrt.(sum((distr[:p1_pos]).^2, dims=2)), [0], [40])
            ) .* exp.(distr.w))
        end
    end
    function cost2(hist)
        sum(hist) do distr
            sum((
                -sqrt.(sum((distr[:p1_pos] .- distr[:p2_pos]).^2, dims=2)) .+
                cost_bound.(sqrt.(sum((distr[:p2_pos]).^2, dims=2)), [0], [40])
            ) .* exp.(distr.w))
        end
    end

    (; p1=cost1, p2=cost2)
end

function make_vistag_prior(zero_state)

    p1_pos = [-25 0; 25 0.0]
    p2_pos = [0 2; 0 2.0]

    function prior() 
        Zygote.ignore() do
            alter(StateDist(zero_state, 2),
                :p1_pos => p1_pos,
                :p1_θ => ones(2, 1)*2π,
                :p2_pos => p2_pos,
                :p2_θ => ones(2, 1)*π/2
            )
        end
    end
end


function test_vistag_active_info_visualization()

    T = 7
    fov = (; p1=1.0, p2=1.0)

    optimization_options = (;
        n_lookahead = T,
        n_iters = 4000,
        batch_size = 2,
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

    p1_obs_hist_step = make_hist_step(:p1_obs_h, :p1_obs, T)
    p2_obs_hist_step = make_hist_step(:p2_obs_h, :p2_obs, T)

    p1_dynamics_step = make_vel_dynamics_step(:p1; control_scale=3)
    p2_dynamics_step = make_vel_dynamics_step(:p2; control_scale=4)

    p1_obs_step = make_vistag_sensing_step(:p1, :p2; fov=fov[:p1])
    p2_obs_step = make_vistag_sensing_step(:p2, :p1; fov=fov[:p2])

    p1_control_step = make_nn_control(:p1, [:p1_pos_h, :p1_obs_h], :p1_vel, t_max=T)
    p2_control_step = make_nn_control(:p2, [:p2_pos_h, :p2_obs_h], :p2_vel, t_max=T)

    attr_ids = [:pos, :vel, :acc, :θ, :vω, :obs, :pos_h, :obs_h, :θ_h]
    attr_w   = [ 2;    2;    2;    1;  2;   2;    2*T;    2*T;    T]
    zero_state = State([
        :t => 1;
        Symbol.(:p1, :_, attr_ids) .=> attr_w;
        Symbol.(:p2, :_, attr_ids) .=> attr_w;
    ]...)

    vistag_game = ContinuousGame([
        clock_step,
        p1_obs_step,       p2_obs_step,
        p1_pos_hist_step,  p2_pos_hist_step,
        p1_obs_hist_step,  p2_obs_hist_step,
        p1_ang_hist_step,  p2_ang_hist_step,
        p1_control_step,   p2_control_step,
        p1_dynamics_step,  p2_dynamics_step,
    ])

    prior_fn = make_vistag_prior(zero_state)
    joint_belief = JointParticleBelief(vistag_game, prior_fn())
    true_state = prior_fn()
    cost_fns  = make_vistag_costs()

    renderer = MakieRenderer()


    true_params = (;
        p1 = make_policy(4*T, 3; t_max=T),
        p2 = make_policy(4*T, 3; t_max=T) 
    )

    scores = []

    n_steps = 1
    for t in 1:n_steps

        k = 1
        true_params = solve(vistag_game, joint_belief, 
                true_params, cost_fns, optimization_options) do params
            println(".")
            k += 1
            if k % optimization_options.steps_per_render == 0
                render_vistag(renderer, [(true_state, params)], vistag_game, fov; T)
            end
            false
        end

        # true_state = step(vistag_game, true_state, true_params)
        # update!(joint_belief, true_params)
        scores = [scores; dist(true_state[1][:p1_pos], true_state[1][:p2_pos])]
        return (renderer, true_state, true_params, vistag_game, fov, T)
    end
end

function render_vistag(renderer, dists, game, fov; T)

    unspool_scheme = [
        :p1_pos_h => :p1_pos,
        :p2_pos_h => :p2_pos,
        :p1_obs_h => :p1_obs,
        :p2_obs_h => :p2_obs,
        :p1_θ_h   => :p1_θ,
        :p2_θ_h   => :p2_θ
    ]

    p1_color = :blue
    p2_color = :red
    colorrange = (-T, T)

    render(renderer) do

        for (col, (dist, params)) in enumerate(dists)

            plan = step(game, dist, params; n=T)[end]

            # render_static_circle(renderer, (0, 0), 45; ax_idx=(1, col))

            for i in 1:length(dist)
                current_state = dist[i]
                lik = 1.0 #exp(dist.w[i])
    

                render_location(renderer, current_state, :p1_pos; 
                    ax_idx=(1, col), alpha=1.0*lik,
                    color=p1_color, colorrange, markersize=16)
                render_location(renderer, current_state, :p2_pos; 
                    ax_idx=(1, col), alpha=1.0*lik,
                    color=p2_color, markersize=16)


                past = unspool(current_state, unspool_scheme...)
                future_particle = plan[i]
                future = unspool(future_particle, unspool_scheme...)
                history = future
                # history = [past; future[1]]

                render_trajectory(renderer, history, :p1_pos; 
                    ax_idx=(1, col), alpha=0.5*lik,
                    color=p1_color, colorrange, linewidth=3)
                render_trajectory(renderer, history, :p2_pos; 
                    ax_idx=(1, col), alpha=0.5*lik,
                    color=p2_color, colorrange, linewidth=3)
                
                for (i, state) in enumerate(history[2:end])
                    t = i-T
                    # render_location(renderer, state, :p1_obs; 
                    #     ax_idx=(1, col), alpha=1.0 * ((t > 0) ? 0 : 1), marker='x',
                    #     color=p1_color, colorrange)
                    # render_location(renderer, state, :p2_obs; 
                    #     ax_idx=(1, col), alpha=1.0 * ((t > 0) ? 0 : 1), marker='x',
                    #     color=p2_color, colorrange)
                    render_fov(renderer, state, fov[:p1], :p1_pos, :p1_θ; 
                        ax_idx=(1, col), alpha=0.1*lik,
                        color=p1_color, colorrange)
                    render_fov(renderer, state, fov[:p2], :p2_pos, :p2_θ; 
                        ax_idx=(1, col), alpha=0.1*lik,
                        color=p2_color, colorrange)

                    t += 1
                end
            end
        end
    end
end