using Dates

# `hastag` (Hide-and-seek Tag) is a variant of field of view tag. The play area includes fixed circular obstacles 
# that block a player's field of view. Agents receive high-variance observations for obstructed opponents in the
# same way as those outside the field of view. Players are permitted to pass through obstacles, but receive a penalty
# for doing so.

# default blocks (obstacles)
de_blocks = [[0 10], [0 -10], [10 0], [-10 0], [20 20], [-20 20], [20 -20], [-20 -20], [0 -37], [0 37], [-37 0], [37 0]]
# default block radii that correspond to the above list
de_brads = [3, 3, 3, 3, 5, 5, 5, 5, 7, 7, 7, 7]

function make_hastag_sensing_step(agent, other; fov, scale=100.0, offset=0.5, blocks=de_blocks, brads=de_brads)
    id_obs = Symbol("$(agent)_obs")
    id_own_θ = Symbol("$(agent)_θ")
    id_own_pos = Symbol("$(agent)_pos")
    id_other_pos = Symbol("$(other)_pos")

    function get_sensing_noise(dθ, cbool)
        d = (fov/2 - abs(dθ))
        offset + d * ((d > 0 && cbool) ? (-0.001) : (-scale))
    end

    function get_gaussian_params(state_dist)
        our_pos = state_dist[id_own_pos]
        their_pos = state_dist[id_other_pos]
        # The angle between heading and the x axis
        θ1 = state_dist[id_own_θ] .+ π
        # The angle between 1. the line connecting two players, and 2. the x axis
        θ2 = atan.(our_pos[:, 2] .- their_pos[:, 2], our_pos[:, 1] .- their_pos[:, 1])
        # The angle between 1. the line connecting us to center of block, and 2. the x axis
        θ3 = mapreduce(hcat, collect(1:size(brads, 1))) do i
            atan.(our_pos[:, 2] .- blocks[i][2], our_pos[:, 1] .- blocks[i][1])
        end
        # The distance between us and blocks
        bdist = mapreduce(hcat, collect(1:size(brads, 1))) do i
            sqrt.((our_pos[:,1] .- blocks[i][1]).^2 .+ (our_pos[:,2] .- blocks[i][2]).^2)
        end
        # The angle between 1. the line connecting us to the center of block, and 2. the tangent line to the block that passes through us
        θbrad = mapreduce(hcat, collect(1:size(brads, 1))) do i
            atan.(brads[i], bdist[:, i])
        end
        # The angle between 1. the line connecting us to the center of block, and 2. the line connecting two players
        dθ2 = mapreduce(hcat, collect(1:size(brads, 1))) do i
            abs.(angdiff.(θ3[:, i], θ2))
        end
        # true if uncovered, false otherwise.
        uncov = all(collect(1:size(brads, 1))) do i
            (θbrad[:, i] < dθ2[:, i]) || (bdist[:, i] > sqrt.((our_pos[:,1] .- their_pos[:,1]).^2 .+ (our_pos[:,2] .- their_pos[:,2]).^2))
        end
        dθ = angdiff.(θ1, θ2)
        σ = get_sensing_noise.(dθ, uncov)
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

        # μ:    (|dist| x 2) => (|dist| x 1 x 2)
        # σ2: (|dist| x 1) => (|dist| x 1 x 1)
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

function make_hastag_costs(blocks=de_blocks, brads=de_brads)

    function cost_obstacle(d, brad)
        (d > brad) ? -0.001*(d-brad) : -1000*(d-brad)
    end

    function cost1(hist)
        sum(hist) do distr
            cost_each_obstacle = mapreduce(hcat, collect(1:size(brads, 1))) do i
                cost_obstacle.(sqrt.(sum((distr[:p1_pos] .- blocks[i]).^2, dims=2)), brads[i])
            end
            total_cost_obstacle = sum(cost_each_obstacle, dims=2)
            sum((
                sum((distr[:p1_pos] .- distr[:p2_pos]).^2, dims=2) .+
                cost_bound.(sqrt.(sum((distr[:p1_pos]).^2, dims=2)), [0], [40]) .+
                total_cost_obstacle
            ) .* exp.(distr.w))
        end
    end

    function cost2(hist)
        sum(hist) do distr
            cost_each_obstacle = mapreduce(hcat, collect(1:size(brads, 1))) do i
                cost_obstacle.(sqrt.(sum((distr[:p2_pos] .- blocks[i]).^2, dims=2)), brads[i])
            end
            total_cost_obstacle = sum(cost_each_obstacle, dims=2)
            sum((
                -sum((distr[:p1_pos] .- distr[:p2_pos]).^2, dims=2) .+
                cost_bound.(sqrt.(sum((distr[:p2_pos]).^2, dims=2)), [0], [40]) .+
                total_cost_obstacle
            ) .* exp.(distr.w))
        end
    end

    (; p1=cost1, p2=cost2)
end

function make_hastag_prior(zero_state; n=16)
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

function test_hastag_active_info_gathering()
    open("results.txt", "w") do io
        for (p1_solver, p2_solver) in [
            (:active, :active),
            (:inactive, :active),
            (:active, :inactive),
            (:inactive, :inactive)
            ]

            println(io, "=======\nP1 $p1_solver, P2 $p2_solver")
            println("=======\nP1 $p1_solver, P2 $p2_solver")
            for s in 1:20
                Random.seed!(42+s)

                T = 6
                fov = (; p1=1.0, p2=1.0)

                optimization_options = (;
                    n_lookahead = T,
                    n_iters = 10,
                    batch_size = 10,
                    max_wall_time = 120000, # Not respected right now
                    steps_per_seed = 1,
                    steps_per_render = 10
                )
                renderer = MakieRenderer()

                clock_step = make_clock_step(1.0)

                p1_pos_hist_step = make_hist_step(:p1_pos_h, :p1_pos, T)
                p2_pos_hist_step = make_hist_step(:p2_pos_h, :p2_pos, T)
                p1_obs_hist_step = make_hist_step(:p1_obs_h, :p1_obs, T)
                p2_obs_hist_step = make_hist_step(:p2_obs_h, :p2_obs, T)
                p1_ang_hist_step = make_hist_step(:p1_θ_h, :p1_θ, T)
                p2_ang_hist_step = make_hist_step(:p2_θ_h, :p2_θ, T)

                p1_obs_hist_step_planning = (p1_solver == :active) ? p1_obs_hist_step : make_identity_step()
                p2_obs_hist_step_planning = (p2_solver == :active) ? p2_obs_hist_step : make_identity_step()

                p1_dynamics_step = make_vel_dynamics_step(:p1; control_scale=1.4)
                p2_dynamics_step = make_vel_dynamics_step(:p2; control_scale=2)

                p1_obs_step = make_hastag_sensing_step(:p1, :p2; fov=fov[:p1])
                p2_obs_step = make_hastag_sensing_step(:p2, :p1; fov=fov[:p2])

                p1_control_step = make_nn_control(:p1, [:p1_pos_h, :p1_obs_h], :p1_vel, t_max=T)
                p2_control_step = make_nn_control(:p2, [:p2_pos_h, :p2_obs_h], :p2_vel, t_max=T)

                attr_ids = [:pos, :vel, :acc, :θ, :vω, :obs, :pos_h, :obs_h, :θ_h]
                attr_w   = [ 2;    2;    2;    1;  2;   2;    2*T;    2*T;    T]
                zero_state = State([
                    :t => 1;
                    Symbol.(:p1, :_, attr_ids) .=> attr_w;
                    Symbol.(:p2, :_, attr_ids) .=> attr_w;
                ]...)

                hastag_game = ContinuousGame([
                    clock_step,
                    p1_obs_step,       p2_obs_step,
                    p1_pos_hist_step,  p2_pos_hist_step,
                    p1_obs_hist_step,  p2_obs_hist_step,
                    p1_ang_hist_step,  p2_ang_hist_step,
                    p1_control_step,   p2_control_step,
                    p1_dynamics_step,  p2_dynamics_step,
                ])
                
                planning_hastag_game = ContinuousGame([
                    clock_step,
                    p1_obs_step,       p2_obs_step,
                    p1_pos_hist_step,  p2_pos_hist_step,
                    p1_obs_hist_step_planning,  p2_obs_hist_step_planning,
                    p1_ang_hist_step,  p2_ang_hist_step,
                    p1_control_step,   p2_control_step,
                    p1_dynamics_step,  p2_dynamics_step,
                ])

                prior_fn = make_hastag_prior(zero_state, n=1000)
                p1_belief = HybridParticleBelief(hastag_game, prior_fn(), 0.01)
                p2_belief = HybridParticleBelief(hastag_game, prior_fn(), 0.01)

                true_state = StateDist([prior_fn()[rand(1:10)]])
                p1_ids = [:t; :p1_pos; :p1_vel; :p1_θ; :p1_obs; :p1_pos_h; :p1_obs_h]
                p2_ids = [:t; :p2_pos; :p2_vel; :p2_θ; :p2_obs; :p2_pos_h; :p2_obs_h]

                cost_fns  = make_hastag_costs()

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

                scores = []

                n_steps = 40

                for t in 1:n_steps
                    # Each player solves their game
                    iter = 0
                    p1_params = solve(planning_hastag_game, p1_belief, true_params, cost_fns, optimization_options) do params
            
                        print(".")
                        iter += 1
                        if iter % optimization_options.steps_per_render != 0
                            return false
                        end
            
                        return false
                    end
            
                    p2_params = solve(planning_hastag_game, p2_belief, true_params, cost_fns, optimization_options)
                    true_params = (;
                        p1 = p1_params[:p1],
                        p2 = p2_params[:p2]
                    )
            
                    true_state = step(hastag_game, true_state, true_params)
                    update!(p1_belief, select(true_state, p1_ids...)[1], true_params)
                    update!(p2_belief, select(true_state, p2_ids...)[1], true_params)
                    scores = [scores; dist(true_state[1][:p1_pos], true_state[1][:p2_pos])]
                    
                    render_hastag(renderer, [
                        (true_state, true_params),
                        ], hastag_game, fov; T)
                end
                println(io, sum(scores)/n_steps)
                println(sum(scores)/n_steps)
            end
        end
    end
end

function render_hastag(renderer, dists, game, fov; block_pos=de_blocks, block_r=de_brads, T)

    unspool_scheme = [
        :p1_pos_h => :p1_pos,
        :p2_pos_h => :p2_pos,
        :p1_obs_h => :p1_obs,
        :p2_obs_h => :p2_obs,
        :p1_θ_h   => :p1_θ,
        :p2_θ_h   => :p2_θ
    ]

    colorrange = (-T, T)

    render(renderer) do
        for (col, (dist, params)) in enumerate(dists)

            plan = step(game, dist, params; n=T)[end]

            render_static_circle(renderer, [0 0], 45; ax_idx=(1, col))
            for i = 1:size(block_r, 1)
                render_static_circle(renderer, block_pos[i], block_r[i]; ax_idx=(1, col))
            end

            for i in 1:length(dist)
                current_state = dist[i]
                lik = exp(dist.w[i])

                past = unspool(current_state, unspool_scheme...)
                future_particle = plan[i]
                future = unspool(future_particle, unspool_scheme...)
                history = [past; future]

                render_trajectory(renderer, history, :p1_pos; 
                    ax_idx=(1, col), color=:blue, alpha=lik, linewidth=4)
                render_trajectory(renderer, history, :p2_pos; 
                    ax_idx=(1, col), color=:red, alpha=lik, linewidth=4)
                
                for (i, state) in enumerate(history)
                    t = i-T

                    if length(dist) < 5
                        render_location(renderer, state, :p1_obs; 
                            ax_idx=(1, col), color=:blue, alpha=1.0 * ((t > 0) ? 0 : 1), marker='x', markersize=16)
                        render_location(renderer, state, :p2_obs; 
                            ax_idx=(1, col), color=:red, alpha=1.0 * ((t > 0) ? 0 : 1), marker='x', markersize=16)
                    end
                    render_fov(renderer, state, fov[:p1], :p1_pos, :p1_θ; 
                        ax_idx=(1, col), color=:blue, alpha=0.1*lik)
                    render_fov(renderer, state, fov[:p2], :p2_pos, :p2_θ; 
                        ax_idx=(1, col), color=:red, alpha=0.1*lik)

                    t += 1
                end

                render_location(renderer, current_state, :p1_pos; 
                    ax_idx=(1, col), color=:black, alpha=1.0*lik, markersize=8)
                render_location(renderer, current_state, :p2_pos; 
                    ax_idx=(1, col), color=:black, alpha=1.0*lik, markersize=8)

            end
        end
    end
end