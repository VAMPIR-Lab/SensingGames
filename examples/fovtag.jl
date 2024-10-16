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
        σ = Float32.(get_sensing_noise.(dθ))
        μ = state_dist[id_other_pos]

        return μ, σ.^2
    end

    # function trim_quantile(q, μ, σ2, l, u)
    #     new_quantile = if σ2 > 5^2
    #         prc1 = gauss_cdf.(μ, σ2, l)
    #         prc2 = gauss_cdf.(μ, σ2, u)
    #         prc1 .+ quantile.*(prc2 .- prc1)
    #     else
    #         quantile
    #     end
    # end

    function rollout_observation(state_dist::StateDist, game_params)
        μ, σ2 = get_gaussian_params(state_dist)   
        quantile = rand(length(state_dist), 2) |> gpu
        # obs = sample_trimmed_gauss.(μ, σ2, -40, 40, quantile)

        # quantile = trim_quantile.(quantile, μ, σ2, -40, 40)

        obs = log.(1 ./ quantile .- 1) .* sqrt.(σ2) ./ -1.702 .+ μ

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
        q = exp.(-0.5 *(x .- μ).^2 ./ σ2) ./ sqrt.(2π .* σ2)
        p = prod(q, dims=3)
        log.(p .+ 1e-10)
    end

    GameComponent(rollout_observation, lik_observation, [id_obs])
end

function make_fovtag_costs()

    function cost_bound(d)
        dd = (d .- 40^2)
        dg = (dd .> 0)
        dl = (dd .< 0)
        k = (dl .* (0.1*d)) .+ (dg .* d.^4)
    end

    function cost1(hist)
        sum(hist) do distr
            sum((
                (sum((distr[:p1_pos] .- distr[:p2_pos]).^2, dims=2)) .+
                cost_bound(sum((distr[:p1_pos]).^2, dims=2))
            ) .* exp.(distr.w))
        end
    end
    function cost2(hist)
        sum(hist) do distr
            sum((
                -(sum((distr[:p1_pos] .- distr[:p2_pos]).^2, dims=2)) .+
                cost_bound(sum((distr[:p2_pos]).^2, dims=2))
            ) .* exp.(distr.w))
        end
    end

    (; p1=cost1, p2=cost2)
end


function make_fovtag_prior(zero_state; n=16)
    function prior() 
        Zygote.ignore() do
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
end

function test_fovtag_gamma()
    n_sim_steps = 500
    p1_solves = 4
    p2_solves = 4

    for (γ1, γ2) in [[(γ, 0.1) for γ in 0:0.1:1]; [(0.1, γ) for γ in 0:0.1:1]]
        for s in 1:20
            Random.seed!(42+s)

            open("gamma.txt", "a") do file
                write(file, "$γ1, $γ2, trial: $s\n")
            end

            renderer = MakieRenderer()

            T = 6
            fov = (; p1=1.0, p2=1.0)

            optimization_options = (;
                n_lookahead = T,
                n_iters = 10,
                batch_size = 10,
                steps_per_seed = 1
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

            p1_dynamics_step = make_vel_dynamics_step(:p1; control_scale=1.4)
            p2_dynamics_step = make_vel_dynamics_step(:p2; control_scale=2)

            p1_obs_step = make_fovtag_sensing_step(:p1, :p2; fov=fov[:p1])
            p2_obs_step = make_fovtag_sensing_step(:p2, :p1; fov=fov[:p2])

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
            ])

            prior_fn = make_fovtag_prior(zero_state, n=1000)
            p1_belief = HybridParticleBelief(fovtag_game, prior_fn(), γ1)
            p2_belief = HybridParticleBelief(fovtag_game, prior_fn(), γ2)

            true_state = StateDist([prior_fn()[rand(1:10)]])
            p1_ids = [:t; :p1_pos; :p1_vel; :p1_θ; :p1_obs; :p1_pos_h; :p1_obs_h]
            p2_ids = [:t; :p2_pos; :p2_vel; :p2_θ; :p2_obs; :p2_pos_h; :p2_obs_h]

            cost_fns  = make_fovtag_costs()

            p1_params = map(1:p1_solves) do _
                (;
                    p1 = make_policy(4*T, 3; t_max=T),
                    p2 = make_policy(4*T, 3; t_max=T) 
                )
            end

            p2_params = map(1:p2_solves) do _
                (;
                    p1 = make_policy(4*T, 3; t_max=T),
                    p2 = make_policy(4*T, 3; t_max=T)
                )
            end

            scores = []
            for t in 1:n_sim_steps
                print(".")

                p1_params = map(p1_params) do params
                    solve(fovtag_game, p1_belief, params, cost_fns, optimization_options)
                end

                p2_params = map(p2_params) do params
                    solve(fovtag_game, p2_belief, params, cost_fns, optimization_options)
                end

                true_params = (;
                    p1 = rand(p1_params).p1,
                    p2 = rand(p2_params).p2 
                )

                true_state = step(fovtag_game, true_state, true_params)
                multi_update!(p2_belief, select(true_state, p2_ids...)[1], p2_params)
                multi_update!(p1_belief, select(true_state, p1_ids...)[1], p1_params)

                # render_fovtag(renderer, [
                #     (draw(p1_belief.dist; n=20, weight=true), p1_params[1]),
                #     (true_state, true_params),
                #     (draw(p2_belief.dist; n=20, weight=true), p2_params[1])
                #     ], fovtag_game, fov; T)
                roll!(scores, dist(true_state[1][:p1_pos], true_state[1][:p2_pos]), 50)

                open("gamma.txt", "a") do file
                    write(file, "cost: $(string(sum(scores)/length(scores)))\n")
                end
            end
        end
    end
end

function test_fovtag_active_info_gathering()
    n_sim_steps = 20
    for (p1_solver, p2_solver) in Iterators.product([:active; :inactive], [:active; :inactive])
        for s in 1:20
            Random.seed!(42+s)

            open("fovtag_$(p1_solver)_$(p2_solver).txt", "a") do file
                write(file, "trial: $s\n")
            end

            renderer = MakieRenderer()

            T = 6
            fov = (; p1=1.0, p2=1.0)

            optimization_options = (;
                n_lookahead = T,
                n_iters = 10,
                batch_size = 10,
                steps_per_seed = 1
            )

            clock_step = make_clock_step(1.0)

            p1_pos_hist_step = make_hist_step(:p1_pos_h, :p1_pos, T)
            p2_pos_hist_step = make_hist_step(:p2_pos_h, :p2_pos, T)
            p1_obs_hist_step = make_hist_step(:p1_obs_h, :p1_obs, T)
            p2_obs_hist_step = make_hist_step(:p2_obs_h, :p2_obs, T)
            p1_ang_hist_step = make_hist_step(:p1_θ_h, :p1_θ, T)
            p2_ang_hist_step = make_hist_step(:p2_θ_h, :p2_θ, T)
            p1_obs_hist_step_planning = (p1_solver == :active) ? p1_obs_hist_step : make_identity_step()
            p2_obs_hist_step_planning = (p2_solver == :active) ? p2_obs_hist_step : make_identity_step()


            p1_obs_hist_step = make_hist_step(:p1_obs_h, :p1_obs, T)
            p2_obs_hist_step = make_hist_step(:p2_obs_h, :p2_obs, T)

            p1_dynamics_step = make_vel_dynamics_step(:p1; control_scale=1.4)
            p2_dynamics_step = make_vel_dynamics_step(:p2; control_scale=2)

            p1_obs_step = make_fovtag_sensing_step(:p1, :p2; fov=fov[:p1])
            p2_obs_step = make_fovtag_sensing_step(:p2, :p1; fov=fov[:p2])

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
            ])

            planning_fovtag_game = ContinuousGame([
                clock_step,
                p1_obs_step,       p2_obs_step,
                p1_pos_hist_step,  p2_pos_hist_step,
                p1_obs_hist_step_planning,  p2_obs_hist_step_planning,
                p1_ang_hist_step,  p2_ang_hist_step,
                p1_control_step,   p2_control_step,
                p1_dynamics_step,  p2_dynamics_step,
            ])

            prior_fn = make_fovtag_prior(zero_state, n=1000)
            p1_belief = HybridParticleBelief(fovtag_game, prior_fn(), 0.1)
            p2_belief = HybridParticleBelief(fovtag_game, prior_fn(), 0.1)

            true_state = StateDist([prior_fn()[rand(1:10)]])
            p1_ids = [:t; :p1_pos; :p1_vel; :p1_θ; :p1_obs; :p1_pos_h; :p1_obs_h]
            p2_ids = [:t; :p2_pos; :p2_vel; :p2_θ; :p2_obs; :p2_pos_h; :p2_obs_h]

            cost_fns  = make_fovtag_costs()

            p1_params = (;
                    p1 = make_policy(4*T, 3; t_max=T),
                    p2 = make_policy(4*T, 3; t_max=T) 
                )

            p2_params = (;
                    p1 = make_policy(4*T, 3; t_max=T),
                    p2 = make_policy(4*T, 3; t_max=T)
                )

            true_params = (;
                    p1 = p1_params[:p1],
                    p2 = p2_params[:p2] 
            )

            scores = []
            for t in 1:n_sim_steps
                print(".")

                p1_params = solve(planning_fovtag_game, p1_belief, true_params, cost_fns, optimization_options)
                p2_params = solve(planning_fovtag_game, p2_belief, true_params, cost_fns, optimization_options)

                true_params = (;
                        p1 = p1_params[:p1],
                        p2 = p2_params[:p2] 
                )

                true_state = step(fovtag_game, true_state, true_params)
                update!(p1_belief, select(true_state, p1_ids...)[1], p1_params)
                update!(p2_belief, select(true_state, p2_ids...)[1], p2_params)

                # render_fovtag(renderer, [
                #     (draw(p1_belief.dist; n=20, weight=true), p1_params[1]),
                #     (true_state, true_params),
                #     (draw(p2_belief.dist; n=20, weight=true), p2_params[1])
                #     ], fovtag_game, fov; T)
                roll!(scores, dist(true_state[1][:p1_pos], true_state[1][:p2_pos]), 50)

                open("fovtag_$(p1_solver)_$(p2_solver).txt", "a") do file
                    write(file, "cost: $(string(sum(scores)/length(scores)))\n")
                end
            end
        end
    end
end

function test_fovtag_equilibrium_selection()
    n_sim_steps = 100

    for (p1_solves, p2_solves) in Iterators.product(1:4, 1:4)

        for s in 1:5
            Random.seed!(42+s)

            open("multi_eq.txt", "a") do file
                write(file, "P1: $p1_solves solves; P2: $p2_solves; trial: $s\n")
            end

            renderer = MakieRenderer()

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


            clock_step = make_clock_step(1.0)

            p1_pos_hist_step = make_hist_step(:p1_pos_h, :p1_pos, T)
            p2_pos_hist_step = make_hist_step(:p2_pos_h, :p2_pos, T)
            p1_obs_hist_step = make_hist_step(:p1_obs_h, :p1_obs, T)
            p2_obs_hist_step = make_hist_step(:p2_obs_h, :p2_obs, T)
            p1_ang_hist_step = make_hist_step(:p1_θ_h, :p1_θ, T)
            p2_ang_hist_step = make_hist_step(:p2_θ_h, :p2_θ, T)

            p1_obs_hist_step = make_hist_step(:p1_obs_h, :p1_obs, T)
            p2_obs_hist_step = make_hist_step(:p2_obs_h, :p2_obs, T)

            p1_dynamics_step = make_vel_dynamics_step(:p1; control_scale=1.4)
            p2_dynamics_step = make_vel_dynamics_step(:p2; control_scale=2)

            p1_obs_step = make_fovtag_sensing_step(:p1, :p2; fov=fov[:p1])
            p2_obs_step = make_fovtag_sensing_step(:p2, :p1; fov=fov[:p2])

            p1_control_step = make_nn_control(:p1, [:p1_pos_h, :p1_obs_h, :p2_θ_h], :p1_vel, t_max=T)
            p2_control_step = make_nn_control(:p2, [:p2_pos_h, :p2_obs_h, :p1_θ_h], :p2_vel, t_max=T)

            attr_ids = [:pos, :vel, :acc, :θ, :vω, :obs, :pos_h, :obs_h, :θ_h]
            attr_w   = [ 2;    2;    2;    1;  2;   2;    2*T;    2*T;    T]

            zero_state = State([
                :t => 1;
                Symbol.(:p1, :_, attr_ids) .=> attr_w;
                Symbol.(:p2, :_, attr_ids) .=> attr_w;
            ]...)

            
            # We're assuming active information gathering,
            #   so planning and true game are the same.
            fovtag_game = ContinuousGame([
                clock_step,
                p1_obs_step,       p2_obs_step,
                p1_pos_hist_step,  p2_pos_hist_step,
                p1_obs_hist_step,  p2_obs_hist_step,
                p1_ang_hist_step,  p2_ang_hist_step,
                p1_control_step,   p2_control_step,
                p1_dynamics_step,  p2_dynamics_step,
            ])


            prior_fn = make_fovtag_prior(zero_state, n=1000)
            p1_belief = HybridParticleBelief(fovtag_game, prior_fn(), 0.1)
            p2_belief = HybridParticleBelief(fovtag_game, prior_fn(), 0.1)

            true_state = StateDist([prior_fn()[rand(1:10)]])
            p1_ids = [:t; :p1_pos; :p1_vel; :p1_θ; :p1_obs; :p1_pos_h; :p1_obs_h]
            p2_ids = [:t; :p2_pos; :p2_vel; :p2_θ; :p2_obs; :p2_pos_h; :p2_obs_h]

            cost_fns  = make_fovtag_costs()

            # This is NOT shared brain. 

            p1_policy = make_policy(5*T, 3; t_max=T)
            p2_policy = make_policy(5*T, 3; t_max=T)

            p1_params = map(1:p1_solves) do _
                (;
                    p1 = p1_policy,
                    p2 = make_policy(5*T, 3; t_max=T) 
                )
            end

            p2_params = map(1:p2_solves) do _
                (;
                    p1 = make_policy(5*T, 3; t_max=T),
                    p2 = p2_policy
                )
            end

            scores = []
            for t in 1:n_sim_steps
                println(".")

                p1_params = map(p1_params) do params
                    solve(fovtag_game, p1_belief, params, cost_fns, optimization_options)
                end

                p2_params = map(p2_params) do params
                    solve(fovtag_game, p2_belief, params, cost_fns, optimization_options)
                end

                true_params = (;
                    p1 = rand(p1_params).p1,
                    p2 = rand(p2_params).p2 
                )

                true_state = step(fovtag_game, true_state, true_params)
                multi_update!(p2_belief, select(true_state, p2_ids...)[1], p2_params)
                multi_update!(p1_belief, select(true_state, p1_ids...)[1], p1_params)

                # render_fovtag(renderer, [
                #     (draw(p1_belief.dist; n=20, weight=true), p1_params[1]),
                #     (true_state, true_params),
                #     (draw(p2_belief.dist; n=20, weight=true), p2_params[1])
                #     ], fovtag_game, fov; T)
                scores = [scores; dist(true_state[1][:p1_pos], true_state[1][:p2_pos])]
<<<<<<< HEAD
                # println("$t\t" * string(dist(true_state[1][:p1_pos], true_state[1][:p2_pos])))
=======
            end
            println(sum(scores)/n_steps)
        end
    end
end

function test_fovtag_multi_policy_robustness()
    n_solves = 4
    n_sim_steps = 1000

    for s in 1:20

        renderer = MakieRenderer()

        T = 7
        fov = (; p1=4.0, p2=1.0)

        optimization_options = (;
            n_lookahead = T,
            n_iters = 10,
            batch_size = 5,
            max_wall_time = 120000, # Not respected right now
            steps_per_seed = 1,
            steps_per_render = 10
        )

        Random.seed!(42+s)

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

        p1_obs_step = make_fovtag_sensing_step(:p1, :p2; fov=fov[:p1])
        p2_obs_step = make_fovtag_sensing_step(:p2, :p1; fov=fov[:p2])

        p1_control_step = make_nn_control(:p1, [:p1_pos_h, :p1_obs_h, :p2_θ_h], :p1_vel, t_max=T)
        p2_control_step = make_nn_control(:p2, [:p2_pos_h, :p2_obs_h, :p1_θ_h], :p2_vel, t_max=T)

        attr_ids = [:pos, :vel, :acc, :θ, :vω, :obs, :pos_h, :obs_h, :θ_h]
        attr_w   = [ 2;    2;    2;    1;  2;   2;    2*T;    2*T;    T]

        zero_state = State([
            :t => 1;
            Symbol.(:p1, :_, attr_ids) .=> attr_w;
            Symbol.(:p2, :_, attr_ids) .=> attr_w;
        ]...)

        
        # We're assuming active information gathering,
        #   so planning and true game are the same.
        fovtag_game = ContinuousGame([
            clock_step,
            p1_obs_step,       p2_obs_step,
            p1_pos_hist_step,  p2_pos_hist_step,
            p1_obs_hist_step,  p2_obs_hist_step,
            p1_ang_hist_step,  p2_ang_hist_step,
            p1_control_step,   p2_control_step,
            p1_dynamics_step,  p2_dynamics_step,
        ])


        prior_fn = make_fovtag_prior(zero_state, n=1000)
        # p2_belief = JointParticleBelief(fovtag_game, prior_fn())
        # p1_belief = JointParticleBelief(fovtag_game, prior_fn())
        p1_belief = HybridParticleBelief(fovtag_game, prior_fn(), 0.4)
        p2_belief = HybridParticleBelief(fovtag_game, prior_fn(), 0.01)

        true_state = StateDist([prior_fn()[rand(1:10)]])
        p1_ids = [:t; :p1_pos; :p1_vel; :p1_θ; :p1_obs; :p1_pos_h; :p1_obs_h]
        p2_ids = [:t; :p2_pos; :p2_vel; :p2_θ; :p2_obs; :p2_pos_h; :p2_obs_h]

        cost_fns  = make_fovtag_costs()

        # This is NOT shared brain. 

        p1_policy = make_policy(5*T, 3; t_max=T)
        p2_policy = make_policy(5*T, 3; t_max=T)

        p1_params = map(1:n_solves) do _
            (;
                p1 = p1_policy,
                p2 = make_policy(5*T, 3; t_max=T) 
            )
        end

        p2_params = map(1:n_solves) do _
            (;
                p1 = make_policy(5*T, 3; t_max=T),
                p2 = p2_policy
            )
        end

        scores = []
        for t in 1:n_sim_steps

            p1_params = map(p1_params) do params
                println()
                print("1")
                solve(fovtag_game, p1_belief, params, cost_fns, optimization_options) do _
                    print(".")
                    return false
                end
            end

            p2_params = map(p2_params) do params
                println()
                print("2")
                solve(fovtag_game, p2_belief, params, cost_fns, optimization_options) do _
                    print(".")
                    return false
                end
            end

            true_params = (;
                p1 = rand(p1_params).p1,
                p2 = rand(p2_params).p2 
            )

            true_state = step(fovtag_game, true_state, true_params)
            multi_update!(p2_belief, select(true_state, p2_ids...)[1], p2_params)
            multi_update!(p1_belief, select(true_state, p1_ids...)[1], p1_params)
            # multi_update!(p2_belief, p2_params)
            # multi_update!(p1_belief, p1_params)

            render_fovtag(renderer, [
                (draw(p1_belief.dist; n=20, weight=true), rand(p1_params)),
                (true_state, true_params),
                (draw(p2_belief.dist; n=20, weight=true), rand(p2_params))
                ], fovtag_game, fov; T)
            scores = [scores; dist(true_state[1][:p1_pos], true_state[1][:p2_pos])]
            # println("$t\t" * string(dist(true_state[1][:p1_pos], true_state[1][:p2_pos])))
        end
        println(sum(scores)/n_steps)
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

            render_static_circle(renderer, (0, 0), 45; ax_idx=(1, col))

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

                render_trajectory(renderer, history, :p1_pos; 
                    ax_idx=(1, col), color=0, alpha=0.5*lik,
                    colormap=p1_colormap, colorrange)
                render_trajectory(renderer, history, :p2_pos; 
                    ax_idx=(1, col), color=0, alpha=0.5*lik,
                    colormap=p2_colormap, colorrange)
>>>>>>> hastag
                

                p1_retrokl = string(approx_kl(p2_belief.dist, p1_belief.dist, :p1_pos))
                p2_retrokl = string(approx_kl(p1_belief.dist, p2_belief.dist, :p2_pos))
                p1_targkl  = string(approx_kl(p2_belief.dist, p1_belief.dist, :p2_pos))
                p2_targkl  = string(approx_kl(p1_belief.dist, p2_belief.dist, :p1_pos))

                p1_surp1 = string(approx_surprisal(p1_belief.dist, true_state[1], :p1_pos))
                p1_surp2 = string(approx_surprisal(p1_belief.dist, true_state[1], :p2_pos))
                p2_surp1 = string(approx_surprisal(p2_belief.dist, true_state[1], :p1_pos))
                p2_surp2 = string(approx_surprisal(p2_belief.dist, true_state[1], :p2_pos))

                open("multi_eq.txt", "a") do file
                    write(file, "$p1_retrokl \t$p2_retrokl \t$p1_targkl \t$p2_targkl \t$p1_surp1 \t$p1_surp2 \t$p2_surp1 \t$p2_surp2\n")
                end
            end


            open("multi_eq.txt", "a") do file
                write(file, "cost: $(string(sum(scores)/n_sim_steps))\n")
            end
        end
    end
end