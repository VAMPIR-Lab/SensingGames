using Dates

# Chain tag is an N-player, non-zero-sum variant of FOV tag.

function make_multitag_sensing_step(agent, num_p, num_e; fov, scale=100.0, offset=2)
    id_obs = Symbol("$(agent)_obs")
    id_own_θ = Symbol("$(agent)_θ")
    id_own_pos = Symbol("$(agent)_pos")
    id_other_pos = Symbol[]
    for i = 1:num_p
        p = Symbol("p$(i)")
        if p != agent
            id_pos = Symbol("p$(i)_pos")
            push!(id_other_pos, id_pos)
        end
    end
    for j = 1:num_e
        e = Symbol("e$(j)")
        if e != agent
            id_pos = Symbol("e$(j)_pos")
            push!(id_other_pos, id_pos)
        end
    end

    function get_sensing_noise(dθ)
        d = (fov/2 - abs(dθ))
        offset + d * ((d > 0) ? (-0.001) : (-scale))
    end

    function get_gaussian_params(state_dist)
        our_pos = state_dist[id_own_pos]

        map(id_other_pos) do i
            their_pos = state_dist[i]
            θ1 = state_dist[id_own_θ] .+ π
            θ2 = atan.(our_pos[:, 2] .- their_pos[:, 2], our_pos[:, 1] .- their_pos[:, 1])
            dθ = angdiff.(θ1, θ2)
            μ = their_pos
            σ2 = get_sensing_noise.(dθ).^2
            (μ, σ2)
        end
    end

    function rollout_observation(state_dist::StateDist, game_params)
        gaussian_params = get_gaussian_params(state_dist)

        obs = mapreduce(hcat, enumerate(id_other_pos)) do (i, id) 
            sample_trimmed_gauss.(gaussian_params[i][1], gaussian_params[i][2]; l=-40, u=40)
        end

        alter(state_dist, 
            id_obs => obs
        )
    end

    function lik_observation(state_dist, compare_dist, game_params) 
        μ, σ2 = get_gaussian_params(state_dist)   
        x = compare_dist[id_obs]
        x = reshape(x, (1, length(compare_dist), 2))
        map(id_other_pos) do i 
            μ_i = reshape(μ[i], (length(state_dist), 1, 2))
            σ2_i = reshape(σ2[i], (length(state_dist), 1, 1))
            p = prod(SensingGames.gauss_pdf.(x, μ_i, σ2_i), dims=3)
            log.(p .+ 1e-10)
        end
    end

    GameComponent(rollout_observation, lik_observation, [id_obs])
end

function make_multitag_simple_costs(num_p, num_e)
    costs = NamedTuple()
   
    function cost_bound(d)
        dd = (d .- 40^2)
        dg = (dd .> 0)
        dl = (dd .< 0)
        k = (dl .* (0.001*d)) .+ (dg .* d.^4)
    end

    # pursuer costs
    for i = 1:num_p
        target = (i+1) % num_e + 1
        p = Symbol("p$(i)")
        p_pos = Symbol("p$(i)_pos")
        e_pos = Symbol("e$(target)_pos")
        function cost_p(hist)
            sum(hist) do distr
                sum((
                    sqrt.(sum((distr[p_pos] .- distr[e_pos]).^2, dims=2))
                ) .* exp.(distr.w))
            end
        end
        

        costs = Base.merge(costs, (p => cost_p,))
    end

    # evader costs
    for i = 1:num_e
        target = (i) % num_p + 1
        e = Symbol("e$(i)")
        p_pos = Symbol("p$(target)_pos")
        e_pos = Symbol("e$(i)_pos")
        function cost_e(hist)
            sum(hist) do distr
                sum((
                    -sqrt.(sum((distr[p_pos] .- distr[e_pos]).^2, dims=2))
                ) .* exp.(distr.w))
            end
        end
        costs = Base.merge(costs, (e => cost_e,))
    end
    costs
end

function make_multitag_costs(num_p, num_e)
    costs = NamedTuple()
   
    function cost_bound(d)
        dd = (d .- 40^2)
        dg = (dd .> 0)
        dl = (dd .< 0)
        k = (dl .* (0.001*d)) .+ (dg .* d.^4)
    end

    # pursuer costs
    for i = 1:num_p
        target = (i+1) % num_e + 1
        p = Symbol("p$(i)")
        p_pos = Symbol("p$(i)_pos")
        e_pos = Symbol("e$(target)_pos")
        function cost_p(hist)
            sum(hist) do distr
                sum((
                    sqrt.(sum((distr[p_pos] .- distr[e_pos]).^2, dims=2)) .+
                    cost_bound(sum((distr[p_pos]).^2, dims=2))
                ) .* exp.(distr.w))
            end
        end
        

        costs = Base.merge(costs, (p => cost_p,))
    end

    # evader costs
    for i = 1:num_e
        target = (i) % num_p + 1
        e = Symbol("e$(i)")
        p_pos = Symbol("p$(target)_pos")
        e_pos = Symbol("e$(i)_pos")
        function cost_e(hist)
            sum(hist) do distr
                sum((
                    -sqrt.(sum((distr[p_pos] .- distr[e_pos]).^2, dims=2)) .+
                    cost_bound(sum((distr[e_pos]).^2, dims=2))
                ) .* exp.(distr.w))
            end
        end
        costs = Base.merge(costs, (e => cost_e,))
    end
    # This is a named tuple that contains the cost function for all players, with keys being p1, p2, ... and e1, e2, ...
    costs
end

function make_multitag_prior(zero_state, num_p, num_e; n=16)
    function prior()
        prior = StateDist(zero_state, n)
        for i = 1:num_p
            p_pos = Symbol("p$(i)_pos")
            p_vel = Symbol("p$(i)_vel")
            p_θ = Symbol("p$(i)_θ")
            prior = alter(prior, 
            p_pos => rand(n, 2)*30 .- 15,
            p_vel => rand(n, 2)*0.01,
            p_θ => rand(n, 1)*2π
            )
        end
        
        for j = 1:num_e
            e_pos = Symbol("e$(j)_pos")
            e_vel = Symbol("e$(j)_vel")
            e_θ = Symbol("e$(j)_θ")
            prior = alter(prior, 
            e_pos => rand(n, 2)*30 .- 15,
            e_vel => rand(n, 2)*0.01,
            e_θ => rand(n, 1)*2π
            )
        end
        prior
    end
end

function test_multitag(num_p=2, num_e=2)
    for solvers in [
        (:active, :active, :active, :active),
        (:active, :active, :inactive, :inactive),
        (:inactive, :inactive, :active, :active),
        (:inactive, :inactive, :inactive, :inactive)]

        open("multitag.txt", "a") do file
            write(file, "=======\nP1 $(solvers[1]), P2 $(solvers[2]), E1 $(solvers[3]), E2 $(solvers[4])\n")
        end

        for s = 1:20
            open("multitag.txt", "a") do file
                write(file, "Trial $s\n")
            end
            Random.seed!(42+s)

            T = 6
            n_particles=10000

            fov = NamedTuple()
            for i = 1:num_p
                p = Symbol("p$(i)")
                fov = Base.merge(fov, (p => 1.0,))
            end
            for j = 1:num_e
                e = Symbol("e$(j)")
                fov = Base.merge(fov, (e => 1.0,))
            end


            optimization_options = (;
                n_lookahead = T,
                n_iters = 10,
                batch_size = 5,
                steps_per_seed = 1,
                steps_per_render = 10
            )

            clock_step = make_clock_step(1.0)
            pos_hist_step = NamedTuple()
            obs_hist_step = NamedTuple()
            ang_hist_step = NamedTuple()
            dyn_step = NamedTuple()
            bound_step = NamedTuple()
            obs_step = NamedTuple()
            control_step = NamedTuple()
            obs_hist_step_planning = NamedTuple()

            for i = 1:num_p
                p = Symbol("p$(i)")

                # pos_hist_step
                p_pos_h = Symbol("p$(i)_pos_h")
                p_pos = Symbol("p$(i)_pos")
                pos_hist_step = Base.merge(pos_hist_step, (p => make_hist_step(p_pos_h, p_pos, T),))

                # obs_hist_step
                p_obs_h = Symbol("p$(i)_obs_h")
                p_obs = Symbol("p$(i)_obs")
                p_obs_hist_step = make_hist_step(p_obs_h, p_obs, T)
                obs_hist_step = Base.merge(obs_hist_step, (p => p_obs_hist_step,))

                # obs_hist_step_planning
                solver = solvers[i]
                p_obs_hist_step_planning = (solver == :active) ? p_obs_hist_step : make_identity_step()
                obs_hist_step_planning = Base.merge(obs_hist_step_planning, (p => p_obs_hist_step_planning,))

                # ang_hist_step
                p_θ_h = Symbol("p$(i)_θ_h")
                p_θ = Symbol("p$(i)_θ")
                ang_hist_step = Base.merge(ang_hist_step, (p => make_hist_step(p_θ_h, p_θ, T),))

                # dyn_step
                dyn_step = Base.merge(dyn_step, (p => make_vel_dynamics_step(p; control_scale=1.4),))

                # bound_step
                bound_step = Base.merge(bound_step, (p => make_bound_step(p_pos, -42, 42),))

                # obs_step
                obs_step = Base.merge(obs_step, (p => make_multitag_sensing_step(p, num_p, num_e; fov=fov[p]),))

                # control_step
                p_vel = Symbol("p$(i)_vel")
                control_step = Base.merge(control_step, (p => make_nn_control(p, [p_pos_h, p_obs_h], p_vel, t_max=T),))

            end

            for j = 1:num_e
                e = Symbol("e$(j)")

                # pos_hist_step
                e_pos_h = Symbol("e$(j)_pos_h")
                e_pos = Symbol("e$(j)_pos")
                pos_hist_step = Base.merge(pos_hist_step, (e => make_hist_step(e_pos_h, e_pos, T),))

                # obs_hist_step
                e_obs_h = Symbol("e$(j)_obs_h")
                e_obs = Symbol("e$(j)_obs")
                e_obs_hist_step = make_hist_step(e_obs_h, e_obs, T)
                obs_hist_step = Base.merge(obs_hist_step, (e => e_obs_hist_step,))

                # obs_hist_step_planning
                solver = solvers[num_p + j]
                e_obs_hist_step_planning = (solver == :active) ? e_obs_hist_step : make_identity_step()
                obs_hist_step_planning = Base.merge(obs_hist_step_planning, (e => e_obs_hist_step_planning,))

                # ang_hist_step
                e_θ_h = Symbol("e$(j)_θ_h")
                e_θ = Symbol("e$(j)_θ")
                ang_hist_step = Base.merge(ang_hist_step, (e => make_hist_step(e_θ_h, e_θ, T),))

                # dyn_step
                dyn_step = Base.merge(dyn_step, (e => make_vel_dynamics_step(e; control_scale=2),))

                # bound_step
                bound_step = Base.merge(bound_step, (e => make_bound_step(e_pos, -42, 42),))

                # obs_step
                obs_step = Base.merge(obs_step, (e => make_multitag_sensing_step(e, num_p, num_e; fov=fov[e]),))

                # control_step
                e_vel = Symbol("e$(j)_vel")
                control_step = Base.merge(control_step, (e => make_nn_control(e, [e_pos_h, e_obs_h], e_vel, t_max=T),))

            end

            attr_ids = [:pos, :vel, :acc, :θ, :vω, :obs,                 :pos_h, :obs_h,                 :θ_h]
            attr_w   = [ 2;    2;    2;    1;  2;   2*(num_p+num_e-1);    2*T;    2*(num_p+num_e-1)*T;    T]
            zero_state_arg = [:t => 1]
            for i = 1:num_p
                p = Symbol("p$(i)")
                append!(zero_state_arg, Symbol.(p, :_, attr_ids) .=> attr_w)
            end
            for j = 1:num_e
                e = Symbol("e$(j)")
                append!(zero_state_arg, Symbol.(e, :_, attr_ids) .=> attr_w)
            end

            zero_state = State(zero_state_arg...)


            multitag_game = ContinuousGame([
                clock_step,
                # These are all named tuples (player symbol => func)
                values(obs_step)...,
                values(pos_hist_step)...,
                values(obs_hist_step)...,
                values(ang_hist_step)...,
                values(control_step)...,
                values(dyn_step)...,
            ])

            planning_multitag_game = ContinuousGame([
                clock_step,
                # These are all named tuples (player symbol => func)
                values(obs_step)...,
                values(pos_hist_step)...,
                values(obs_hist_step_planning)...,
                values(ang_hist_step)...,
                values(control_step)...,
                values(dyn_step)...,
            ])

            prior_fn = make_multitag_prior(zero_state, num_p, num_e, n=n_particles)

            # Shared brain: Everyone agrees on the belief
            joint_belief = JointParticleBelief(multitag_game, prior_fn())

            true_state = make_multitag_prior(zero_state, num_p, num_e, n=1)()
            hist = [true_state]

            ids = NamedTuple()
            for i = 1:(num_p)
                p = Symbol("p$(i)")
                p_pos = Symbol("p$(i)_pos")
                p_vel = Symbol("p$(i)_vel")
                p_θ = Symbol("p$(i)_θ")
                p_obs = Symbol("p$(i)_obs")
                p_pos_h = Symbol("p$(i)_pos_h")
                p_obs_h = Symbol("p$(i)_obs_h")
                ids = Base.merge(ids, (p => [:t; p_pos; p_vel; p_θ; p_obs; p_pos_h; p_obs_h],))
            end
            for j = 1:(num_e)
                e = Symbol("e$(j)")
                e_pos = Symbol("e$(j)_pos")
                e_vel = Symbol("e$(j)_vel")
                e_θ = Symbol("e$(j)_θ")
                e_obs = Symbol("e$(j)_obs")
                e_pos_h = Symbol("e$(j)_pos_h")
                e_obs_h = Symbol("e$(j)_obs_h")
                ids = Base.merge(ids, (e => [:t; e_pos; e_vel; e_θ; e_obs; e_pos_h; e_obs_h],))
            end

            cost_fns  = make_multitag_costs(num_p, num_e)
            simple_cost_fns  = make_multitag_simple_costs(num_p, num_e)
            renderer = MakieRenderer()

            params = NamedTuple()
            for i = 1:(num_p)
                p = Symbol("p$(i)")
                params = Base.merge(params, (p => make_policy(2*(num_p+num_e)*T, 3; t_max=T),))
            end
            for j = 1:(num_e)
                e = Symbol("e$(j)")
                params = Base.merge(params, (e => make_policy(2*(num_p+num_e)*T, 3; t_max=T),))
            end
            scores_active = []
            scores_inactive = []

            n_steps = 20
            for t in 1:n_steps
                print(t)

                params = solve(planning_multitag_game, joint_belief, params, cost_fns, optimization_options) do params
                    print(".")
                    return false
                end
                println()

                true_state = step(multitag_game, true_state, params)
                hist = [hist; true_state]

                # Fully shared brain
                update!(joint_belief, params)

                scores_active = [scores_active; min_dist([true_state[1][:p1_pos], (true_state[1][Symbol("e$(m)_pos")] for m in 1:num_e)...])]
                scores_inactive = [scores_inactive; min_dist([true_state[1][:p2_pos], (true_state[1][Symbol("e$(n)_pos")] for n in 1:num_e)...])]

                render_multitag(renderer, [
                    (true_state, params)
                    ], multitag_game, keys(fov), fov; T)
                
            end
            open("multitag.txt", "a") do file
                for player in keys(simple_cost_fns)
                    write(file, "$(string(player)): $(string(simple_cost_fns[player](hist)/n_steps))\n")
                end
            end
        end
    end
end

function render_multitag(renderer, dists, game, players, fov; T)

    unspool_scheme = mapreduce(vcat, players) do id
        [
            Symbol("$(id)_pos_h") => Symbol("$(id)_pos"),
            Symbol("$(id)_obs_h") => Symbol("$(id)_obs"),
            Symbol("$(id)_θ_h") => Symbol("$(id)_θ")
        ]
    end


    render(renderer) do
        for (col, (dist, params)) in enumerate(dists)

            plan = step(game, dist, params; n=T)[end]
            render_static_circle(renderer, (0, 0), 45; ax_idx=(1, col))

            for i in 1:length(dist)
                current_state = dist[i] |> cpu
                lik = exp(cpu(dist.w)[i])

                for (p_num, player) in enumerate(players)

                    id_pos = Symbol("$(player)_pos")
                    id_θ = Symbol("$(player)_θ")

                    colormap = :seismic
                    color = if 'p' in String(player)
                        -p_num
                    else
                        p_num
                    end

                    colorrange = (-length(players)+1, length(players)+1)
                    past = unspool(current_state, unspool_scheme...)
                    future_particle = plan[i] |> cpu
                    future = unspool(future_particle, unspool_scheme...)
                    history = [past; future]

                    render_trajectory(renderer, history, id_pos; 
                        ax_idx=(1, col), color, alpha=lik,
                        colormap, colorrange, linewidth=4)
                    
                    for (i, state) in enumerate(history)
                        t = i-T
                        render_fov(renderer, state, fov[player], id_pos, id_θ; 
                            ax_idx=(1, col), color, alpha=0.1*lik,
                            colormap, colorrange)
                        t += 1
                    end

                    render_location(renderer, current_state, id_pos; 
                        ax_idx=(1, col), color=:black, alpha=1.0*lik,
                        colormap, colorrange, markersize=8)

                end
            end
        end
    end
end