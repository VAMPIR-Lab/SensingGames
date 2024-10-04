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

    

    #dθ is the angle from the center of the fov angle to the opponent.
    function get_sensing_noise(dθ)
        d = (fov/2 - abs(dθ))
        offset + d * ((d > 0) ? (-0.001) : (-scale))
    end

    function get_gaussian_params(state_dist)
        our_pos = state_dist[id_own_pos]
        # μ = NamedTuple()
        # σ2 = NamedTuple()

        # This returns a vector with (μ, σ2) pairs.
        map(id_other_pos) do i
            their_pos = state_dist[i]
            θ1 = state_dist[id_own_θ] .+ π
            θ2 = atan.(our_pos[:, 2] .- their_pos[:, 2], our_pos[:, 1] .- their_pos[:, 1])
            dθ = angdiff.(θ1, θ2)
            μ = their_pos
            σ2 = get_sensing_noise.(dθ).^2
            (μ, σ2)
            # μ = Base.merge(μ, (i => state_dist[i],))
            # σ2 = Base.merge(σ2, (i => get_sensing_noise.(dθ).^2,))
        end

        
        # return gaussian_params
    end

    function rollout_observation(state_dist::StateDist, game_params)
        gaussian_params = get_gaussian_params(state_dist)
        # println(length(μ))
        # println(length(σ2))
        # obs = Matrix{Float64}(undef, 20, 2 * length(id_other_pos))
        # obs = Matrix{Float64}(undef, 20, 0)
        # count = 1
        # println(id_other_pos)

        obs = mapreduce(hcat, enumerate(id_other_pos)) do (i, id) 
            sample_trimmed_gauss.(gaussian_params[i][1], gaussian_params[i][2]; l=-40, u=40)
        end

        # for i in id_other_pos
        #     # println(count)
        #     # println(size(sample_trimmed_gauss.(μ[i], σ2[i]; l=-40, u=40)))
        #     obs_i = sample_trimmed_gauss.(μ[i], σ2[i]; l=-40, u=40)
        #     # println(size(obs_i))
            
        #     obs = hcat(obs, obs_i)
        #     # obs[:, (2*count-1):(2*count)] = obs_i
        #     # println(22222222222222222)
        #     count = count + 1
        # end
        alter(state_dist, 
            id_obs => obs
        )
    end

    function lik_observation(state_dist, compare_dist, game_params) 
        μ, σ2 = get_gaussian_params(state_dist)   
        x = compare_dist[id_obs]
        # x:     (|info| x 2) => (1 x |info| x 2)
        x = reshape(x, (1, length(compare_dist), 2))
        # lik = NamedTuple()
        map(id_other_pos) do i 
            μ_i = reshape(μ[i], (length(state_dist), 1, 2))
            σ2_i = reshape(σ2[i], (length(state_dist), 1, 1))
            p = prod(SensingGames.gauss_pdf.(x, μ_i, σ2_i), dims=3)
            log.(p .+ 1e-10)
        end
        # for i in id_other_pos
        #     # mu:    (|dist| x 2) => (|dist| x 1 x 2)
        #     # sigma: (|dist| x 1) => (|dist| x 1 x 1)
        #     μ_i = reshape(μ[i], (length(state_dist), 1, 2))
        #     σ2_i = reshape(σ2[i], (length(state_dist), 1, 1))
        #     # Note that we use the true Gaussian PDF (not the trimmed one)
        #     p = prod(SensingGames.gauss_pdf.(x, μ_i, σ2_i), dims=3)
        #     # lik = Base.merge(lik, (i => log.(p .+ 1e-10),))
        # end        
        # lik
    end

    GameComponent(rollout_observation, lik_observation, [id_obs])
end

function make_multitag_costs(num_p, num_e)
    costs = NamedTuple()

    # pursuer costs
    for i = 1:num_p
        p = Symbol("p$(i)")
        p_pos = Symbol("p$(i)_pos")
        p_vω = Symbol("p$(i)_vω")
        function cost_p(hist)
            # distr is an element in hist
            sum(hist) do distr
                # d is the squared distances between p1_pos and p2_pos.
                d = 0
                for l = 1:num_e
                    e_pos = Symbol("e$(l)_pos")
                    d = d .+ sum((distr[p_pos] .- distr[e_pos]).^2, dims=2)
                end
                d = d / num_p
                b = cost_bound.(distr[p_pos], [-40 -40], [40 40])
                r = cost_regularize.(distr[p_vω], α=10)
                sum((b .+ r .+ d) .* exp.(distr.w))
            end
        end
        costs = Base.merge(costs, (p => cost_p,))
    end

    # evader costs
    for k = 1:num_e
        e = Symbol("e$(k)")
        e_pos = Symbol("e$(k)_pos")
        e_vω = Symbol("e$(k)_vω")
        function cost_e(hist)
            # distr is an element in hist
            sum(hist) do distr
                # d is the squared distances between p1_pos and p2_pos.
                d = 0
                for l = 1:num_p
                    p_pos = Symbol("p$(l)_pos")
                    d = d .- sum((distr[p_pos] .- distr[e_pos]).^2, dims=2)
                end
                d = d / num_p
                b = cost_bound.(distr[e_pos], [-40 -40], [40 40])
                r = cost_regularize.(distr[e_vω], α=10)
                sum((b .+ r .+ d) .* exp.(distr.w))
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
    T = 5
    n_particles=1000

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
        n_iters = 1,
        batch_size = 20,
        max_wall_time = 120000, # Not respected right now
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

    for i = 1:num_p
        p = Symbol("p$(i)")

        # pos_hist_step
        p_pos_h = Symbol("p$(i)_pos_h")
        p_pos = Symbol("p$(i)_pos")
        pos_hist_step = Base.merge(pos_hist_step, (p => make_hist_step(p_pos_h, p_pos, T),))

        # obs_hist_step
        p_obs_h = Symbol("p$(i)_obs_h")
        p_obs = Symbol("p$(i)_obs")
        obs_hist_step = Base.merge(obs_hist_step, (p => make_hist_step(p_obs_h, p_obs, T),))

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
        obs_hist_step = Base.merge(obs_hist_step, (e => make_hist_step(e_obs_h, e_obs, T),))

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

    prior_fn = make_multitag_prior(zero_state, num_p, num_e, n=n_particles)

    # Shared brain: Everyone agrees on the belief
    joint_belief = JointParticleBelief(multitag_game, prior_fn())

    # <Mel> Aside: I added some explanation of what these <X>ParticleBeliefs mean
    #   over in `beliefs.jl`.

    # Separate brains: Everyone has their own belief
    # beliefs = NamedTuple()
    # for i = 1:(num_p)
    #     p = Symbol("p$(i)")
    #     beliefs = Base.merge(beliefs, (p => HybridParticleBelief(fovtag_game, prior_fn(), 0.01),))
    # end
    # for j = 1:(num_e)
    #     e = Symbol("e$(j)")
    #     beliefs = Base.merge(beliefs, (e => HybridParticleBelief(fovtag_game, prior_fn(), 0.01),))
    # end


    true_state = make_multitag_prior(zero_state, num_p, num_e, n=1)() # Just updated; not tested

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
    renderer = MakieRenderer()


    # In split brain we would need one of these `params` tuples for EVERY player
    #   (each of which would contain policy parameters for every player)
    params = NamedTuple()
    for i = 1:(num_p)
        p = Symbol("p$(i)")
        params = Base.merge(params, (p => make_policy(2*(num_p+num_e)*T, 3; t_max=T),))
    end
    for j = 1:(num_e)
        e = Symbol("e$(j)")
        params = Base.merge(params, (e => make_policy(2*(num_p+num_e)*T, 3; t_max=T),))
    end

    # params = (;
    #     p1 = make_policy(4*T, 3; t_max=T),
    #     e1 = make_policy(4*T, 3; t_max=T)
    # )

    for t in 1:1000

        # <Mel> The do block on `solve` is optional and what had been here was cruft (my bad)
        params = solve(multitag_game, joint_belief, params, cost_fns, optimization_options)

        true_state = step(multitag_game, true_state, params)
        println(true_state)
        # Shared brain
        update(joint_belief, params)

        # Separate brains
        # for i = 1:(num_p)
        #     p = Symbol("p$(i)")
        #     update(beliefs[p], select(true_state, ids[p]...)[1], params[p])
        # end
        # for j = 1:(num_e)
        #     e = Symbol("e$(j)")
        #     update(beliefs[e], select(true_state, ids[e]...)[1], params[e])
        # end
        
        # render_multitag(renderer, [
        #     (draw(p1_belief; n=20), true_params),
        #     (true_state, true_params),
        #     (draw(p2_belief; n=20), true_params)
        #     ], multitag_game, fov; T)
    end
end

function render_multitag(renderer, dists, game, fov; T)

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