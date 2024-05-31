using LinearAlgebra
using Statistics
using Dates
using PythonCall

function make_blurtag_sensing(agent, other; blur_coef)
    id_obs = Symbol("$(agent)_obs")
    id_own_pos = Symbol("$(agent)_pos")
    id_own_vel = Symbol("$(agent)_vel")
    id_own_θ = Symbol("$(agent)_θ")
    id_other_pos = Symbol("$(other)_pos")

    state = State(
        id_obs => 4
    )
    
    function dyn!(state::State, history::Vector{State}, game_params)::State
        our_pos = state[id_own_pos]
        their_pos = state[id_other_pos]

        our_vel = state[id_own_vel]
        σ2 = sum(our_vel' * our_vel) * blur_coef + 0.001

        their_pos = [
            sample_gauss(state[id_other_pos][1], σ2)
            sample_gauss(state[id_other_pos][2], σ2)
        ]

        our_pos = [
            state[id_own_pos][1]
            state[id_own_pos][2]
        ]

        alter(state,
            id_obs => [state[id_own_pos]; their_pos]
        )
    end

    state, dyn!
end

function constraint_cost(agent, state; λ=20)
    id_us = Symbol("$(agent)_pos")
    z  = λ*penalty(20 - state[id_us][1])
    z += λ*penalty(20 - state[id_us][2])
    z += λ*penalty(20 + state[id_us][1])
    z += λ*penalty(20 + state[id_us][2])
end

function reg_cost(agent, state; α=1)
    id_ctrl = Symbol("$(agent)_acc")
    α * sum(state[id_ctrl].^2)
end

function make_blurtag_cost(agent)
    function cost(state)
        d = dist2(state[:p1_pos], state[:p2_pos])
        d = (agent == :p2) ? -d : d
        d + constraint_cost(agent, state) #+ reg_cost(agent, state)
    end
end

function render_blurtag_game(g::SensingGame, plt; n_hist=8)

    agents = [:p1,   :p2]
    pos_colors = [:blue, :red]
    obs_colors = [:orange, :purple]

    hist = last(g.history, n_hist)

    for i in 1:2
        id_pos = Symbol("$(agents[i])_pos")
        id_obs = Symbol("$(agents[i])_obs")
        id_vel = Symbol("$(agents[i])_vel")
        # id_θ = Symbol("$(agents[i])_θ")

        state_x = [s[id_pos][1] for s in hist]
        state_y = [s[id_pos][2] for s in hist]
        obs_x =   [s[id_obs][3] for s in hist]
        obs_y =   [s[id_obs][4] for s in hist]

        plot!(state_x, state_y,
            color=pos_colors[i],
            alpha=0.1,
            label=""
        )

        plot!(state_x, state_y,
            seriestype=:scatter,
            color=pos_colors[i],
            alpha=0.3,
            label=""
        )

        plot!(obs_x, obs_y,
            seriestype=:scatter,
            color=obs_colors[i],
            alpha=0.5,
            label=""
        )

        θ = atan(hist[end][id_vel][2], hist[end][id_vel][1])
        heading_x = hist[end][id_pos][1] .+ [0; cos(θ); NaN]
        heading_y = hist[end][id_pos][2] .+ [0; sin(θ); NaN]
        
        plot!(heading_x, heading_y,
            color=pos_colors[i],
            alpha=0.4,
            label=""
        )
    end
end

function characterize_blurtag_stationary(agent, particles; t=100)
    id_us = Symbol("$(agent)_pos")
    states = mapreduce(vcat, particles) do p
        map(last(p.history, t)) do s
            s[id_us]
        end
    end
    svd(cov(states)).S
end

function test_blurtag_game(; blur_coef1, blur_coef2, logger)
    global_logger(ConsoleLogger(stdout, Logging.Debug))
    @debug "Making dynamics"
    state1, sdyn1 = make_acc_dynamics(:p1; control_scale=1, drag=0.5)
    state2, sdyn2 = make_acc_dynamics(:p2; control_scale=1, drag=0.5)

    # state1, sdyn1 = make_vel_dynamics(:p1)
    # state2, sdyn2 = make_vel_dynamics(:p2)

    obs1, odyn1 = make_blurtag_sensing(:p1, :p2; blur_coef=blur_coef1)
    obs2, odyn2 = make_blurtag_sensing(:p2, :p1; blur_coef=blur_coef2)

    ctrl1 = make_horizon_control(:p1, :p1_obs, :p1_acc)
    ctrl2 = make_horizon_control(:p2, :p2_obs, :p2_acc)

    initial_state = merge(state1, state2, obs1, obs2)

    components = [odyn1, odyn2, ctrl1, ctrl2, sdyn1, sdyn2]
    n_particles = 50
    n_samples = 1
    n_lookahead = 5
    n_render = 1
    resample_freq = -1
    
    prior = () -> alter(initial_state,
        :p1_pos => [2*randn();  2*randn()],
        :p2_pos => [2*randn(),  2*randn()]
    ) 

    cost_fn_1 = make_blurtag_cost(:p1)
    cost_fn_2 = make_blurtag_cost(:p2)

    @debug "Generating particles and policies"
    train_particles = [SensingGame(components, prior()) for _ in 1:n_particles]
    test_particles = [SensingGame(components, prior()) for _ in 1:n_particles]

    costs_1::Vector{Float64} = []
    costs_2::Vector{Float64} = []
    speeds_1::Vector{Float64} = []
    speeds_2::Vector{Float64} = []

    game_params = (; 
        policies = (;
            p1 = LinearPolicy(4, 2; t_max=n_lookahead),
            p2 = LinearPolicy(4, 2, t_max=n_lookahead)
        )
    )

    # return test_particles[1], game_params
    training_mode = true
    @debug "Beginning descent"
    for t in 1:10000

        t == 1 && @debug "Making scores"
        score1 = make_score(cost_fn_1, train_particles;    
            n_samples, n_lookahead, parallel=(t!=1))
        score2 = make_score(cost_fn_2, train_particles; 
            n_samples, n_lookahead, parallel=(t!=1))

        c1 = c2 = g1 = g2 = nothing

        try
            # If using wandb to log the python and julia garbage collectors can interfere
            #  (wandb uses PythonCall under the hood, which is not thread safe)
            PythonCall.GC.disable()
            if training_mode
                if t == 1 @debug "Taking gradients" end
                c1, g1 = Flux.withgradient(score1, game_params)
                c2, g2 = Flux.withgradient(score2, game_params)

                if t == 1 @debug "Applying gradients" end
                apply_gradient!(game_params.policies[:p1], g1[1].policies[:p1])
                apply_gradient!(game_params.policies[:p2], g2[1].policies[:p2])
            end
            # PythonCall.GC.enable()
        catch e
            if e isa InterruptException
                close(logger)
                throw(e)
            else
                # throw(e)
                @warn "Gradient failed with $e; attempting to continue..."
                continue
            end
        end

        # Take the actual step (we don't do the gradient on this,
        #   although technically we could I think?)
        t == 1 && @debug "Stepping particles"
        if (resample_freq > 0) && (t % resample_freq == 0)
            train_particles = [SensingGame(components, prior()) for _ in 1:n_particles]
            test_particles = [SensingGame(components, prior()) for _ in 1:n_particles]
        else
            map(train_particles) do prt::SensingGame
                update!(prt, step(prt, game_params, n=1))
            end

            map(test_particles) do prt::SensingGame
                s = step(prt, game_params, n=1)
                update!(prt, s)

                d = dist2(s[end][:p1_pos], s[end][:p2_pos])
                push!(costs_1, d)
                push!(costs_2, -d)

                s1 = sqrt(s[end][:p1_vel]' * s[end][:p1_vel])
                s2 = sqrt(s[end][:p2_vel]' * s[end][:p2_vel])
                push!(speeds_1, s1)
                push!(speeds_2, s2)

            end

            n = 100 * n_particles
            costs_1 = last(costs_1, n)
            costs_2 = last(costs_2, n)
            cc1 = sum(costs_1)/n
            cc2 = sum(costs_2)/n

            speeds_1 = last(speeds_1, n)
            speeds_2 = last(speeds_2, n)
            sp1 = sum(speeds_1)/n
            sp2 = sum(speeds_2)/n

            s = characterize_blurtag_stationary(:p2, test_particles)

            if ! isnothing(logger)
                Wandb.log(logger, Dict([
                    "t" => t,
                    "cc1" => cc1,
                    "cc2" => cc2,
                    "s1" => s[1],
                    "s2" => s[2],
                    "sp1" => sp1,
                    "sp2" => sp2
                ]))
            else
                @info "Step" t cc1 cc2 s[1] s[2]
            end
        end


        println(t)
        if t == 1 @debug "Plotting" end
        plt = plot(aspect_ratio=:equal, lims=(-22, 22))
        title!("$(training_mode ? "Training" : "Testing") planar blurtag: t=$t")
        for particle in test_particles[1:n_render]
            render_blurtag_game(particle, plt)
        end
        display(plt)

        t == 1 && @debug "Iterating..."
    end
    return game_params
end

function wandb_test_blurtag_game()
    for blur_coef1 in [2.0, 20.0, 200.0]
        for blur_coef2 in [0.0, 0.0]
            lg = WandbLogger(
                min_level=Debug,
                project = "SensingGames",
                name = "stationaries-$(now())-$blur_coef1-$blur_coef2",
                config = Dict("blur_coef1" => blur_coef1, "blur_coef2" => blur_coef2)
            )
            gp = test_blurtag_game(; blur_coef1, blur_coef2, logger=lg)
            close(lg)

            model_state_1 = Flux.state(gp.policies.p1)
            model_state_2 = Flux.state(gp.policies.p2)
            JLD2.jldsave("./models/p1-$(now()).jld2"; model_state_1)
            JLD2.jldsave("./models/p2-$(now()).jld2"; model_state_2)
        end
    end
end
