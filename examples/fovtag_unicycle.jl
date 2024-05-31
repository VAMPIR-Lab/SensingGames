
function make_fovtag_sensing(agent, other, targs)
    id_obs = Symbol("$(agent)_obs")
    id_own_pos = Symbol("$(agent)_pos")
    id_own_vel = Symbol("$(agent)_vel")
    id_own_θ = Symbol("$(agent)_θ")
    id_other_pos = Symbol("$(other)_pos")

    state = State(
        id_obs => 5
    )
    
    function dyn!(state, history, game_params)
        our_pos = state[id_own_pos]
        their_pos = state[id_other_pos]

        our_θ = state[id_own_θ]
        # our_vel = state[id_own_vel]

        # θ1 = atan(our_vel[2], our_vel[1])
        θ1 = our_θ[1]
        θ2 = atan(our_pos[2] - their_pos[2], our_pos[1] - their_pos[1])
        dθ = abs(angdiff(θ1, θ2))
        
        # σ2 = softif(0.8 - dθ, 0, 10)
        σ2 = 0

        their_pos = [
            sample_gauss(state[id_other_pos][1], σ2)[1]
            sample_gauss(state[id_other_pos][2], σ2)[1]
        ]

        alter(state,
            id_obs => [state[id_own_pos]; their_pos; our_θ]
        )
    end

    state, dyn!
end

function make_fovtag_cost(agent::Symbol, partner::Symbol; neg=false)
    id_us = Symbol("$(agent)_pos")
    id_ctrl = Symbol("$(agent)_u")
    id_them = Symbol("$(partner)_pos")

    function cost(state)
        d = dist2(state[id_us], state[id_them])
        r = 0*sum(state[id_ctrl].^2)

        c = softif(100 - sqrt(state[id_us]'state[id_us]), 0, 100000)
        neg ? (-d+r+c) : (d+r+c)
    end
end

function render_fovtag_game(g::SensingGame, plt)

    agents = [:p1,   :p2]
    colors = [:blue, :red]

    for i in 1:2
        id_pos = Symbol("$(agents[i])_pos")
        # id_θ = Symbol("$(agents[i])_θ")

        state_x = [s[id_pos][1] for s in g.history]
        state_y = [s[id_pos][2] for s in g.history]

        plot!(state_x, state_y,
            color=colors[i],
            alpha=0.1,
            label=""
        )

        plot!(state_x, state_y,
            seriestype=:scatter,
            color=colors[i],
            alpha=0.5,
            label=""
        )
    end
end

function test_fovtag_game()

    targs = [
        [],
        []
    ]

    state1, sdyn1 = make_unicycle_dynamics(:p1)
    state2, sdyn2 = make_unicycle_dynamics(:p2)

    obs1, odyn1 = make_fovtag_sensing(:p1, :p2, targs[1])
    obs2, odyn2 = make_fovtag_sensing(:p2, :p1, targs[2])

    ctrl1 = make_horizon_control(:p1, :p1_obs, :p1_u)
    ctrl2 = make_horizon_control(:p2, :p2_obs, :p2_u)

    initial_state = merge(state1, state2, obs1, obs2)

    game_params = (; 
        policies = Dict([
            :p1 => LinearPolicy(5, 2; t_max=8)
            :p2 => LinearPolicy(5, 2, t_max=8)
        ]),
        dθ=0.0
    )

    prior = () -> alter(initial_state,
        :p1_pos => [10*randn();  10*randn()],
        :p2_pos => [10*randn(),  10*randn()]
    ) 

    particles = []
    components = [odyn1, odyn2, ctrl1, ctrl2, sdyn1, sdyn2]
    n_particles = 10
    particles = [SensingGame(components, prior()) for _ in 1:n_particles]

    println("Created particles")


    cost_fn_1 = make_fovtag_cost(:p1, :p2)
    cost_fn_2 = make_fovtag_cost(:p1, :p2; neg=true)

    costs_1 = [0.0]
    costs_2 = [0.0]

    function score1(params)
        mapreduce(+, particles) do prt
            # ctrl1(prt.history[end], prt.history, params)[:p1_pos][1]
            state1 = step(prt, params)
            state2 = step(prt, params; extra_hist = [state1])
            state3 = step(prt, params; extra_hist = [state1; state2])
            update!(prt, state1)
            update!(prt, state2)
            update!(prt, state3)
            cost_fn_1(state3)
        end / length(particles)
    end

    function score2(params)
        mapreduce(+, particles) do prt
            # ctrl1(prt.history[end], prt.history, params)[:p1_pos][1]
            state1 = step(prt, params)
            state2 = step(prt, params; extra_hist = [state1])
            state3 = step(prt, params; extra_hist = [state1; state2])
            update!(prt, state1)
            update!(prt, state2)
            update!(prt, state3)
            cost_fn_2(state3)
        end / length(particles)
    end

    for t in 1:10000

        c1, grads1 = Flux.withgradient(score1, game_params)
        c2, grads2 = Flux.withgradient(score2, game_params)

        # map(particles) do prt
        #     state = step(prt, game_params) 
        #     update!(prt, state)
        #     cost_fn(state)
        # end

        sleep(0.05)
        plt = plot(aspect_ratio=:equal)
        for particle in particles[1:1]
            render_fovtag_game(particle, plt)
        end
        display(plt)

        n = min(100, length(costs_1))
        push!(costs_1, c1)
        push!(costs_2, c2)
        println("t=$(t)\tc1=$(sum(costs_1[end-n+1:end])/n)\tc2=$(sum(costs_2[end-n+1:end])/n)")

        grads = grads1[1]
        apply_gradient!(game_params.policies[:p1], grads.policies[:p1][])

        grads = grads2[1]
        apply_gradient!(game_params.policies[:p2], grads.policies[:p2][])
    end
end