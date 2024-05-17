
function make_sensing(agent::Symbol, targ; dt=1.0)
    id_obs = Symbol("$(agent)_obs")
    id_pos = Symbol("$(agent)_pos")

    state = State(
        id_obs => 2
    )

    function dyn!(state, game_params)
        pos = state[id_pos]
        σ2 = 4*(targ - pos[2])^2
        x1, _ = sample_gauss(pos[1], σ2)
        x2, _ = sample_gauss(pos[2], σ2)

        alter(state,
            id_obs => [x1; x2]
        )
    end
    
    state, dyn!
end

function make_cost(agent::Symbol, targ)
    id_pos = Symbol("$(agent)_pos")

    function cost(state)
        dist2(state[id_pos], targ)
    end
end


function render_localization_game(hists)
    plt = plot(
        lims=(-3, 3)
    )

    for states in hists

        state_x1 = [s[:p1_pos][1] for s in states]
        state_y1 = [s[:p1_pos][2] for s in states]
        state_x2 = [s[:p2_pos][1] for s in states]
        state_y2 = [s[:p2_pos][2] for s in states]

        plot!(state_x1, state_y1,
            color=:blue,
            alpha=0.3,
            label=""
        )

        plot!(state_x1, state_y1,
            seriestype=:scatter,
            color=:blue,
            alpha=0.3,
            label=""
        )

        plot!(state_x2, state_y2,
            color=:red,
            alpha=0.3,
            label=""
        )

        plot!(state_x2, state_y2,
            seriestype=:scatter,
            color=:red,
            alpha=0.3,
            label=""
        )
    end

    display(plt)

end

function test_localization_game() 

    state1, sdyn1 = make_vel_dynamics(:p1)
    state2, sdyn2 = make_vel_dynamics(:p2)

    obs1, odyn1 = make_sensing(:p1, 0.0)
    obs2, odyn2 = make_sensing(:p2, 0.0)

    ctrl1 = make_horizon_control(:p1, :p1_obs, :p1_vel)
    ctrl2 = make_horizon_control(:p2, :p2_obs, :p2_vel)

    initial_state = merge(state1, state2, obs1, obs2)
    initial_state = alter(initial_state,
        :p1_pos => [randn(); -0.3],
        :p2_pos => [randn(),  0.3]
    )

    game_params = (; 
        policies = Dict([
            :p1 => LinearPolicy(2, 2; t_max=8)
            :p2 => LinearPolicy(2, 2; t_max=8)
        ])
    )

    prior = () -> alter(initial_state,
        :p1_pos => [randn(); -0.3],
        :p2_pos => [randn(),  0.3]
    ) 

    game = SensingGame(
        [sdyn1, sdyn2, odyn1, odyn2, ctrl1, ctrl2],
        prior
    )

    cost1 = make_cost(:p1, [0.0; -2.0])
    cost2 = make_cost(:p2, [0.0; 2.0])

    run_game = params -> rollout(game, params, T=5)
    score1 = params -> mapreduce(_ -> cost1(run_game(params)[end]), +, 1:10)
    score2 = params -> mapreduce(_ -> cost2(run_game(params)[end]), +, 1:10)

    for t in 1:1000
        println("t=$(t)")

        if t % 10 == 0
            hists = [run_game(game_params) for _ in 1:10]
            render_localization_game(hists)
        end

        grads1 = Flux.gradient(score1, game_params)[1]
        grads2 = Flux.gradient(score2, game_params)[1]

        apply_gradient!(game_params.policies[:p1], grads1.policies[:p1][])
        apply_gradient!(game_params.policies[:p2], grads2.policies[:p2][])
    end
end

