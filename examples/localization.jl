# using SensingGames
using Distributions
using Flux

function dist2(a, b)
    (a-b)'*(a-b)
end

function sensor_dynamics(state; loc)
    dist = MvNormal(state[1:2], [5*(loc - state[2]); 0])
    ob = rand(dist)
    return ob, pdf(dist, ob)
end

function state_dynamics(state, action; dt=1.0)
    ω = rand(MvNormal([0; 0], 0.0))
    # [
    #     state[1:2] + dt.*state[3:4] + .5*dt^2*action
    #     state[3:4] + dt.*action + ω
    # ], 1.0
    v = (action + ω)
    [
        state[1:2] + dt.*v
        v
    ], 1.0
end

function prior(;loc, σ=1.0)
    dist = MvNormal(loc, [σ; 0])
    x = rand(dist)
    [x; zeros(2)], pdf(dist, x)
end

function cost(hist; targ)
    # hist.states[1]
    state = eachcol(hist.states)[end]
    sqrt(dist2(state[1:2], targ)) #* hist.prob
end


function render_localization_game(hists)

    plt = plot(
        lims=(-3, 3)
    )

    for hist in hists
        state_hist = hist.states
        ob_hist = hist.obs

        state_x = [state[1] for state in eachcol(state_hist)]
        state_y = [state[2] for state in eachcol(state_hist)]

        ob_x = [ob[1] for ob in eachcol(ob_hist)]
        ob_y = [ob[2] for ob in eachcol(ob_hist)]

        plot!(state_x, state_y,
            color=:black,
            alpha=0.3,
            label=""
        )

        plot!(state_x, state_y,
            seriestype=:scatter,
            color=:black,
            alpha=0.3,
            label=""
        )

        # plot!(plt, ob_x, ob_y,
        #     seriestype=:scatter,
        #     markersize=3,
        #     color=:steelblue,
        # )

        # plot!(plt, ob_x, ob_y,
        #     alpha=0.3,
        #     color=:steelblue
        # )

        # loc = [0; 0.0]
        # dists = [sqrt(dist2(state[1:2], loc)) for state in eachcol(state_hist)]

        # plot!(plt, ob_x, ob_y,
        #     seriestype=:scatter,
        #     markersize=dists
        # )

        # for (state, ob) in zip(state_hist, ob_hist)
            
        # end
    end

    display(plt)

end

function test_localization_game() 

    sensor_location = 0.0
    target_location = [0.0; -2.0]
    prior_location = [0.0; -0.5]

    prior_noise = 1.0

    T = 8
    D = 8


    game = SensingGames.SensingGame(
        () -> prior(;loc=prior_location, σ=prior_noise), 
        state_dynamics,
        x -> sensor_dynamics(x; loc=sensor_location),
        x -> cost(x; targ=target_location),
        zeros(2),
        t_max=T,
        max_obs_used=D
    )

    # policy = SensingGames.LinearPolicy(2, 2, T)

    policy = SensingGames.EmbedPolicy(2, 8, T, [
        Flux.Dense(8 => 2)
    ])
    costs = []

    for i in 1:2000
        push!(costs, gradient_step!(policy, game))
        n = min(length(costs), 100)
        if i % 10 == 0
            hists = [rollout(game, policy) for i in 1:40]
            render_localization_game(hists)
            println("Average cost: $(sum(costs[end-n+1:end])/n)")
            # println("Rendered cost: $(cost(hist; targ=target_location))")
        end
    end
end