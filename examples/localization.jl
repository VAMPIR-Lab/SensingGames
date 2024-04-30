# using SensingGames
using Distributions
using Flux

function dist2(a, b)
    (a-b)'*(a-b)
end

function sample_gauss(mean, var)
    # Bowling approximation of inverse CDF
    v = log(1/rand() - 1) * sqrt(var) / -1.702 + mean
    ll = (-0.5*(v - mean)^2 / var) - log(sqrt(2 * pi * var))
    return v,ll
end

function sensor_dynamics(state; loc)
    σ2 = 4*(loc - state[2])^2
    # sample_gauss(state[1], σ2)[1]
    x1, ll1 = sample_gauss(state[1], σ2)
    x2, ll2 = sample_gauss(state[2], σ2)
    [x1; x2], ll1 + ll2
    # state[1:2], 0
end

# function sensor_dynamics(state; loc)
#     dist = MvNormal(state[1:2], (loc - state[2])^2 + 0.1)
#     # ob = state[1:2]
#     ob = rand(dist)
#     # ob = state[1:2] + randn(2)
#     ll = logpdf(dist, ob)
#     # ll = logpdf(dist, state[1:2])
#     return ob, ll
# end

function state_dynamics(state, action; dt=1.0)
    ω = rand(MvNormal([0; 0], 0.0001))
    [
        state[1:2] + dt.*state[3:4] + .5*dt^2*action
        state[3:4] + dt.*action + ω
    ], 0.0
    # v = (action + ω)
    # [
    #     state[1:2] + dt.*v
    #     v
    # ], 0.0
end

function prior(;loc, σ=1.0)
    dist = Normal(loc[1], σ)
    x = rand(dist)
    [x; loc[2]; zeros(2)], logpdf(dist, x)
end

function cost(hist; targ)
    state = eachcol(hist.states)[end]
    # -hist.log_lik
    sqrt(dist2(state[1:2], targ))
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

    policy = SensingGames.LinearPolicy(2, 2, T)

    # policy = SensingGames.EmbedPolicy(2, 4, T, [
    #     Flux.Dense(4 => 2),
    # ])

    costs = []
    for i in 1:2000
        push!(costs, gradient_step!(policy, game))
        n = min(length(costs), 100)
        println("$(i)\t$(sum(costs[end-n+1:end])/n)")
        if i % 10 == 0
            hists = mapreduce(vcat, 1:10) do i
                [rollout(game, policy)]
            end
            render_localization_game(hists)
            # println("Rendered cost: $(cost(hist; targ=target_location))")
        end
    end
end