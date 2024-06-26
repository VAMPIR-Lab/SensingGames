struct SensingGame <: Game
    prior_fn::Function
    dyn_fns::Vector{Function}
    cost_fn::Function

    history::Vector{StateDist}
    history_len::Int
end

function SensingGame(prior_fn::Function, dyn_fns::Vector{Function}, cost_fn::Function; history_len=100)
    SensingGame(prior_fn, dyn_fns, cost_fn, [], history_len)
end

function restart!(g::SensingGame)
    empty!(g.history)
end

function update!(g::SensingGame, states)
    for s in states
        roll!(g.history, s, g.history_len)
    end
    states
end

function step(g::SensingGame, game_params; n=1, lik_check=true)
    

    past_hist = isempty(g.history) ? [g.prior_fn()] : g.history
    all_res = [(past_hist, past_hist[end])]

    for t in 1:n
        for dyn_fn in g.dyn_fns
            new_all_res = mapreduce(vcat, all_res) do (hist, current_dist)
                next_dists = dyn_fn(current_dist, hist, game_params)

                # Returning just a single StateDist is fine
                if ! (typeof(next_dists) <: AbstractArray{StateDist})
                    next_dists = [next_dists]
                end

                [(hist, next_dist) for next_dist in next_dists]
            end
            all_res = new_all_res # Zygote bug with variables with the same name
        end

        # We update the history only once every time step
        #   (not every dynamics substep)
        new_all_res = map(all_res) do (hist, current_dist)
            ([hist; current_dist], current_dist)
        end
        all_res = new_all_res
    end

    # Check: Over all possible histories,
    #  the likelihood should be = 1
    if lik_check
        lik = 0
        for (hist, _) in all_res
            dist = hist[end]
            lik += sum(exp.(dist.w))
        end
        if ! (lik ≈ Float32(1.0))
            println("Warning: Loss of probability density: $lik != 1.0")
            sleep(10)
        end
    end

    [hist for (hist, _) in all_res]
end

# function make_cross_step(dyn_fns, outs, rnk_fns; nd=nothing)

#     function cross_dyn(state_dist::StateDist, history, game_params)
#         t = length(history)
#         m = length(state_dist)

#         # By default we take two draws for each fn in the cross
#         n_draws_per_fn = isnothing(nd) ? fill(2, length(dyn_fns)) : nd[t]

#         reps_per_fn = map(n_draws_per_fn) do n_draws
#             draw(state_dist; n=n_draws)
#         end

#         outs = map(Iterators.product(reps_per_fn...)) do reps
#             reduce(enumerate(reps), init=state_dist) do (dist, rep)
#                 res = 
#             end
#             # map(enumerate(reps)) do (i, rep)
#             #     dyn_fns[i](rep)
#             # end
#         end

#         # 

#         reps_1 = draw(state_dist; n=nd[t])
#         reps_2 = draw(state_dist; n=1)
#         # TODO DRY this

#         obs1 = map(reps_1) do rep
#             μ_1_rep = rep[:p1_pos]
#             σ_1_rep = 0.1 + 4*(targ_1 - μ_1_rep[2])^2
#             Zygote.ignore() do
#                 repeat(sample_gauss.(μ_1_rep, σ_1_rep)', m)
#             end
#         end

#         obs2 = map(reps_2) do rep
#             μ_2_rep = rep[:p2_pos]
#             σ_2_rep = 0.1 + 4*(targ_2 - μ_2_rep[2])^2
#             Zygote.ignore() do
#                 repeat(sample_gauss.(μ_2_rep, σ_2_rep)', m)
#             end
#         end
        
#         obs = Iterators.product(obs1, obs2)

#         lls = map(obs) do (ob_1, ob_2)
#             μ_1_true = state_dist[:p1_pos]
#             μ_2_true = state_dist[:p2_pos]
#             σ_1_true = 0.1 .+ 4 * (targ_1 .- μ_1_true[:, 2]).^2
#             σ_2_true = 0.1 .+ 4 * (targ_2 .- μ_2_true[:, 2]).^2

#             ll_1 = sum(SensingGames.gauss_logpdf.(ob_1, μ_1_true, σ_1_true), dims=2)
#             ll_2 = sum(SensingGames.gauss_logpdf.(ob_2, μ_2_true, σ_2_true), dims=2)
#             ll = ll_1 #.+ ll_2

#             # Prevent ll from being too extreme
#             ll = softclamp.(ll, -50, 50)
#             vec(ll)
#         end


#         ll_norm = log.(sum(ll -> exp.(ll), lls))

#         # println("===")
#         # @show exp.(lls[1] .- ll_norm)[1:end]
#         # @show exp.(lls[2] .- ll_norm)[1:end]

#         res = map(zip(obs, lls)) do (((ob_1, ob_2), ll))
#             new_dist = alter(state_dist,
#                 :p1_obs => ob_1,
#                 :p2_obs => ob_2
#             )
#             new_dist = reweight(new_dist, 
#                 (ll - ll_norm)
#             )
#             new_dist
#         end

#         res = vec(res)
        
#         # This enables / disables universally consistent observations
#         # 
#         # On: (o1a, u1a) -> (o2a, u2a)
#         #                   (o2b, u2a)
#         #     (o1b, u1a) -> (o2a, u2a)  (sampling from entire state dist at t=2)
#         #                   (o2b, u2a)
#         #Off: (o1a, u1a) -> (o2a, u2a)
#         #                   (o2b, u2a)
#         #     (o1b, u1a) -> (o2c, u2b)  (sampling from state dist GIVEN first obs)
#         #                   (o2d, u2b) 

#         # I think what's actually correct is NEITHER of these.
#         # In this scenario we want:
#         #     (o1a, u1a) -> (o2a, u2a)
#         #                   (o2b, u2a)
#         #     (o1b, u1a) -> (o2c, u2a)
#         #                   (o2d, u2a) 
#         # Universally consistent is conservative (and also happens
#         # to be a lot faster implementation wise)

#         z = mapreduce(dist -> dist.z, vcat, res)
#         w = mapreduce(dist -> dist.w, vcat, res)
#         res = StateDist(z, w, state_dist.ids, state_dist.map)


#         res
#     end
# end

function step!(g::SensingGame, game_params; n=1)
    update!(g, step(g, game_params; n))
end

function clone(g::SensingGame)
    SensingGame(g.prior_fn, g.dyn_fns, g.cost_fn; history_len=g.history_len)
end