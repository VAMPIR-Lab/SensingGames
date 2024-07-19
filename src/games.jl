struct SensingGame <: Game
    prior_fn::Function
    dyn_fns::Vector{Function}
end

function rollout(g::SensingGame, game_params; n)
    state = g.prior_fn()
    hist = [state]

    for _ in 1:n
        for dyn_fn in g.dyn_fns
            state = dyn_fn(state, game_params)
        end
        hist = [hist; state]
    end
    
    hist
end


# In general a game is a sequence of steps.
#   Each step function maps from the current state distribution
#   (which is being rolled out) and the current game params
#   (which are subject to optimization) to the next state
#   distribution.
#   For the most part state distributions and single states
#   behave identically for this purpose.
# Each step function is a closure: we define e.g. `make_step`
#   to give it all its parameters, and that spits out whatever
#   piece of state that it needs, and a `step` function
#   that actually performs the step.

# This works nicely for a few reasons:
# * Steps can call `alter(::State, pairs...)` to move to the next
#   state. This doesn't do any in-place array modification, so it's
#   Zygote-safe.
# * States are symbol indexed. We can merge all the state fragments
#   into one big state and the step functions will all still work
#   regardless of the organization of the state in memory or the 
#   other state functions that might be present.

# Beware of the danger of closures / type instability when
#   writing steps. The performance impact is small but insidious.


function make_hist_step(id_hist, ids, m, t_max)

    function hist_step(dist, params)
        hist = [dist[ids] dist[id_hist]][:, begin:(m*t_max)]
        
        alter(dist,
            id_hist => hist
        )
    end

    s = State(id_hist => m*t_max)
    s = alter(s, 
        id_hist => s[id_hist] .+ 0
    )
    
    s, hist_step
end

function make_clock_step(dt)
    function clock_step(dist, params)
        t = Zygote.ignore() do 
            dist[:t] 
        end
        alter(dist, :t => t .+ dt)
    end
    State(:t => 1), clock_step
end

function make_cross_step(dyn_fn, ll_fn, alter_id, info_id, n; dedupe=false)

    num_reps = ((n isa AbstractVector) ? (t) -> n[t] : (t) -> n) 

    infoset_num = 1
    function cross_step(state_dist, game_params)

        t = Int(state_dist[:t][1]) 
        if t == 1
            infoset_num = 1
        end
        r = num_reps(t)
        
        infosets = Zygote.ignore() do 
            unique(state_dist[info_id])
        end

        dists = map(infosets) do infoset
            # quantile = rand(SensingGames._game_rng, r, 2) #[randu.(SensingGames._game_rng, 0.1, 0.9) randu.(SensingGames._game_rng, 0.1, 0.9)]
            quantile = -1
            rows = vec(state_dist[info_id] .== infoset)

            info_dist = StateDist(
                state_dist.z[rows, :],
                state_dist.w[rows],
                state_dist.ids,
                state_dist.map
            )

            in_weight = sum(exp.(info_dist.w))

            m = length(info_dist)


            # It's possible to get duplicate outputs; dedupe those if desired
            rep_dist = if dedupe
                # Zygote.ignore() do
                    out_dist = dyn_fn(info_dist, game_params, quantile)
                    unique_idxs = Zygote.ignore() do
                        unique(i -> out_dist[i][alter_id], 1:length(out_dist))
                    end

                    deduped_out_dist = StateDist(
                        out_dist.z[unique_idxs, :],
                        out_dist.w[unique_idxs],
                        out_dist.ids,
                        out_dist.map
                    )
                    draw(deduped_out_dist, n=r)
                # end
            else
                pre_dist = draw(info_dist, n=r)
                dyn_fn(pre_dist, game_params, quantile)
            end

            lls = ll_fn(info_dist, rep_dist)

            # Enforce ∑[o] P(o | S) = 1 for all S
            lls = (lls .- log.(sum(exp.(lls), dims=2)))

            z_new = repeat(info_dist.z, outer=(r, 1))
            w_new = repeat(info_dist.w, outer=(r, 1)) 
            w_new = vec(w_new .+ reshape(lls, (:)))
            info_new = ([infoset_num:(infoset_num+r-1)...])
            info_new = repeat(info_new, inner=(m, 1))

            split_dist = alter(StateDist(z_new, w_new, state_dist.ids, state_dist.map),
                alter_id => repeat(rep_dist[alter_id], inner=(m, 1)),
                info_id => Float64.(info_new)
            )
            infoset_num += r

            out_weight = sum(exp.(split_dist.w))
            
            Zygote.ignore() do 
                if ! (out_weight ≈ in_weight)
                    @warn "Change of branch weight: $in_weight != $out_weight in infoset $info_id=$infoset"
                end
            end

            split_dist
        end

        res = StateDist(
            vcat([dist.z for dist in dists]...),
            vcat([dist.w for dist in dists]...),
            dists[begin].ids,
            dists[begin].map
        )
        res
    end

    State(info_id => 1), cross_step
end


# Prune unlikely / counterfactual branches from further consideration
# function make_prune_step(n)
#     function prune_step(state_dist, game_params)
#         idxs = Zygote.ignore() do 
#             sortperm(-state_dist.w)
#         end

#         StateDist(
#             state_dist.z[idxs],
#             state_dist.w[idxs],
#             state_dist.ids,
#             state_dist.map
#         )
#     end
# endZ