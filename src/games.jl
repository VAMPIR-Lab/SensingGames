using LinearAlgebra

struct SensingGame <: Game
    prior_fn::Function
    dyn_fns::Vector{Function}
end

function rollout(g::SensingGame, game_params; n)
    state = g.prior_fn()
    hist = [state]


    for t in 1:n
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


function make_hist_step(id_hist, id_store, t_max)
    function hist_step(dist, params)
        m = size(dist[id_store])[2]
        hist = [dist[id_store] dist[id_hist]][:, begin:(m*t_max)]
        
        alter(dist,
            id_hist => hist
        )
    end
end

function make_clock_step(dt)
    function clock_step(dist, params)
        t = Zygote.ignore() do 
            dist[:t] 
        end
        alter(dist, :t => t .+ dt)
    end
end

function make_cross_step(dyn_fn, ll_fn, n, alter_ids)

    num_counterfactuals = ((n isa AbstractVector) ? (t) -> n[t] : (t) -> n) 
    function cross_step(state_dist, game_params)

        t = Int(state_dist[:t][1])
        r = num_counterfactuals(t)
        N = length(state_dist)
        
        quantile = rand(SensingGames._game_rng, N, 2) 

        new_state_dist = dyn_fn(state_dist, game_params, quantile)
        if r <= 0
            return new_state_dist
        end

        lls_n = ll_fn(state_dist, new_state_dist)
        lls_n_norm = sum(exp.(lls_n), dims=2)    

        # confusion_matrix = (lls_n .- log.(lls_n_norm) )
        # confusion_matrix = confusion_matrix .+ info_dist.w
        # confusion_matrix = exp.(confusion_matrix)
        # confusion_matrix = (.! I(N)) .* confusion_matrix


        confusion_entries = Zygote.ignore() do
            rand([Iterators.product(1:N, 1:N)...], r)
        end
        ground_entries = [(i, i) for i in 1:N]
        all_entries = [ground_entries; confusion_entries]


        confusion_mask_buffer = Zygote.bufferfrom(fill(false, (N, N)))
        for entry in confusion_entries
            confusion_mask_buffer[entry...] = true
        end
        confusion_mask = copy(confusion_mask_buffer)
        ground_mask = I(N)
        all_mask = confusion_mask .|| ground_mask
        

        liks = exp.(lls_n)
        new_liks = all_mask .* (liks)
        liks_norm = sum(new_liks, dims=2) 
        new_liks_normed = new_liks ./ liks_norm
        new_liks_marg = new_liks_normed .* exp.(state_dist.w)
        new_liks_masked = all_mask .* (new_liks_marg)

        new_w = map(all_entries) do entry
            log(new_liks_masked[entry...])
        end

        ground_rows = [a for (a, b) in all_entries]
        confusion_rows = [b for (a, b) in all_entries]

        dist = StateDist(
            state_dist.z[ground_rows, :],
            new_w,
            state_dist.ids,
            state_dist.map
        )

        alterations = [id => new_state_dist[id][confusion_rows, :] for id in alter_ids]

        alter(dist, alterations...)
    end
end


# Prune unlikely / counterfactual branches from further consideration
# function make_prune_step(n)
#     function prune_step(state_dist, game_params)
#         idxs = Zygote.ignore() do 
            # @show exp.(lls)
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