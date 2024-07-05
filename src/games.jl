struct SensingGame <: Game
    prior_fn::Function
    dyn_fns::Vector{Function}
end

function rollout(g::SensingGame, game_params; n=5)
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


function make_hist_step(agent, ids, m, t_max)
    id_hist = Symbol(agent, "_hist")

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
        # t = dist[:t] .+ dt
        alter(dist, :t => t .+ dt)
    end
    State(:t => 1), clock_step
end

function make_cross_step(dyn_fn, ll_fn, alter_id, info_id, n)

    num_reps = ((n isa AbstractVector) ? (t) -> n[t] : (t) -> n) 

    infoset_id = 1

    function cross_step(state_dist, game_params)

        t = Int(state_dist[:t][1]) 
        if t == 1
            infoset_id = 1
        end

        r = num_reps(t)
        
        infosets = Zygote.ignore() do 
            unique(state_dist[info_id])
        end

        dists = map(infosets) do infoset
            rows = vec(state_dist[info_id] .== infoset)


            info_dist = StateDist(
                state_dist.z[rows, :],
                state_dist.w[rows],
                state_dist.ids,
                state_dist.map
            )
            m = length(info_dist)

            rep_dist = draw(info_dist; n=r)

            out_dist = dyn_fn(rep_dist, game_params)
            lls = ll_fn(info_dist, out_dist)
            lls = Float32.(lls .- log.(sum(exp.(lls), dims=2)))

            z_new = repeat(info_dist.z, outer=(r, 1))
            w_new = repeat(info_dist.w, outer=(r, 1)) 
            w_new = vec(w_new .+ reshape(lls, (:)))
            info_new = Float32.([infoset_id:(infoset_id+r-1)...])
            info_new = repeat(info_new, inner=(m, 1))
            
            res = StateDist(z_new, w_new, state_dist.ids, state_dist.map)

            res = alter(res,
                alter_id => repeat(out_dist[alter_id], inner=(m, 1)),
                info_id => info_new
            )
            infoset_id += r

            res
        end

        StateDist(
            vcat([dist.z for dist in dists]...),
            vcat([dist.w for dist in dists]...),
            dists[begin].ids,
            dists[begin].map
        )
    end

    State(info_id => 1), cross_step
end