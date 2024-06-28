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
            println("Warning: Change of probability density: $lik != 1.0")
            sleep(10)
        end
    end

    [hist for (hist, _) in all_res]
end

function make_cross_step(dyn_fn, ll_fn, id, n)

    num_reps = ((n isa AbstractVector) ? (t) -> n[t] : (t) -> n) 

    function cross_dyn(state_dist, history, game_params)

        r = num_reps(length(history))
        m = length(state_dist)

        rep_dist = draw(state_dist; n=r)
        out_dist = dyn_fn(rep_dist, history, game_params)
        lls = ll_fn(state_dist, out_dist) #.+ 0.01
        # lls = softclamp.(lls, -20, 20)
        lls = Float32.(lls .- log.(sum(exp.(lls), dims=2)))

        z_new = repeat(state_dist.z, outer=(r, 1))
        w_new = repeat(state_dist.w, outer=(r, 1)) 
        w_new = vec(w_new .+ reshape(lls, (:)))
        
        res = StateDist(z_new, w_new, state_dist.ids, state_dist.map)
        alter(res,
            id => repeat(out_dist[id], inner=(m, 1))
        )
    end
end


function step!(g::SensingGame, game_params; n=1)
    update!(g, step(g, game_params; n))
end

function clone(g::SensingGame)
    SensingGame(g.prior_fn, g.dyn_fns, g.cost_fn; history_len=g.history_len)
end