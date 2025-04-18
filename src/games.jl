using LinearAlgebra

struct GameComponent
    rollout_fn::Function
    lik_fn::Function
    output_ids::Vector{Symbol}
end

struct ContinuousGame
    components::Vector{GameComponent}
end


# If no likelihood function is provided,
#   just assume effectively uniform
function GameComponent(rollout_fn, output_ids)
    GameComponent(
        rollout_fn,
        (state_in, state_out, params) -> zeros(length(state_in)),
        output_ids
    )
end



function step(g::ContinuousGame, initial_state, game_params; n=1)
    state = initial_state
    hist = [state]


    for t in 1:n
        for component in g.components
            # Zygote.ignore(() -> println(String(Symbol(component))))
            state = component.rollout_fn(state, game_params)
            if NaN ∈ state.z
                throw(ArgumentError("Transitioned to a NaN state"))
            end
        end
        hist = [hist; state]
    end
    
    (n == 1) ? hist[end] : hist
end

function step(g::ContinuousGame, initial_dist::StateDist, ground_state::State, game_params; normalize=true)

    state_dist = initial_dist
    w = state_dist.w

    # Assume all states are going to the same ground
    ground_dist = StateDist(ground_state, length(state_dist))
    ground_dist_single = StateDist([ground_state])

    for component in g.components
        ids = component.output_ids
        matches = [id ∈ ground_state.ids for id in ids]

        if all(matches)
            lik = component.lik_fn(state_dist, ground_dist_single, game_params)
            w = (w .+ lik)[:, 1, 1]
            
            state_dist = alter(state_dist, 
                (ids .=> [ground_dist[id] for id in ids])...
            )
        else
            state_dist = component.rollout_fn(state_dist, game_params)
        end
    end
    
    if(normalize)
        w = log.(exp.(w) ./ sum(exp.(w)))
    end

    reweight(state_dist, w)
end
