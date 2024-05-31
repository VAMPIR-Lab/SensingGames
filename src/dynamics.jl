# Should all output a (state, step) pair.

function make_vel_dynamics(agent::Symbol; dt=1.0, dims=2, control_scale=1.0)
    id_pos = Symbol("$(agent)_pos")
    id_vel = Symbol("$(agent)_vel")

    state = State(
        id_pos => dims,
        id_vel => dims
    )

    function dyn!(state::State, history::Vector{State}, game_params)::State
        pos = state[id_pos] + dt*state[id_vel]*control_scale
        alter(state,
            id_pos => softclamp.(pos, -11 + rand(), 11 + rand())
        )
    end
    
    state, dyn!
end

function make_acc_dynamics(agent::Symbol; dt=1.0, dims=2, drag=0.4, control_scale=1.0)
    id_pos = Symbol("$(agent)_pos")
    id_vel = Symbol("$(agent)_vel")
    id_acc = Symbol("$(agent)_acc")

    state = State(
        id_pos => dims,
        id_vel => dims,
        id_acc => dims
    )

    function dyn!(state::State, history::Vector{State}, game_params)::State
        acc = dt*state[id_acc] * control_scale
        vel = dt*state[id_vel]*(1-drag) + acc + 0.1*randn(2)
        pos = state[id_pos] + vel + 0.5*acc.^2

        alter(state, 
            id_pos => softclamp.(pos, -21 + rand(), 21 + rand()),
            id_vel => vel
        )
    end
    
    state, dyn!
end

function make_unicycle_dynamics(agent::Symbol; dt=1.0)
    id_pos = Symbol("$(agent)_pos")
    id_θ = Symbol("$(agent)_θ")
    id_u = Symbol("$(agent)_u")

    state = State(
        id_pos => 2,
        id_θ => 1,
        id_u => 2
    )

    function dyn!(state::State, history::Vector{State}, game_params)::State
        v = 0.1 
        # θ = game_params.dθ
        # dθ = 0
        # dθ = game_params.dθ
        v  = 1 + (1 + tanh.(state[id_u][1]))*1
        dθ = 0.5*tanh.(state[id_u][2])

        θ = state[id_θ][1] + dt*dθ
        if θ > π
            θ -= 2π
        elseif θ < -π
            θ += 2π
        end

        # dθ = game_params.dθ
        vel = v .* [cos(θ); sin(θ)]

        alter(state, 
            id_pos => state[id_pos] .+ dt*vel,
            id_θ => [θ]
        )
    end
    
    state, dyn!
end