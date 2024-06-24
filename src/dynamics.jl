# These define dynamics that operate on States.
#   Each make_<?>_dynamics function outputs a zero State with any
#   state variables the dynamics define, and a function that rolls 
#   the dynamics a single step. 

# Planar velocity integrator
function make_vel_dynamics(agent::Symbol; dt=1.0, dims=2, control_scale=1.0)
    id_pos = Symbol("$(agent)_pos")
    id_vel = Symbol("$(agent)_vel")

    state = State(
        id_pos => dims,
        id_vel => dims
    )

    function dyn!(state::StateDist, history, game_params)
        pos = state[id_pos] + dt*state[id_vel]*control_scale
        alter(state,
            id_pos => pos
        )
    end
    
    state, dyn!
end

# Planar acceleration integrator
function make_acc_dynamics(agent::Symbol; dt=1.0, dims=2, drag=0.4, control_scale=1.0)
    id_pos = Symbol("$(agent)_pos")
    id_vel = Symbol("$(agent)_vel")
    id_acc = Symbol("$(agent)_acc")

    state = State(
        id_pos => dims,
        id_vel => dims,
        id_acc => dims
    )

    function dyn!(state::StateDist, history, game_params)
        acc = dt*state[id_acc] * control_scale
        vel = dt*state[id_vel]*(1-drag) + acc + 0.1*randn(2)
        pos = state[id_pos] + vel + 0.5*acc.^2

        alter(state, 
            id_pos => pos,
            id_vel => vel
        )
    end
    
    state, dyn!
end

# Unicycle dynamics
function make_unicycle_dynamics(agent::Symbol; dt=1.0)
    id_pos = Symbol("$(agent)_pos")
    id_θ = Symbol("$(agent)_θ")
    id_u = Symbol("$(agent)_u")

    state = State(
        id_pos => 2,
        id_θ => 1,
        id_u => 2
    )

    function dyn!(state::StateDist, history, game_params)
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

# Bound dynamics: Gently bounces players off a given bound
#   This prevents players from accidentally overshooting the
#   bound (porentially receiving a very high penalty) - should be
#   set slightly outside of the actual constraint
function make_bound_dynamics(id, lower, upper; ω=1.0)

    function dyn!(state::StateDist, history, gp)
        alter(state,
            id => Float32.(softclamp.(state[id], lower + ω*rand(), upper + ω*rand()))
        )
    end

    nothing, dyn!
end