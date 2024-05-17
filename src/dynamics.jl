# Should all output a (state, step) pair.

function make_vel_dynamics(agent::Symbol; dt=1.0, dims=2)
    id_pos = Symbol("$(agent)_pos")
    id_vel = Symbol("$(agent)_vel")

    state = State(
        id_pos => dims,
        id_vel => dims
    )

    function dyn!(state, game_params)
        alter(state,
            id_pos => state[id_pos] + dt*state[id_vel]
        )
    end
    
    state, dyn!
end

function make_acc_dynamics(agent::Symbol; dt=1.0, dims=2)
    id_pos = Symbol("$(agent)_pos")
    id_vel = Symbol("$(agent)_vel")
    id_acc = Symbol("$(agent)_acc")

    state = State(
        id_pos => dims,
        id_vel => dims,
        id_acc => dims
    )

    function dyn!(state, game_params)
        alter(state, 
            id_pos => state[id_pos] + dt*state[id_vel] + dt/2*state[id_acc].^2,
            id_vel => state[id_vel] + dt*state[id_acc]
        )
    end
    
    state, dyn!
end