# These define dynamics that operate on States.
#   Each make_<?>_dynamics function outputs a zero State with any
#   state variables the dynamics define, and a function that rolls 
#   the dynamics a single step. 

# Planar velocity integrator
function make_vel_dynamics_step(agent::Symbol; dt=1.0, control_scale=1.0)
    id_pos = Symbol("$(agent)_pos")
    id_vel = Symbol("$(agent)_vel")
    id_θ = Symbol("$(agent)_θ")


    function dynamics(state::StateDist, game_params)
        
        vel = state[id_vel]
        # println(1111111111111)
        # println(vel[1, :])
        orig_pos = state[id_pos]
        # println(orig_pos[1, :])
        # println(dt)
        # println(control_scale)

        pos = orig_pos + dt*vel * control_scale
        # println(pos[1, :])
        # println(vel[1,2])
        # println(vel[1,1])
        # println(atan(vel[1, 2], vel[1, 1]))

        # telepos = 60 * rand(size(pos)...) .- 30
        # pos = pos .+ round.(rand(size(pos)...) .- 0.48) .* telepos

        alter(state,
            id_pos => pos,
            id_θ => atan.(vel[:, 2], vel[:, 1])[:, :]
        )
    end

    GameComponent(dynamics, [id_pos, id_θ])
end

# Planar acceleration integrator
function make_acc_dynamics_step(agent::Symbol; dt=1.0, control_scale=1.0, max_vel=2.0)
    id_pos = Symbol("$(agent)_pos")
    id_vel = Symbol("$(agent)_vel")
    id_acc = Symbol("$(agent)_acc")
    id_θ = Symbol("$(agent)_θ")

    function dynamics(state::StateDist, game_params)
        acc = state[id_acc] .* control_scale
        vel = state[id_vel] .+ acc*dt
        vel = softclamp.(vel, -max_vel, max_vel)
        pos = state[id_pos] .+ state[id_vel] .+ 0.5.*acc*dt.^2

        alter(state, 
            id_pos => pos,
            id_vel => vel,
            id_θ => atan.(vel[:, 2], vel[:, 1])[:, :]
        )
    end

    GameComponent(dynamics, [id_pos, id_vel, id_θ])
end

# Unicycle dynamics
function make_vel_unicycle_dynamics_step(agent::Symbol; dt=1.0, control_scale=1.0, angular_control_scale=1.0)
    id_pos = Symbol("$(agent)_pos")
    id_θ = Symbol("$(agent)_θ")
    id_vω = Symbol("$(agent)_vω")

    function dynamics(state::StateDist, game_params)
        v = (tanh.(state[id_vω][:, 1]) .+ 1)./2 .+ 1
        ω = 0.1*state[id_vω][:, 2]
        θ = state[id_θ]


        θ_new = ω[:, :]
        vel = v .* [cos.(θ_new) sin.(θ_new)]

        pos = state[id_pos] + dt*vel*control_scale
        alter(state,
            id_pos => pos,
            id_θ => θ_new
        )
    end

    GameComponent(dynamics, [id_pos, id_θ])
end

# Bound: Clamp part of the state into a certain region
#   This prevents players from accidentally overshooting the
#   bound (porentially receiving a very high penalty) - should be
#   set slightly outside of the actual constraint
function make_bound_step(id, lower, upper; ω=0.0)
    function bound(state::StateDist, gp)
        alter(state,
            id => clamp.(state[id], lower + ω*rand(), upper + ω*rand())
        )
    end

    GameComponent(bound, [id])
end