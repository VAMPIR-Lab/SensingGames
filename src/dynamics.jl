# These define dynamics that operate on States.
#   Each make_<?>_dynamics function outputs a zero State with any
#   state variables the dynamics define, and a function that rolls 
#   the dynamics a single step. 

# Planar velocity integrator
function make_vel_dynamics_step(agent::Symbol; dt=1.0, dims=2, control_scale=1.0)
    id_pos = Symbol("$(agent)_pos")
    id_vel = Symbol("$(agent)_vel")

    function vel_dyn(state::StateDist, game_params)
        pos = state[id_pos] + dt*state[id_vel]*control_scale
        if agent == :p1
            pos = pos .+ randn(size(pos)...)
        end
        q = alter(state,
            id_pos => pos
        )
        q
    end
end

# Planar acceleration integrator
function make_acc_dynamics_step(agent::Symbol; dt=1.0, dims=2, drag=0.0, control_scale=1.0, max_vel=100.0)
    id_pos = Symbol("$(agent)_pos")
    id_vel = Symbol("$(agent)_vel")
    id_acc = Symbol("$(agent)_acc")

    function acc_dyn(state::StateDist, game_params)
        acc = state[id_acc] .* control_scale
        vel = state[id_vel] .+ acc*dt
        vel = softclamp.(vel, -max_vel, max_vel)
        pos = state[id_pos] .+ state[id_vel] .+ 0.5.*acc*dt.^2

        alter(state, 
            id_pos => pos,
            id_vel => vel
        )
    end
end

# Unicycle dynamics
function make_unicycle_dynamics_step(agent::Symbol; dt=1.0)
    id_pos = Symbol("$(agent)_pos")
    id_θ = Symbol("$(agent)_θ")
    id_u = Symbol("$(agent)_u")

    function unicycle_dyn(state::StateDist, game_params)
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
end

# Bound: Gently bounces players off a given bound
#   This prevents players from accidentally overshooting the
#   bound (porentially receiving a very high penalty) - should be
#   set slightly outside of the actual constraint
function make_bound_step(id, lower, upper; ω=1.0)
    function bound_dyn(state::StateDist, gp)
        alter(state,
            id => softclamp.(state[id], lower + ω*rand(), upper + ω*rand())
        )
    end
end