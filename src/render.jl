abstract type Renderer end

mutable struct MakieRenderer <: Renderer
    figure::Union{Nothing, Figure}
    problem_data::Union{Nothing, Observable}
    observables
    observable_idx
    axes
end


function MakieRenderer()
    fig = Figure()
    MakieRenderer(
        fig,
        nothing,
        [],
        0,
        fill([], (10, 10)) # horrible hack
    )
end


function _get_axis(r::MakieRenderer, axis_idx)
    if isempty(r.axes[axis_idx...])
        r.axes[axis_idx...] = [Axis(r.figure[axis_idx...], limits=(-50, 50, -50, 50))]
    end
    r.axes[axis_idx...][]
end

function _update_observable(r::MakieRenderer, data)
    r.observable_idx += 1
    idx = r.observable_idx

    if idx > length(r.observables)
        o = Observable(data)
        r.observables = [r.observables; o]
        return o
    else
        r.observables[idx][] = data
        return nothing
    end
end


function render(f, r::MakieRenderer)
    r.observable_idx = 0
    f()
    Base.display(r.figure)
end


function render_trajectory(r::MakieRenderer, states::AbstractArray{State}, id; ax_idx,
    color=:black, alpha=0.2, kwargs...)
    states = _update_observable(r, states)
    isnothing(states) && return
    ax = _get_axis(r, ax_idx)
    
    x = @lift([state[id][1] for state in $states])
    y = @lift([state[id][2] for state in $states])

    lines!(ax, x, y;
        color, alpha, kwargs...
    )

    a_x = @lift([$x[end-1]])
    a_y = @lift([$y[end-1]])
    a_dx = @lift([$x[end] - $x[end-1]])
    a_dy = @lift([$y[end] - $y[end-1]])

    arrows!(ax, a_x, a_y, a_dx, a_dy;
        color, alpha, kwargs...
    )
end


function render_location(r::MakieRenderer, state::State, id::Symbol; ax_idx,
        color=:black, alpha=0.8, kwargs...)
    state = _update_observable(r, state)
    isnothing(state) && return
    ax = _get_axis(r, ax_idx)

    x = @lift([$state[id][1]])
    y = @lift([$state[id][2]])

    scatter!(ax, x, y;
        color, alpha, kwargs...
    )
end


function render_fov(r::MakieRenderer, state::State, fov, id_pos, id_θ; ax_idx, 
        color=:black, alpha=0.2, scale=5, kwargs...)
    state = _update_observable(r, state)
    isnothing(state) && return
    ax = _get_axis(r, ax_idx)

    vertices = @lift(
        [
            (
                $state[id_pos][1], 
                $state[id_pos][2]
            ), (
                $state[id_pos][1] + scale*cos($state[id_θ][1] + fov/2),
                $state[id_pos][2] + scale*sin($state[id_θ][1] + fov/2)
            ), (
                $state[id_pos][1] + scale*cos($state[id_θ][1] - fov/2),
                $state[id_pos][2] + scale*sin($state[id_θ][1] - fov/2)
            )
        ]
    )
    poly!(ax, vertices; 
        color, alpha, kwargs...
    )
end