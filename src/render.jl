abstract type Renderer end

mutable struct MakieRenderer <: Renderer
    figure::Figure
    state_dist::Union{Nothing, Observable{StateDist}}
end

function MakieRenderer()
    fig = Figure()
    Base.display(fig)
    MakieRenderer(
        fig,
        nothing
    )
end
# Render an entire distribution
#   Doesn't require render context (top level)
function render_dist(f, renderer::MakieRenderer, state_dist::StateDist; kargs...)
    if isnothing(renderer.state_dist)
        # We haven't rendered yet - let's set up
        renderer.state_dist = Observable(state_dist)
        for i in 1:length(state_dist)
            weight=@lift(exp($(renderer.state_dist).w[i]))
            
            row = (i-1)%4+1
            col = (i-1)÷4+1
            max_row = (length(state_dist)-1)%4+1

            lims = default(kargs, :lims, (-50, 50))

            backgroundcolor = @lift($weight < 0.0005 ? :grey : :white)
            
            ax = Axis(
                renderer.figure[row, col];
                backgroundcolor
                )
            row == max_row || hidexdecorations!(ax, grid=false)
            col == 1 || hideydecorations!(ax, grid=false)
            xlims!(ax, lims)
            ylims!(ax, lims)


            context = (; 
                renderer, 
                prt=i, 
                lims,
                ax,
                weight=1.0 # For testing purposes
            )

            f((@lift($(renderer.state_dist)[i]), context))

            info_1 = @lift(Int($(renderer.state_dist)[i][:p1_info][]))
            info_2 = @lift(Int($(renderer.state_dist)[i][:p2_info][]))

            weight_string = @lift(@sprintf("%.2f", $weight*100))
            label = @lift(
                "Reality #($($info_1), $($info_2))" * 
                # string(i) * 
                ": " * 
                $weight_string * 
                "%")
            text!(ax, lims[1]+5, lims[2]-10, text=label)
        end
    else
        # We have already rendered; we can just update the Observables
        renderer.state_dist[] = state_dist
    end
end


# Renders agents with colors
#   For now, the same across all renderers
function render_agents(f, agents, context)
    colors_primary = (; p1=:blue, p2=:red, p3=:green)
    colors_secondary = (; p1=:purple, p2=:orange, p3=:yellow)

    for agent in agents
        context = (; context...,
            color_primary=colors_primary[agent],
            color_secondary=colors_secondary[agent],
            agent
        )
        f((agent, context))
    end
end

# Render a trajectory in `id` (with connecting lines)
#  e.g. `id = :pos` will render positional trajectory
function render_traj(renderer::MakieRenderer, states, id, context; fov=1, scale=5)
    points = @lift(mapreduce(state -> state[id]', vcat, $states))

    color = default(context, :color_primary, :black)
    alpha = default(context, :weight, 1.0)
    ax = context.ax

    directions = @lift($(points)[begin+1:end, :] .- $(points)[begin:end-1, :])

    θ = @lift(atan.($directions[:, 2], $directions[:, 1]))

    seg_pts = @lift($points[2:end, :])

    map(1:length(states[])-1) do i
        vertices = @lift([
            ($seg_pts[i, 1], $seg_pts[i, 2]), 
            ($seg_pts[i, 1]+scale*cos($θ[i]+fov/2), $seg_pts[i, 2]+scale*sin($θ[i]+fov/2)),
            ($seg_pts[i, 1]+scale*cos($θ[i]-fov/2), $seg_pts[i, 2]+scale*sin($θ[i]-fov/2)) 
        ])

        poly!(ax, vertices; color, alpha=0.2*alpha)
    end


    scatter!(ax,
        @lift([$points[begin, 1]]),
        @lift([$points[begin, 2]]);
        color, alpha
    )

    lines!(ax, points; color, alpha)

    arrows!(ax, 
        @lift([$points[end-1, 1]]), 
        @lift([$points[end-1, 2]]), 
        @lift([$directions[end, 1]]), 
        @lift([$directions[end, 2]]);
        color, alpha=alpha*0.5
    )
end


# Renders points in a scattered manner 
function render_points(renderer::MakieRenderer, states, id, context)
    raw_pairs = @lift(mapreduce(state -> state[id]', vcat, $states)')
    pairs = @lift(clamp.($raw_pairs, (context.lims .* 0.95)...))

    t = string.(1:length(states[]))

    color = default(context, :color_primary, :grey)
    alpha = default(context, :weight, 1.0) * 0.5
    ax = context.ax

    scatter!(ax, pairs; color, alpha, markersize=20, marker='o')
    text_pairs = @lift($pairs .- [1 1.6])
    text!(ax, text_pairs , text=t, fontsize=8)

    for i in 1:length(states[])
        raw = @lift($raw_pairs[:, i])
        clamped = @lift($pairs[:, i])
        dist = @lift(sqrt(dist2($raw, $clamped)))
        text = @lift(@sprintf("%.0fm", $dist))
        visible = @lift($dist > 0)

        placement = @lift(
            ($raw[1] > $clamped[1]) ? (:left) :
            ($raw[1] < $clamped[1]) ? (:right) :
            ($raw[2] > $clamped[2]) ? (:below) : (:above)
        )

        tooltip!(ax, @lift($clamped[1]), @lift($clamped[2]); 
            fontsize=8, text, visible, placement, outline_linewidth=0.5,
            textpadding=(2, 2, 2, 2))
    end
end

function render_info(renderer::MakieRenderer, text, line_num, context)
    color = default(context, :color_primary, :grey)
    lims = context.lims
    ax = context.ax

    text!(ax, lims[1] + 5, lims[1] + (line_num-1)*6 + 5; text, color, fontsize=10)
end