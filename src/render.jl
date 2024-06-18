
# Renders :p<agent>_pos (agent position) as a line and scatter plot 
function render_traj(hist, agent)
    color = (; p1=:blue, p2=:red, p3=:green)[agent]
    id_pos = Symbol("$(agent)_pos")

    state_x = [s[id_pos][1] for s in hist]
    state_y = [s[id_pos][2] for s in hist]

    plot!(state_x, state_y,
            color=color, alpha=0.1, label="")

    plot!(state_x, state_y,
            seriestype=:scatter,
            color=color, alpha=0.8, label="")
end

# Renders :p<agent>_obs (agent's observation) as a scatter plot
# By default it is assumed the first two elements of the observation
# are (x,y), but different elements can be selected with the `range` parameter
function render_obs(hist, agent; range=1:2)
    color = (; p1=:purple, p2=:orange, p3=:yellow)[agent]
    id_obs = Symbol("$(agent)_obs")

    obs_x = [s[id_obs][range][1] for s in hist]
    obs_y = [s[id_obs][range][2] for s in hist]

    plot!(obs_x, obs_y,
            seriestype=:scatter,
            color=color, alpha=0.5, label="")
end

# Renders agent's velocity heading as a small line in that direction.
# The line may look crooked on non-square visualizations; this is correct 
function render_heading(hist, agent)
    color = (; p1=:blue, p2=:red, p3=:green)[agent]
    id_pos = Symbol("$(agent)_pos")
    id_vel = Symbol("$(agent)_vel")

    θ = atan(hist[end][id_vel][2], hist[end][id_vel][1])
    heading_x = hist[end][id_pos][1] .+ [0; cos(θ); NaN]
    heading_y = hist[end][id_pos][2] .+ [0; sin(θ); NaN]

    plot!(heading_x, heading_y,
            color=color, alpha=0.4, label="")
end