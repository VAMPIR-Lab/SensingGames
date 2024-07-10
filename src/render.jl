
# Renders :p<agent>_pos (agent position) as a line and scatter plot 
function render_traj(hist, agent; m=4, prt=nothing)
    id_hist = Symbol(agent, "_hist")
    end_dist = hist[end]
    color = (; p1=:blue, p2=:red, p3=:green)[agent]

    # T = 2
    T = length(hist)
    # We'd like to go particle-by-particle 
    #   rather than dist-by-dist
    hist = debranch_history(hist)

    infosets = unique(hist[end][:p1_info])


    for p in (isnothing(prt) ? (1:length(end_dist)) : [prt]) 

        
        state = end_dist[p]
        p1_set = abs(Int(state[:p1_info][1]))
        color = [:red, :red, :blue, :blue][findfirst(infosets .== p1_set)]

        h = reshape(state[id_hist], (m, :))
        state_x = h[1, :]
        state_y = h[2, :]

        state_alpha = isnothing(prt) ? exp(hist[end].w[p]) : 1.0
        
        labels = map(1:T) do t
            string(t)
        end

        plot!(state_x, state_y,
                color=color, alpha=.4*(0.8*state_alpha + 0.2), label="", hover=labels)

        plot!(state_x, state_y,
                seriestype=:scatter,
                color=color, alpha=state_alpha, label="", hover=labels)
    end
end

function render_obs(hist, agent; m=4, prt=nothing)
    id_hist = Symbol(agent, "_hist")
    end_dist = hist[end]
    color = (; p1=:purple, p2=:orange, p3=:cyan)[agent]

    # We'd like to go particle-by-particle 
    #   rather than dist-by-dist
    hist = debranch_history(hist)

    T = length(hist)

    for p in (isnothing(prt) ? (1:length(end_dist)) : [prt]) 
        
        state = end_dist[p]
        h = reshape(state[id_hist], (m, :))
        obs_x = h[3, :]
        obs_y = h[4, :]

        # state_alpha = isnothing(prt) ? exp(hist[end].w[p]) : 1.0
        state_alpha = 0.3

        # hover = ["1"; "2"; "3"]
        labels = map(1:T) do t
            string(t)
        end

        plot!(obs_x, obs_y,
                seriestype=:scatter,
                color=color, alpha=state_alpha, label="", hover=labels)
    end
end


function render_heading(hist, agent; m=4, fov=2.0, scale=5, prt=nothing)
    id_hist = Symbol(agent, "_hist")
    end_dist = hist[end]
    color = (; p1=:black, p2=:black, p3=:green)[agent]

    for p in (isnothing(prt) ? (1:length(end_dist)) : [prt]) 
        
        state = end_dist[p]

        h = reshape(state[id_hist], (m, :))
        state_x = h[1, :]
        state_y = h[2, :]
        v_x = state_x[1:(end-1)] - state_x[2:end]
        v_y = state_y[1:(end-1)] - state_y[2:end]
        state_x = state_x[1:(end-1)]
        state_y = state_y[1:(end-1)]

        v_x = v_x
        v_y = v_y
        state_x = state_x
        state_y = state_y
        

        spacer = fill(NaN, length(state_x))
        θ = atan.(v_y, v_x)

        area_x = vec(hcat(
            state_x,
            state_x .+ scale * cos.(θ .+ fov/2),
            spacer,
            state_x,
            state_x .+ scale * cos.(θ .- fov/2),
            spacer
        )')

        area_y = vec(hcat(
            state_y,
            state_y .+ scale * sin.(θ .+ fov/2),
            spacer,
            state_y,
            state_y .+ scale * sin.(θ .- fov/2),
            spacer
        )')

        state_alpha = isnothing(prt) ? exp(hist[end].w[p]) : 1.0

        plot!(area_x, area_y,
                color=color, alpha=state_alpha, label="", linestyle=:dot)
    end
end