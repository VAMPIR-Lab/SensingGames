# `fovtag` is an imperfect information pursuit-evasion game where
#   each player has a field of view in the direction of their velocity.
#   When the opponent is within the field of view observations are perfect;
#   otherwise observations are very noisy (with a steep but smooth transition
#   in between).

function make_fovtag_sensing(agent, other; fov=0.8)
    id_obs = Symbol("$(agent)_obs")
    id_own_pos = Symbol("$(agent)_pos")
    id_own_vel = Symbol("$(agent)_vel")
    id_other_pos = Symbol("$(other)_pos")

    
    function dyn!(state::State, history::Vector{State}, game_params)::State
        our_pos = state[id_own_pos]
        their_pos = state[id_other_pos]

        our_vel = state[id_own_vel]

        θ1 = atan(our_vel[2], our_vel[1]) + π
        θ2 = atan(our_pos[2] - their_pos[2], our_pos[1] - their_pos[1])
        dθ = (fov/2) - abs(angdiff(θ1, θ2))
        σ2 = softif(dθ, 0, 5; hardness=2)^2

        their_pos = [
            sample_trunc_gauss(state[id_other_pos][1], σ2, -22, 22)
            sample_trunc_gauss(state[id_other_pos][2], σ2, -22, 22)
        ]

        our_pos = [
            sample_gauss(state[id_own_pos][1], 0.25)
            sample_gauss(state[id_own_pos][2], 0.25)
        ]

        alter(state,
            id_obs => [state[id_own_pos]; their_pos]
        )
    end

    state = State(id_obs => 4), dyn!
end

function make_fovtag_cost(bnd=20)
    function cost_fn(state)
        d = dist2(state[:p1_pos], state[:p2_pos])
        [
            d + cost_bound(state[:p1_pos], [-bnd; -bnd], [bnd; bnd])
           -d + cost_bound(state[:p2_pos], [-bnd; -bnd], [bnd; bnd])
        ]
    end
end

function make_fovtag_prior(zero_state)
    () -> alter(zero_state,
        :p1_pos => [2*randn();  2*randn()],
        :p2_pos => [2*randn(),  2*randn()]
    ) 
end

function render_fovtag!(ax, hists)
    for h in hists
        for agent in [:p1, :p2]
            render_traj!(ax, h, agent)
            render_obs!(ax, h, agent; range=3:4)
            render_heading!(ax, h, agent)
        end
    end
end

function get_fovtag_hists(options, particles, params)
    [last(prt.history, options.n_lookahead) for prt in particles]
end


function test_fovtag()
    state1, sdyn1 = make_acc_dynamics(:p1; control_scale=1, drag=0.5)
    state2, sdyn2 = make_acc_dynamics(:p2; control_scale=1, drag=0.5)
    _, bdyn1 = make_bound_dynamics(:p1_pos, -22, 22)
    _, bdyn2 = make_bound_dynamics(:p2_pos, -22, 22)

    obs1, odyn1 = make_fovtag_sensing(:p1, :p2)
    obs2, odyn2 = make_fovtag_sensing(:p2, :p1)

    ctrl1 = make_horizon_control(:p1, :p1_obs, :p1_acc)
    ctrl2 = make_horizon_control(:p2, :p2_obs, :p2_acc)

    zero_state = merge(state1, state2, obs1, obs2)

    dyn_fns = [odyn1, odyn2, ctrl1, ctrl2, sdyn1, sdyn2, bdyn1, bdyn2]
    prior_fn = make_blurtag_prior(zero_state)
    cost_fn = make_blurtag_cost()

    fovtag_game = SensingGame(prior_fn, dyn_fns, cost_fn)
    options = (;
        parallel = true,
        n_particles = 5,
        n_lookahead = 8,
        n_render = 1,
        n_iters = 400,
        live=true,
        score_mode = :sum
    )
    
    make_model() = Chain(
        Dense(4 => 64, relu),
        Dense(64 => 64, relu),
        Dense(64 => 2)
    )

    initial_params = (; 
        policies = (;
            p1 = FluxPolicy(make_model, 2; t_max=options.n_lookahead),
            p2 = FluxPolicy(make_model, 2; t_max=options.n_lookahead)
        )
    )

    vis_options = (;
        name = "Fov Tag",
        win_size = (800, 600), 
        ax_lims = ((-30, 30), (-30, 30)) 
    )
    
    solve_unrelated_particles(fovtag_game, initial_params, options, get_fovtag_hists, vis_options, render_fovtag!)
end