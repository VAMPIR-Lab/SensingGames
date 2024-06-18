# `meetup` is a pursuit-pursuit game where two players
#   both want to be near each other, but must first locate their
#   partner by going to a specific point.

# Framework has changed a lot since this was useful - 
#   to be refactored when it becomes relevant again

# function make_meetup_sensing(agent, other, targs)
#     id_obs = Symbol("$(agent)_obs")
#     id_own_pos = Symbol("$(agent)_pos")
#     id_other_pos = Symbol("$(other)_pos")

#     state = State(
#         id_obs => 4
#     )
    
#     function dyn!(history, game_params)
#         state = history[end]
#         d = [dist2(state[id_own_pos], beacon) for beacon in targs]
#         σ2 = 5*minimum(d)
#         us = state[id_own_pos]
#         them = [
#             sample_gauss(state[id_other_pos][1], σ2)[1]
#             sample_gauss(state[id_other_pos][2], σ2)[1]
#         ]

#         alter(state,
#             id_obs => [us; them]
#         )
#     end

#     state, dyn!
# end

# function make_meetup_cost(agent::Symbol, partner::Symbol)
#     id_us = Symbol("$(agent)_pos")
#     id_them = Symbol("$(partner)_pos")

#     function cost(state)
#         sqrt(dist2(state[id_us], state[id_them]))
#     end
# end

# function render_meetup_game(hists, targs)

#     agents = [:p1,   :p2]
#     colors = [:blue, :red]

#     plt = plot(lims=(-3, 3))
#     for i in 1:2
#         for states in hists
#             id_pos = Symbol("$(agents[i])_pos")
#             id_Fθ = Symbol("$(agents[i])_Fθ")

#             state_x = [s[id_pos][1] for s in states]
#             state_y = [s[id_pos][2] for s in states]

#             plot!(state_x, state_y,
#                 color=colors[i],
#                 alpha=0.1,
#                 label=""
#             )

#             plot!(state_x, state_y,
#                 seriestype=:scatter,
#                 color=colors[i],
#                 alpha=0.4,
#                 label=""
#             )

#             headings_x = mapreduce(vcat, states) do state
#                 [state[id_pos][1]
#                 state[id_pos][1] + cos(state[id_Fθ][2]) * 0.2
#                  NaN]
#             end

#             headings_y = mapreduce(vcat, states) do state
#                 [state[id_pos][2]
#                 state[id_pos][2] + sin(state[id_Fθ][2]) * 0.2
#                  NaN]
#             end

#             plot!(headings_x, headings_y,
#                 color=colors[i],
#                 alpha=0.4,
#                 label="",
#                 linestyle=:dash
#             )
#         end

#         for targ in targs[i]
#             plot!([targ[1]], [targ[2]],
#                 color=colors[i],
#                 seriestype=:scatter,
#                 alpha=0.7,
#                 markershape=:star5,
#                 label=""
#             )
#         end
#     end
#     display(plt)
# end

# function test_meetup_game()

#     targs = [
#         [[-1; -1],],
#         [[ 1;  1],]
#     ]

#     state1, sdyn1 = make_unicycle_dynamics(:p1)
#     state2, sdyn2 = make_unicycle_dynamics(:p2)

#     obs1, odyn1 = make_meetup_sensing(:p1, :p2, targs[1])
#     obs2, odyn2 = make_meetup_sensing(:p2, :p1, targs[2])

#     ctrl1 = make_horizon_control(:p1, :p1_obs, :p1_Fθ)
#     ctrl2 = make_horizon_control(:p2, :p2_obs, :p2_vel)

#     initial_state = merge(state1, state2, obs1, obs2)
#     initial_state = alter(initial_state,
#         :p1_pos => [randn(); -0.3],
#         :p2_pos => [randn(),  0.3]
#     )

#     game_params = (; 
#         policies = Dict([
#             :p1 => LinearPolicy(4, 2; t_max=8)
#             :p2 => ZeroPolicy(2)
#         ])
#     )

#     prior = () -> alter(initial_state,
#         :p1_pos => [-1.0 + randn();  1.5 + randn()],
#         :p2_pos => [ 2.0 + randn(),  0.5 + randn()]
#     ) 

#     game = SensingGame(
#         [sdyn1, sdyn2, odyn1, odyn2, ctrl1, ctrl2],
#         prior
#     )

#     cost = make_meetup_cost(:p1, :p2)

#     run_game = params -> rollout(game, params, T=5)
#     score = params -> mapreduce(_ -> cost(run_game(params)[end]), +, 1:5)

#     for t in 1:10000
#         println("t=$(t)")

#         if t % 10 == 0
#             hists = [run_game(game_params) for _ in 1:1]
#             println(hists[1][1][:p1_Fθ])
#             render_meetup_game(hists, targs)
#         end

#         grads = Flux.gradient(score, game_params)[1]

#         apply_gradient!(game_params.policies[:p1], grads.policies[:p1][])
#         # apply_gradient!(game_params.policies[:p2], grads.policies[:p2][])
#     end
# end