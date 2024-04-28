
# function history_to_index(obs; D=3)
#     @bp
#     obs = obs[obs .>= 0]
#     obs = [zeros(D - length(obs)); 1; obs]
#     mapreduce(+, enumerate(obs)) do (i, o)
#         Int(o * 2^(i-1))
#     end
# end

# # function policy(obs; θ, D=2)
# #     n_obs = size(obs)[2]
# #     if n_obs < D
# #         obs = [-ones(2, D-n_obs) obs]
# #     end
# #     obs = [obs...]'
# #     tanh.(θ.out * lrelu.(θ.hidden * lrelu.(θ.in * obs' + θ.in_b) + θ.hidden_b)) ./ 8.0
# # end

# # function sample_gauss(rand, mean, var)
# #     # Bowling approximation of inverse CDF
# #     v = log(1/rand - 1) * sqrt(var) / -1.702 + mean
# #     p =  exp(-0.5*(v - mean)^2 / var) / sqrt(2 * pi * var)
# #     return v,p
# # end



# # function sample_cost(θ, ω, n; T=3, D=2)
    
# # end

# # function sample_gradient(θ, shape, n; T=3, D=2)
# #     f = θ ->sample_cost(θ, shape, n; T, D)
# #     ForwardDiff.gradient(f, θ)
# # end

# function params_shape(T=3, D=2)
#     θ = (;
#         # pol=(2, 2^(D+1))
#         in=(20, D*2),
#         out=(2, 20),
#         in_b=(20, 1),
#         hidden=(20, 20),
#         hidden_b=(20, 1)
#     )
#     n_θ = mapreduce(x -> prod(x), +, θ)

#     ω = (
#         obs=(2, T+1),
#         # init_obs=(1, T),
#         prior=(2,)
#     )
#     n_ω = mapreduce(x -> prod(x), +, ω)

#     return (; θ, ω, n_θ, n_ω)
# end

# function from_raw(z, shape)
#     θ, ω, nθ, nω = shape
#     i = 0
#     map((; θ..., ω...)) do s
#         data = reshape(z[i+1:i+prod(s)], s)
#         i += prod(s)
#         data
#     end
# end

# function to_raw(params)
#     mapreduce(vcat, params) do p
#         [p...]
#     end
# end

# # function compile_problem()
# #     T = 3
# #     shape = params_shape(T)
# #     n = shape.n_θ + shape.n_ω

# #     @variables z[1:n]
# #     z = Symbolics.scalarize(z) 

# #     println("Compiling cost...")
# #     f = cost(sample(from_raw(z, shape))...)
# #     f_fn = eval(build_function(f, z))
# #     # f_pfn = ((prm) -> f_fn(to_raw(prm)))

# #     println("Compiling gradient...")
# #     df = Symbolics.gradient(f, z[1:shape.n_θ])
# #     df_fn = eval(build_function(df, z)[1])
# #     # df_pfn = ((prm) -> df_fn(to_raw(prm)))

# #     return f_fn, df_fn
# # end

# function render_particle(state_hist, obs_hist, prob; ax)

#     obs_colors = [:blue, :red]

#     # Observation beacon (hardcoded)
#     Makie.scatter!([0], [0], color=:orange)
#     Makie.scatter!([0], [0], color=:orange, markersize=20, alpha=0.5)

#     # Goal
#     Makie.scatter!([1.0], [-1.0], color=:red)
    
#     s0 = state_hist[:, begin]

#     Makie.text!([s0[1]], [s0[2]-0.1], text="Start", color=:black, align=(:center, :center))

#     for (s, o) ∈ zip(eachcol(state_hist[:, 2:end]), eachcol(obs_hist[:, begin:end]))
#         x = [s0[1]; s[1]]
#         y = [s0[2]; s[2]]
#         Makie.lines!(x, y; color=:black)

#         Makie.scatter!([s0[1]], [s0[2]], color=:black, markersize=5)
        
#         Makie.scatter!([o[1]], [o[2]], color=:blue, markersize=5)
#             # Makie.scatter!([s0[1]], [s0[2]], color=exp(-o[1]), colorrange=(0, 1), markersize=15, alpha=0.5)
#         s0 = s
#     end
# end

# # function iterate_soln(;ax, num_rendered=1, T=3, D=2)

# #     limits!(ax, -1.5, 1.5, -1.5, 1.5)
    
# #     shape = params_shape(T, D)
# #     θ = rand(shape.n_θ) .- 0.5
# #     res_hist = []

# #     costs = []


# #     for i in 1:100000
# #         ω = rand(shape.n_ω)
# #         z = [θ; ω]
# #         res = sample(from_raw(z, shape); T, D)
# #         push!(costs, cost(res...))
        
# #         if i % 1 == 0
# #             push!(res_hist, res)
        
# #             if length(res_hist) > num_rendered
# #                 res_hist = res_hist[end-num_rendered+1:end]
# #             end

# #             empty!(ax)
# #             for res in res_hist
# #                 render_particle(res...; ax)
# #             end
# #         end

# #         # Report cost
# #         nc = min(length(costs), 100)
# #         println(sum(costs[end-nc+1:end]) / nc)
        
# #         # Descent
# #         grad = sample_gradient(θ, shape, 10; T, D)
# #         # θ -= (10/(i+100)) * grad
# #         θ -= 0.02 * grad
# #     end
# # end