function make_hist_step(id_hist, id_store, t_max)
    # Stores from [newest... to ... oldest]
    function hist_step(dist, params)
        m = size(dist[id_store])[2]
        hist = [dist[id_store] dist[id_hist]][:, begin:(m*t_max)]
        
        alter(dist,
            id_hist => hist
        )
    end

    GameComponent(hist_step, [id_hist])
end

function make_clock_step(dt)
    function clock(dist, params)
        t = Zygote.ignore() do 
            dist[:t] 
        end
        alter(dist, :t => t .+ dt)
    end

    GameComponent(clock, [:t])
end

# function make_cross_step(dyn_fn, ll_fn, n, alter_ids)

#     num_counterfactuals = ((n isa AbstractVector) ? (t) -> n[t] : (t) -> n) 
#     function cross_step(state_dist, game_params)

#         t = Int(state_dist[:t][1])
#         r = num_counterfactuals(t)
#         N = length(state_dist)
        
#         quantile = rand(N, 2) 

#         new_state_dist = dyn_fn(state_dist, game_params, quantile)
#         if r <= 0
#             return new_state_dist
#         end

#         lls_n = ll_fn(state_dist, new_state_dist)
#         lls_n_norm = sum(exp.(lls_n), dims=2)    

#         # confusion_matrix = (lls_n .- log.(lls_n_norm) )
#         # confusion_matrix = confusion_matrix .+ info_dist.w
#         # confusion_matrix = exp.(confusion_matrix)
#         # confusion_matrix = (.! I(N)) .* confusion_matrix


#         confusion_entries = Zygote.ignore() do
#             if r == Inf
#                 [Iterators.product(1:N, 1:N)...]
#             else
#                 rand([Iterators.product(1:N, 1:N)...], r)
#             end
#         end
#         ground_entries = [(i, i) for i in 1:N]
#         all_entries = (r == Inf) ? confusion_entries : [ground_entries; confusion_entries]


#         confusion_mask_buffer = Zygote.bufferfrom(fill(false, (N, N)))
#         for entry in confusion_entries
#             confusion_mask_buffer[entry...] = true
#         end
#         confusion_mask = copy(confusion_mask_buffer)
#         ground_mask = I(N)
#         all_mask = confusion_mask .|| ground_mask
        

#         liks = exp.(lls_n)
#         new_liks = all_mask .* (liks)
#         liks_norm = sum(new_liks, dims=2) 
#         new_liks_normed = new_liks ./ liks_norm
#         new_liks_marg = new_liks_normed .* exp.(state_dist.w)
#         new_liks_masked = all_mask .* (new_liks_marg)

#         new_w = map(all_entries) do entry
#             log(new_liks_masked[entry...])
#         end

#         ground_rows = [a for (a, b) in all_entries]
#         confusion_rows = [b for (a, b) in all_entries]

#         dist = StateDist(
#             state_dist.z[ground_rows, :],
#             new_w,
#             state_dist.ids,
#             state_dist.map
#         )

#         alterations = [id => new_state_dist[id][confusion_rows, :] for id in alter_ids]

#         alter(dist, alterations...)
#     end
# end