# The State struct is a convenience to allow indexing vectors by symbols.
#   Managing big state spaces gets annoying so it's convenient to 
#   be able to say, e.g., state[:p1_pos] and know that it's player 1's position,
#   knowing that `state` is basically a 1D vector but not caring about where
#   its components actually live.
# They're also helpful because Zygote doesn't permit array mutation:
#   States are immutable
#   you can use `alter` to get a new State from an old one and a list of substitutions.

using StatsBase
using Base: ImmutableDict

struct State
    z::AbstractArray{Float32}
    ids::Vector{Symbol}
    map::ImmutableDict{Symbol, UnitRange{Int}}
end

function State(semantics::Pair{Symbol, Int}...)
    end_ids = cumsum([s[2] for s in semantics])
    start_ids = [1; end_ids[begin:end-1] .+ 1]

    ids = [s for (s, _) in semantics]

    merge(dict, i) = begin
        ImmutableDict{Symbol, UnitRange{Int}}(dict, 
            semantics[i][1], start_ids[i]:end_ids[i])
    end

    map = foldl(merge, 1:length(semantics), init=ImmutableDict{Symbol, UnitRange{Int}}()) 

    State(zeros(end_ids[end]) |> gpu, ids, map)
end

function State(init_pairs::Pair{Symbol, <: AbstractArray{Float32}}...)

    pairs = [s => length(v) for (s, v) in init_pairs]
    state = State(pairs...)
    alter(state, init_pairs...)
end

function Base.size(s::State)
    Base.size(s.z)
end

function Base.getindex(s::State, i::Int)::Vector{Float32}
    s.z[i]
end

function Base.getindex(s::State, q::Symbol)::Vector{Float32}
    s.z[s.map[q]]
end

function Base.getindex(s::State, I::Vararg{Int})::Vector{Float32}
    s.z[I...]
end

function Base.getindex(s::State, syms::Vector{Symbol})
    mapreduce(vcat, syms) do q
        s[q]
    end
end

function Base.length(s::State)
    length(s.z)
end

function alter(state::State, substitutions::Pair{Symbol, Vector{Float32}}...)::State
    dict = Dict(substitutions...)
    z::Vector{Float32} = mapreduce(vcat, state.ids) do id::Symbol
        if id in keys(dict)
            dict[id] |> gpu
        else
            state[id]
        end
    end

    State(z, state.ids, state.map)
end

function merge(states::State...)
    
    z = mapreduce(s -> s.z, vcat, states)
    ids = mapreduce(s -> s.ids, vcat, states)

    n = 0
    map = ImmutableDict(mapreduce(vcat, states) do s
        t = n
        n += length(s.z)
        [id => loc.+t for (id, loc) in s.map]
    end...)

    State(z, ids, map)
end 

function select(state::State, ids::Symbol...)
    new_state = State(
        (ids .=> [length(state.map[id]) for id in ids])...
    )
    alter(new_state,
        [id => state[id] for id in ids]...
    )
end

function unspool(state::State, pairs...)
    # For N pairs of IDs (:a, :b), interprets each :a as a list of :b
    #   and creates a new state with each :b in that list.
    #   Useful for parsing histories.
    #   Assumes |a|/|b| is the same for all pairs
    #   (each :a can be divided into the same number of :b's)

    N = length(state[pairs[1][1]]) ÷ length(state[pairs[1][2]])


    reverse(map(1:N) do i
        init_pairs = map(1:length(pairs)) do j
            n_b = length(state[pairs[j][2]])
            range = ((i-1)*n_b+1) : ((i)*n_b)
            pairs[j][2] => state[pairs[j][1]][range]
        end
        State(init_pairs...)
    end)
end

function Flux.gpu(s::State)
    State(
        s.z |> gpu,
        s.ids,
        s.map
    )
end

function Flux.cpu(s::State)
    State(
        s.z |> cpu,
        s.ids,
        s.map
    )
end



# Each row of z is a State
# w is the weight factor for each state
# ids are the components of a State
# map gives the index range for ids
struct StateDist
    z::AbstractArray{Float32}
    w::AbstractArray{Float32}
    ids::Vector{Symbol}
    map::ImmutableDict{Symbol, UnitRange{Int}}
end

function StateDist(state::State, n::Int64)
    StateDist(
        Base.repeat(state.z', n) |> gpu,
        log.(1/n * ones(n)) |> gpu,
        state.ids,
        state.map
    )
end

function StateDist(states::AbstractArray{State})
    zs = [s.z' for s in states]
    StateDist(
        reduce(vcat, zs) |> gpu,
        fill(-log(length(states)), length(states)) |> gpu,
        states[begin].ids,
        states[begin].map
    )
end

function Base.size(s::StateDist)
    size(s.z)
end

function Base.getindex(s::StateDist, i::Union{Int, Colon, UnitRange{Int}})
    State(s.z[i, :], s.ids, s.map)
end

function Base.getindex(s::StateDist, q::Symbol)
    s.z[:, s.map[q]]
end

function Base.getindex(s::StateDist, syms::Vector{Symbol})
    mapreduce(hcat, syms) do q
        s[q]
    end
end

function Base.length(s::StateDist)
    size(s.z)[1]
end

function expectation(f, s::StateDist)
    sum(1:length(s)) do i
        c = f(s[i])
        p = exp(s.w[i])
        p * c
    end
end

function alter(state::StateDist, substitutions::Pair...)::StateDist
    
    z_buf = Zygote.Buffer(state.z)
    z_buf[:, :] = state.z

    for (id, v) in substitutions
        z_buf[:, state.map[id]] = Float32.(v) |> gpu
    end

    StateDist(copy(z_buf), state.w, state.ids, state.map)
end


function select(state_dist::StateDist, ids::Symbol...)
    new_state = State(
        (ids .=> [length(state_dist.map[id]) for id in ids])...
    )
    new_state_dist = StateDist(new_state, length(state_dist))
    alter(new_state_dist,
        [id => state_dist[id] for id in ids]...
    )
end

function reweight(state::StateDist, weights)
    StateDist(state.z |> gpu, weights |> gpu, state.ids, state.map)
end

function draw(dist::StateDist; n=1, as_dist=true, reweight=true, weight=true)
    # Zygote.ignore() do
        if weight
            idxs = wdsample(1:length(dist), exp.(dist.w), n)
        else
            idxs = rand(1:length(dist), n)
        end
        if as_dist
            w = if reweight
                log.(exp.(dist.w[idxs]) ./ sum(exp.(dist.w[idxs])))
            else
                dist.w[idxs]
            end
            StateDist(dist.z[idxs, :], w, dist.ids, dist.map)
        else
            map(idxs) do i
                State(dist.z[i, :], dist.ids, dist.map)
            end
        end
    # end
end


function subdist(states::StateDist; n=20, weighted=true)
    if n > length(states)
        return states
    end
    weights = (weighted) ? Weights(exp.(states.w)) : Weights(ones(length(states)))
    idxs = StatsBase.sample(1:length(states), weights, n)
    w = log.(exp.(states.w[idxs]) ./ sum(exp.(states.w[idxs])))
    StateDist(states.z[idxs, :], w, states.ids, states.map)
end


function subdist(states::StateDist, idxs)
    w = log.(exp.(states.w[idxs]) ./ sum(exp.(states.w[idxs])))
    StateDist(states.z[idxs, :], w, states.ids, states.map)
end

function Base.copy(dist::StateDist)
    StateDist(
        copy(dist.z),
        copy(dist.w),
        dist.ids,
        dist.map
    )
end

function Base.show(io::IO, ::MIME"text/plain", s::State)
    if isempty(s.ids)
        print(io, "Empty State")
        return
    end

    str = "State\n" * mapreduce(*, s.ids) do id
        n_spaces = 10 - length(string(id))

        label = "| " * string(id) * repeat(" ", n_spaces) * "| " 
        
        data = mapreduce(*, enumerate(s[id])) do (i, v)
            @sprintf("%.3f", v) * ((i%10==0) ? "\n|           | " : "\t")
        end * "\n"

        label * data
    end

    print(io, str)
end


function Flux.gpu(s::StateDist)
    StateDist(
        s.z |> gpu,
        s.w |> gpu,
        s.ids,
        s.map
    )
end


function Flux.cpu(s::StateDist)
    StateDist(
        s.z |> cpu,
        s.w |> cpu,
        s.ids,
        s.map
    )
end


# When state spaces get more particles 
#   (as a result of e.g. cross dynamics)
#   the history can have inconsistent numbers 
#   of particles
function debranch_history(hist)
    h = size(hist[end])[1]

    map(hist) do dist
        n = h ÷ length(dist)
        z_new = repeat(dist.z, n)
        w_new = repeat(dist.w, n)
        StateDist(z_new, w_new, state.ids, dist.map)
    end
end

Zygote.@adjoint State(semantics...) = State(semantics...), _ -> (nothing)
Zygote.@adjoint State(z::AbstractVector, ids, map) = State(z, ids, map), p -> (p.z, nothing, nothing)
Zygote.@adjoint StateDist(z, map) = StateDist(z, map), p -> (p.z, nothing)


