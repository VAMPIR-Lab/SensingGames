# The State struct is a convenience to allow indexing vectors by symbols.
#   Managing big state spaces gets annoying so it's convenient to 
#   be able to say, e.g., state[:p1_pos] and know that it's player 1's position,
#   knowing that `state` is basically a 1D vector but not caring about where
#   its components actually live.
# They're also helpful because Zygote doesn't permit array mutation:
#   States are immutable
#   you can use `alter` to get a new State from an old one and a list of substitutions.

using Base: ImmutableDict

struct State
    z::Vector{Float64}
    ids::Vector{Symbol}
    map::ImmutableDict{Symbol, UnitRange{Int}}
end

function State(semantics::Pair{Symbol, Int}...)
    end_ids = cumsum([s[2] for s in semantics])
    start_ids = [1; end_ids[begin:end-1] .+ 1]

    # q = map(1:length(semantics)) do i
    #     semantics[i][1] => start_ids[i]:end_ids[i]
    # end

    k = [s for (s, _) in semantics]

    merge(dict, i) = begin
        ImmutableDict{Symbol, UnitRange{Int}}(dict, 
            semantics[i][1], start_ids[i]:end_ids[i])
    end

    m = foldl(merge, 1:length(semantics), init=ImmutableDict{Symbol, UnitRange{Int}}()) 

    State(
        zeros(end_ids[end]),
        k,
        m
    )
end


function Vector{State}(z, semantics::Pair{Symbol, Int}...)
    end_ids = cumsum([s[2] for s in semantics])
    start_ids = [1; end_ids[begin:end-1] .+ 1]

    k = [s for (s, _) in semantics]

    merge(dict, i) = begin
        ImmutableDict{Symbol, UnitRange{Int}}(dict, 
            semantics[i][1], start_ids[i]:end_ids[i])
    end

    m = foldl(merge, 1:length(semantics), init=ImmutableDict{Symbol, UnitRange{Int}}()) 

    Z = reshape(z, (end_ids[end], :))'
    [State(
        Z[i, :],
        k,
        m
    ) for i in 1:size(Z)[1]]
end

function Base.size(s::State)
    Base.size(s.z)
end

function Base.getindex(s::State, i::Int)::Vector{Float64}
    s.z[i]
end

function Base.getindex(s::State, q::Symbol)::Vector{Float64}
    s.z[s.map[q]]
end

function Base.getindex(s::State, I::Vararg{Int})::Vector{Float64}
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

function alter(state::State, substitutions::Pair{Symbol, Vector{Float64}}...)::State
    dict = Dict(substitutions...)
    z::Vector{Float64} = mapreduce(vcat, state.ids) do id::Symbol
        if id in keys(dict)
            dict[id]
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


struct StateDist
    z::Matrix{Float64}
    w::Vector{Float64}
    ids::Vector{Symbol}
    map::ImmutableDict{Symbol, UnitRange{Int}}
end

function StateDist(state::State, n::Int64)
    StateDist(
        Base.repeat(state.z', n),
        log.(1/n * ones(n)),
        state.ids,
        state.map
    )
end

function StateDist(states::AbstractArray{State})
    zs = [s.z' for s in states]
    StateDist(
        reduce(vcat, zs),
        fill(-log(length(states)), length(states)),
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

function Base.getindex(s::StateDist, q::Symbol)::Matrix{Float64}
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

function alter(state::StateDist, substitutions::Pair{Symbol, Matrix{Float64}}...)::StateDist
    
    z_buf = Zygote.Buffer(state.z)
    z_buf[:, :] = state.z

    for (id, v) in substitutions
        z_buf[:, state.map[id]] = v
    end

    StateDist(copy(z_buf), state.w, state.ids, state.map)
end

function draw(dist::StateDist; n=1, as_dist=true)
    Zygote.ignore() do
        # idxs = wdsample(1:length(dist), exp.(dist.w), n)
        # idxs = dsample(1:length(dist), n)
        # idxs = rand(1:length(dist), n)
        idxs = repeat(1:length(dist), n ÷ length(dist) + 1)[1:n]
        if as_dist
            # w = 
            # w = log.(exp.(w) ./ sum(exp.(w)))
            StateDist(dist.z[idxs, :], dist.w[idxs], dist.ids, dist.map)
        else
            map(idxs) do i
                State(dist.z[i, :], dist.ids, dist.map)
            end
        end
    end
end

function Base.copy(dist::StateDist)
    StateDist(
        copy(dist.z),
        copy(dist.w),
        dist.ids,
        dist.map
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
Zygote.@adjoint StateDist(z::Matrix{Float64}, map) = StateDist(z, map), p -> (p.z, nothing)


