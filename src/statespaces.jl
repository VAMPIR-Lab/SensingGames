
struct State
    z::AbstractVector{Float64}
    ids::AbstractVector{Symbol}
    map::Dict{Symbol, UnitRange{Int}}
end

function State(semantics::Pair{Symbol, Int}...)

    idx = 1

    ids = [map(semantics) do (id, len)
        id
    end...]

    lens = Dict(semantics...)

    m = Dict(map(ids) do (id)
        l = idx
        u = (idx += lens[id])
        id => l:(u-1)
    end...)


    State(
        zeros(idx-1),
        ids,
        m
    )
end

function Base.size(s::State)
    Base.size(s.z)
end

function Base.getindex(s::State, i::Int)
    s.z[i]
end

function Base.getindex(s::State, q::Symbol)
    s.z[s.map[q]]
end

function Base.getindex(s::State, I::Vararg{Int})
    s.z[I...]
end

function Base.length(s::State)
    length(s.z)
end

function alter(state::State, substitutions...)
    dict = Dict(substitutions...)
    z = mapreduce(vcat, state.ids) do id 
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

    n = 0
    map = Dict(mapreduce(vcat, states) do s
        t = n
        n += length(s.z)
        [id => loc.+t for (id, loc) in s.map]
    end...)

    ids = mapreduce(s -> s.ids, vcat, states)
    State(z, ids, map)
end

Zygote.@adjoint State(semantics...) = State(semantics...), p -> (nothing)
Zygote.@adjoint State(z::AbstractVector, ids, map) = State(z, ids, map), p -> (p.z, nothing, nothing)
