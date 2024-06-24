# The State struct is a convenience to allow indexing vectors by symbols.
#   Managing big state spaces gets annoying so it's convenient to 
#   be able to say, e.g., state[:p1_pos] and know that it's player 1's position,
#   knowing that `state` is basically a 1D vector but not caring about where
#   its components actually live.
# They're also helpful because Zygote doesn't permit array mutation:
#   States are "immutable" (you actually can mutate them, but you shouldn't);
#   you can use `alter` to get a new State from an old one and a list of substitutions.

struct State
    z::Vector{Float32}
    ids::Vector{Symbol}
    map::Dict{Symbol, UnitRange{Int}}
end

function State(semantics::Pair{Symbol, Int}...)

    idx = 1

    ids = [map(semantics) do (id, _)
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

function Base.getindex(s::State, i::Int)::Vector{Float32}
    s.z[i]
end

function Base.getindex(s::State, q::Symbol)::Vector{Float32}
    s.z[s.map[q]]
end

function Base.getindex(s::State, I::Vararg{Int})::Vector{Float32}
    s.z[I...]
end

function Base.length(s::State)
    length(s.z)
end

function alter(state::State, substitutions::Pair{Symbol, Vector{Float32}}...)::State
    dict = Dict(substitutions...)
    z::Vector{Float32} = mapreduce(vcat, state.ids) do id::Symbol
        if id in keys(dict)
            Float32.(dict[id])
        else
            state[id]
        end
    end

    State(z, state.ids, state.map)
end

function alter(state::State, substitutions::Pair{Symbol, Vector{Float64}}...)::State
    dict = Dict(substitutions...)
    z::Vector{Float32} = mapreduce(vcat, state.ids) do id::Symbol
        if id in keys(dict)
            Float32.(dict[id])
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


struct StateDist
    z::Matrix{Float32}
    w::Vector{Float32}
    ids::Vector{Symbol}
    map::Dict{Symbol, UnitRange{Int}}
end

function StateDist(state::State, n::Int64)
    StateDist(
        Base.repeat(state.z', n),
        log.(1/n * ones(n)),
        state.ids,
        state.map
    )
end

function Base.size(s::StateDist)
    size(s.z)
end

function Base.getindex(s::StateDist, i::Union{Int, Colon, UnitRange{Int}})
    State(s.z[i, :], s.ids, s.map)
end

function Base.getindex(s::StateDist, q::Symbol)::Matrix{Float32}
    s.z[:, s.map[q]]
end

function Base.length(s::StateDist)
    size(s.z)[1]
end

function expectation(f, s::StateDist)
    sum(1:length(s)) do i
        f(s[i]) * exp(s.w[i])
    end
end

function alter(state::StateDist, substitutions...)::StateDist
    dict = Dict(substitutions...)
    z::Matrix{Float32} = mapreduce(hcat, state.ids) do id::Symbol
        if id in keys(dict)
            Float32.(dict[id])
        else
            state[id]
        end
    end

    StateDist(z, state.w, state.ids, state.map)
end

function reweight(state::StateDist, w)
    w_new = state.w .+ vec(w)
    w_new = w_new .- log(sum(exp.(w_new)))
    StateDist(state.z, w_new, state.ids, state.map)
end

function draw(state::StateDist)
    i = wsample(1:length(state), exp.(state.w))
    State(state.z[i, :], state.ids, state.map)
end

Zygote.@adjoint State(semantics...) = State(semantics...), p -> (nothing)
Zygote.@adjoint State(z::AbstractVector, ids, map) = State(z, ids, map), p -> (p.z, nothing, nothing)
Zygote.@adjoint StateDist(z::Matrix{Float32}, ids, map) = StateDist(z, ids, map), p -> (p.z, nothing, nothing)
