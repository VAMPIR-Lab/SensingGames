

# Sampling from Gaussian distributions
#   We need a closed-ish form of the CDF so that Zygote can
#   differentiate through it (sadly Distributions.jl does not
#   fully support Zygote yet, or vice versa)
#   So I use the Bowling approximation, which is accurate to about eps=0.001:
#       I(x) = 1 / (1 + exp(-1.702(x-mean)/sqrt(var)))
function inverse_gauss_cdf(mean, var, p)
    # Bowling approximation of inverse CDF
    log(1/p - 1) * sqrt(var) / -1.702 + mean
end

function gauss_cdf(mean, var, x)
    1 / (1 + exp(-1.702(x-mean)/sqrt(var)))
end

function sample_gauss(mean, var)
    rng = SensingGames._game_rng
    inverse_gauss_cdf(mean, var, rand(rng))
end


function sample_gauss(mean, var, quantile)
    inverse_gauss_cdf(mean, var, quantile)
end

function sample_trunc_gauss(mean, var, a, b, quantile)
    # Note that mean and variance correspond to the original
    #   distribution (before truncating) - the 
    #   moments after truncating will be different
    new_quantile = Zygote.ignore() do 
        prc1 = gauss_cdf.(mean, var, a)
        prc2 = gauss_cdf.(mean, var, b)
        prc1 .+ quantile*(prc2 .- prc1)
    end
    
    sample_gauss(mean, var, new_quantile)
end

function gauss_logpdf(x, mean, var)
    # p = gauss_pdf(x, mean, var) + 0.01
    # log(p)
    -0.5 * (x-mean)^2 / var - log(2π * var)/2
end

function gauss_pdf(x, mean, var)
    exp(-0.5 *(x-mean)^2/var) / sqrt(2π*var)
end

# Very basic geometry stuff
function dist2(a, b)
    sum((a.-b).^2)
end

dist(a, b) = sqrt(dist2(a, b))

angdiff(a, b) = posmod((a - b + π), 2π) - π
posmod(a, n) = (a - floor(a/n) * n)


# Some soft functions for differentiability purposes
#   Be careful - make sure using these makes sense theoretically
function softif(x, a, b; hardness=1)
    q = 1 / (1 + exp(-hardness * x))
    q * a + (1-q)*b
end

function softclamp(x, a, b)
    if x < a
        a - log(a - x + 1)
    elseif x > b
        b + log(x - b + 1)
    else
        x
    end
end

function penalty(x)
    if x > 0
        return 0
    else
        return x^6
    end
end


# Sigmoid function
function sig(x)
    1 / (1 + exp(-x))
end


# Queue rolling for Julia vectors
#   There is probably a way to do this in base Julia (and certainly
#   with DataStructures.jl) but that's alright
function roll!(a, x, max)
    push!(a, x)
    if length(a) > max
        deleteat!(a, 1)
    end
end


# Cost ornaments: Penalties for bounds and regularizations
function cost_bound(v, lower, upper; λ=100)
    λ * sum(enumerate(v)) do (i, vi)
        penalty(upper[i] - vi) + penalty(-lower[i] + vi)
    end
end

function cost_regularize(v; α=0.1)
    α * sum(v.^2)
end


# Differentiable sampling


function wsample(items, weights)
    weights = weights ./ sum(weights)
    id = findfirst(cumsum(weights) .> rand())
    if isnothing(id)
        return rand(_game_rng, items)
    end
    items[id]
end

function wdsample(v, w, n)
    [mapreduce(vcat, 1:n) do i
        idx = wsample(1:length(v), w)
        res = v[idx]
        v = [v[begin:(idx-1)]; v[(idx+1):end]]
        w = [w[begin:(idx-1)]; w[(idx+1):end]]
        res
    end...]
end


function dsample(v, n)
    w = ones(length(v))
    wdsample(v, w, n)
end


function default(t, i, d)
    try
        t[i]
    catch
        d
    end
end


function l2(v)
    sum(v'*v)
end

function randu(rng, l, u)
    rand(rng) * (u - l) + l
end