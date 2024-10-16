

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

function gauss_logpdf(x, mean, var)
    -0.5 * (x-mean)^2 / var - log(2π * var)/2
end

function gauss_pdf(x, mean, var)
    exp(-0.5 *(x-mean)^2/var) / sqrt(2π*var)
end


function sample_trimmed_gauss(μ, σ2, quantile; l, u, σ_trim=5)
    new_quantile = if σ2 > σ_trim^2
        prc1 = gauss_cdf.(μ, σ2, l)
        prc2 = gauss_cdf.(μ, σ2, u)
        prc1 .+ quantile.*(prc2 .- prc1)
    else
        quantile
    end
    sample_gauss(μ, σ2, new_quantile)
end

function sample_trimmed_gauss(μ, σ2; l, u)
    sample_trimmed_gauss(μ, σ2, rand(); l, u)
end



# Very basic geometry stuff
function dist2(a, b)
    sum((a.-b).^2)
end

dist(a, b) = sqrt(dist2(a, b))

# Function to find the closest point to the first point
function min_dist(points)
    # First point
    p1 = points[1]
    
    # Initialize the minimum distance with a large number
    minimum_dist = Inf
    closest_point = nothing
    
    # Loop through the rest of the points (n-1 points)
    for i in 2:length(points)
        # Compute the distance between p1 and each remaining point
        dist = sqrt(dist2(p1, points[i]))
        
        # Update the minimum distance if a closer point is found
        if dist < minimum_dist
            minimum_dist = dist
            closest_point = points[i]
        end
    end
    
    # Return the minimum distance
    return minimum_dist
end

angdiff(a, b) = posmod((a - b + π), 2π) - π
posmod(a, n) = (a - floor.(a/n) * n)


# Some soft functions for differentiability purposes
#   Be careful - make sure using these makes sense theoretically
function softif(x, value_if_positive, value_if_negative; hardness=1)
    q = 1 / (1 + exp(-hardness * x))
    q * value_if_positive + (1-q)*value_if_negative
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
        return -0.001x
    else
        return x^4
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
    v = repeat(v, n ÷ length(v) + 1)
    w = repeat(w, n ÷ length(w) + 1)

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



@views function makechunks(X::AbstractVector, n::Integer)
    c = length(X) ÷ n
    return [X[1+c*k:(k == n-1 ? end : c*k+c)] for k = 0:n-1]
end