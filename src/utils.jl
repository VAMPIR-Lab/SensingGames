

# Sampling from Gaussian distributions
#   We need a closed-ish form of the CDF so that Zygote can
#   differentiate through it (sadly Distributions.jl does not
#   fully support it yet)
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
    inverse_gauss_cdf(mean, var, rand())
end

function sample_trunc_gauss(mean, var, a, b)
    # Note that mean and variance correspond to the original
    #   distribution (before truncating) - the 
    #   moments after truncating will be different
    prc1 = gauss_cdf(mean, var, a)
    prc2 = gauss_cdf(mean, var, b)
    r = rand() * (prc2 - prc1) + prc1
    v = log(1/r - 1) * sqrt(var) / -1.702 + mean
    return v
end


# Very basic geometry stuff
function dist2(a, b)
    (a-b)'*(a-b)
end

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