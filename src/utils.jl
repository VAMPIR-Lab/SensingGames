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

# Note that mean and variance correspond to the original
#   distribution (before truncating)
function sample_trunc_gauss(mean, var, a, b)
    prc1 = gauss_cdf(mean, var, a)
    prc2 = gauss_cdf(mean, var, b)
    r = rand() * (prc2 - prc1) + prc1
    v = log(1/r - 1) * sqrt(var) / -1.702 + mean
    return v
end

function dist2(a, b)
    (a-b)'*(a-b)
end

angdiff(a, b) = posmod((a - b + π), 2π) - π
posmod(a, n) = (a - floor(a/n) * n)

function softif(x, a, b; hardness=1)
    q = 1 / (1 + exp(-hardness * x))
    q * a + (1-q)*b
end

function penalty(x)
    if x > 0
        return 0
    else
        return x^6
    end
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

function sig(x)
    1 / (1 + exp(-x))
end

function roll!(a, x, max)
    push!(a, x)
    if length(a) > max
        deleteat!(a, 1)
    end
end
