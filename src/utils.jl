function sample_gauss(mean, var)
    # Bowling approximation of inverse CDF
    v = log(1/rand() - 1) * sqrt(var) / -1.702 + mean
    ll = (-0.5*(v - mean)^2 / var) - log(sqrt(2 * pi * var))
    return v,ll
end

function dist2(a, b)
    (a-b)'*(a-b)
end