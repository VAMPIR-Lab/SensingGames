# This defines a derivative rule through ThreadsX.sum, which lets us
#   take a gradient through multiple threads
#   See https://discourse.julialang.org/t/parallel-reductions-with-zygote/75969/9

using ThreadsX
using ChainRules
using ChainRules: RuleConfig, HasReverseMode, rrule_via_ad, ProjectTo, NoTangent, unthunk

function ChainRules.rrule(
    config::RuleConfig{>:HasReverseMode}, ::typeof(ThreadsX.sum), f, xs::AbstractArray)
    fx_and_pullbacks = ThreadsX.map(x->rrule_via_ad(config, f, x), xs)
    y = ThreadsX.sum(first, fx_and_pullbacks)

    pullbacks = ThreadsX.map(last, fx_and_pullbacks)

    project = ProjectTo(xs)


    function sum_pullback(ȳ)
        call(f, x) = f(x)
        # if dims is :, then need only left-handed only broadcast
        # broadcast_ȳ = dims isa Colon  ? (ȳ,) : ȳ
        broadcast_ȳ = ȳ
        f̄_and_x̄s = ThreadsX.map(f->f(ȳ), pullbacks)
        # no point thunking as most of work is in f̄_and_x̄s which we need to compute for both
        f̄ = if fieldcount(typeof(f)) === 0 # Then don't need to worry about derivative wrt f
            NoTangent()
        else
            ThreadsX.sum(first, f̄_and_x̄s)
        end
        x̄s = ThreadsX.map(unthunk ∘ last, f̄_and_x̄s) # project does not support receiving InplaceableThunks
        return NoTangent(), f̄, project(x̄s)
    end
    return y, sum_pullback
end

function ChainRules.rrule(config::RuleConfig{>:HasReverseMode}, ::typeof(ThreadsX.map), f, X::AbstractArray)
    hobbits = ThreadsX.map(X) do x  # this makes an array of tuples
        y, back = rrule_via_ad(config, f, x)
    end
    Y = ThreadsX.map(first, hobbits)
    function map_pullback(dY_raw)
        dY = unthunk(dY_raw)
        backevals = ThreadsX.map(hobbits, dY) do (y, back), dy
            dx, dx = back(dy)
        end
        df = ProjectTo(f)(sum(first, backevals))
        dX = ThreadsX.map(last, backevals)
        return (NoTangent(), df, dX)
    end
    return Y, map_pullback
end
