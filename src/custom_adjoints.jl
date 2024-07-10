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

# function rrule(config::RuleConfig{>:HasReverseMode}, ::typeof(ThreadsX.map), f, xs::AbstractArray)
#     println("Threads.map rrule")
#     length_y = minimum(length, xs)
#     hobbits = ntuple(length_y) do i
#         args = getindex.(xs, i)
#         rrule_via_ad(config, f, args...)
#     end
#     y = ThreadsX.map(first, hobbits)
#     num_xs = Val(length(xs))
#     paddings = Threadsx.map(x -> ntuple(Returns(NoTangent()), (length(x) - length_y)), xs)
#     all(isempty, paddings) || @error """map(f, xs::Tuple...) does not allow mistmatched lengths!
#         But its `rrule` does; when JuliaLang/julia #42216 is fixed this warning should be removed."""
#     function map_pullback(dy_raw)
#         dy = unthunk(dy_raw)
#         # We want to call the pullbacks in `rrule_via_ad` in reverse sequence to the forward pass:
#         backevals = ntuple(length_y) do i
#             rev_i = length_y - i + 1
#             last(hobbits[rev_i])(dy[rev_i])
#         end |> reverse
#         # This df doesn't infer, could test Base.issingletontype(F), but it's not the only inference problem.
#         df = ProjectTo(f)(sum(first, backevals))
#         # Now unzip that. Because `map` like `zip` should when any `x` stops, some `dx`s may need padding.
#         # Although in fact, `map(+, (1,2), (3,4,5))` is an error... https://github.com/JuliaLang/julia/issues/42216
#         dxs = ntuple(num_xs) do k
#             dx_short = ThreadsX.map(bv -> bv[k+1], backevals)
#             ProjectTo(xs[k])((dx_short..., paddings[k]...))  # ProjectTo makes the Tangent for us
#         end
#         return (NoTangent(), df, dxs...)
#     end
#     map_back(dy::AbstractZero) = (NoTangent(), NoTangent(), ntuple(Returns(NoTangent()), num_xs)...)
#     return y, map_pullback
# end