

struct FluxPolicy <: Policy
    model
    flux_setup
end

function FluxPolicy(layers; optimizer=Flux.Adam(0.001))
    model = Flux.Chain(layers...)
    FluxPolicy(
        model,
        Flux.setup(optimizer, model) 
    )
end

function (policy::FluxPolicy)(obs)

    t = size(obs)[ndims(obs)]
    # act_in = selectdim(acts, ndims(acts), t)
    ob_in = selectdim(obs, ndims(obs), t)
    input = Float32.(ob_in)
    output = policy.model(input)
    output, 0.0
end

function gradient_step!(policy::FluxPolicy, game::SensingGame; n=1)
    loss_fn = pol -> tree_evaluate(game, pol; n)

    loss, grads = Flux.withgradient(loss_fn, policy) 
    # println(grads)
    Flux.update!(policy.flux_setup, policy.model, grads[1].model)
    loss
end

function reset!(policy::FluxPolicy)
    Flux.reset!(policy.model)
end

# ===================
# Random policy: Picks a uniform random action on every query

struct RandomPolicy <: Policy
    action_space
end

function (p::RandomPolicy)(obs::Matrix{Float64})
    rand(p.action_space...), 0.0
end

# ===================
# Linear policy

struct LinearPolicy <: Policy
    models
    in
    out
    flux_setups
end

function LinearPolicy(in, out, T; optimizer=Flux.Adam(0.5))
    models = [Dense(in => out) for t in 1:T]
    # models = [Dense(in*t => out) for t in 1:T]
    LinearPolicy(
        models, in, out,
        [Flux.setup(optimizer, model) for model in models]
    )
end

function (policy::LinearPolicy)(obs)
    T = size(obs)[ndims(obs)]

    act = mapreduce(+, 1:T) do t
        model = policy.models[t]
        input = Float32.(selectdim(obs, ndims(obs), t))
        model(input)
    end
    
    # model = policy.models[T]
    # input = Float32.(obs)
    # act = model([input...])
    0.01*act, 0.0
end

function gradient_step!(policy::LinearPolicy, game::SensingGame; n=3)
    loss_fn = pol -> tree_evaluate(game, pol; n)
    loss, grads = Flux.withgradient(loss_fn, policy) 

    for (i, m) in enumerate(policy.models)
        Flux.update!(policy.flux_setups[i], m, grads[1].models[i])
    end
    loss
end

function reset!(policy::Policy)
    # Do nothing by default
end


# ===================
# Linear to arbitrary policy

struct EmbedPolicy <: Policy
    top
    bottom
end

function EmbedPolicy(in, k, T, layers)
    EmbedPolicy(
        LinearPolicy(in, k, T),
        FluxPolicy(layers)
    )
end

function (policy::EmbedPolicy)(obs)
    e, _ = policy.top(obs)
    e = reshape(e, size(e)..., 1)
    policy.bottom(e)
end

function gradient_step!(policy::EmbedPolicy, game::SensingGame; n=1)
    loss_fn = pol -> tree_evaluate(game, pol; n)
    loss, grads = Flux.withgradient(loss_fn, policy) 
    # H = Flux.hessian(loss_fn, policy)
    
    # Update bottom
    Flux.update!(policy.bottom.flux_setup, policy.bottom.model, grads[1].bottom.model)

    # Update top
    for (i, m) in enumerate(policy.top.models)
        Flux.update!(policy.top.flux_setups[i], m, grads[1].top.models[i])
    end
    loss
end

function reset!(policy::EmbedPolicy)
    reset!(policy.bottom)
    reset!(policy.top)
end
