# These [X]ParticleBeliefs are different ways of updating
#   our particles based (or not) on a player's real world observations.
# On one end is `JointParticleBelief`, where we don't use the real observations
#   at all. We just evolve all the particles according to the optimal policies.
#   This works really well in the "shared brain" model, where we solve the Nash
#   equilibrium problem once for all players, because it means that players by definition
#   agree where the equilibrium is and never disagree on the optimal policies.
# On the other end is `ConditionalParticleBelief`, which weights all
#   our particles by our real observations. This gives a very tight
#   distribution of particles, but we can't really use this in "shared brain" because we
#   can't use everyone's real observations. In "split brain" (maybe "separate brains" is a better term),
#   where everyone solves their own version of the Nash equilibrium problem, we CAN use our real observations,
#   but it can lead to our belief disagreeing with others', and we might disagree on where the equilibria are, which
#   is suboptimal for everyone. 
# HybridParticleBelief essentially forms a spectrum between these
#   two, where γ=1.0 is "all particles are conditioned on real observations" and γ=0.0 is "no particles
#   are conditioned."
# LateParticleBelief conditions all the particles, but only several timesteps after the real observations
#   have been received. This actually doesn't work very well as far as I can tell, but I'd like to give it
#   another try very well.

mutable struct LateParticleBelief
    game::ContinuousGame
    prior_fn
    delay::Int
    past_params::AbstractArray
    past_dist::StateDist
    past_ego_states::AbstractArray
    t_world::Int
end

function LateParticleBelief(prior_fn, delay, game)
    LateParticleBelief(
        game,
        prior_fn,
        delay,
        [],
        prior_fn(),
        [],
        0
    )
end

function draw(d::LateParticleBelief; n)
    old_dist = (d.t_world > d.delay) ? d.past_dist : d.prior_fn()
    dist = draw(old_dist; n)
    for params in d.past_params
        dist = step(d.game, dist, params)
    end
    dist
end

function update!(d::LateParticleBelief, ego_state, true_params)
    d.t_world += 1

    if d.t_world > d.delay
        d.past_dist = step(d.game, d.past_dist, d.past_ego_states[1], d.past_params[1])
    end

    roll!(d.past_params, true_params, d.delay)
    roll!(d.past_ego_states, ego_state, d.delay)
end



mutable struct ConditionedParticleBelief
    game::ContinuousGame
    dist::StateDist
end

function draw(belief::ConditionedParticleBelief; n)
    draw(belief.dist; n)
end

function update!(belief::ConditionedParticleBelief, ego_state, params)
    belief.dist = step(belief.game, belief.dist, ego_state, params)
end



mutable struct JointParticleBelief
    game::ContinuousGame
    dist::StateDist
end

function draw(belief::JointParticleBelief; n)
    Zygote.ignore() do
        draw(belief.dist; n)
    end
end

function update!(belief::JointParticleBelief, ego_state, params::NamedTuple)
    # In this version we don't even care about the ego state
    # This is "technically correct" in that as soon as we use our truego
    #    state in our belief, we diverge from the opponent
    belief.dist = step(belief.game, belief.dist, params)
end


function update!(belief::JointParticleBelief, params::NamedTuple)
    belief.dist = step(belief.game, belief.dist, params)
end

function multi_update!(belief::JointParticleBelief, params::AbstractArray)
    chunks = split(belief.dist, length(params))
    belief.dist = stack(map(enumerate(chunks)) do (i, chunk)
        step(belief.game, chunk, params[i])
    end)
end

mutable struct HybridParticleBelief
    game::ContinuousGame
    dist::StateDist
    γ::Float64
end

function draw(belief::HybridParticleBelief; n)
    Zygote.ignore() do
        draw(belief.dist; n)
    end
end

function update!(belief::HybridParticleBelief, ego_state, params)
    
    N = length(belief.dist)
    γ = belief.γ
    c = Int(floor(γ*N))
    c = clamp(c, 1, N-1)
    j = (N - c)

    joint_dist = step(belief.game, belief.dist, params)
    joint_dist = draw(joint_dist; n=j)

    cond_dist = step(belief.game, belief.dist, ego_state, params; normalize=true)
    cond_dist = draw(cond_dist; n=c, reweight=true)

    Z = [joint_dist.z; cond_dist.z]
    w = [joint_dist.w .+ log(1 - γ); cond_dist.w .+ log(γ)]
    # w = log.(exp.(w) ./ sum(exp.(w))) 

    belief.dist = StateDist(Z, w, joint_dist.ids, joint_dist.map)
end


function multi_update!(belief::HybridParticleBelief, ego_state, params::AbstractArray)
    chunks = split(belief.dist, length(params))
    belief.dist = stack(map(enumerate(chunks)) do (i, chunk)
        N = length(chunk)
        γ = belief.γ
        c = Int(floor(γ*N))
        c = clamp(c, 1, N-1)
        j = (N - c)
    
        joint_dist = step(belief.game, chunk, params[i])
        joint_dist = draw(joint_dist; n=j, reweight=true)
    
        cond_dist = step(belief.game, chunk, ego_state, params[i]; normalize=true)
        cond_dist = draw(cond_dist; n=c, reweight=true)
    
        Z = [joint_dist.z; cond_dist.z]
        w = [joint_dist.w .+ log(1 - γ); cond_dist.w .+ log(γ)]
    
        StateDist(Z, w, joint_dist.ids, joint_dist.map)
    end)
end