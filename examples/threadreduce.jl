# using ChainRules: RuleConfig, HasReverseMode, rrule_via_ad, ProjectTo, NoTangent, unthunk


score_serial(z) = sum(z) do i
    sleep(0.00001)
    i^2
end


score_parallel(z) = ThreadsX.sum(z) do i
    sleep(0.00001)
    i^2
end



@time begin
    for i in 1:100
        z = rand(10)
        Flux.gradient(score_serial, z)
    end
end

@time begin
    for i in 1:100
        z = rand(10)
        Flux.gradient(score_parallel, z)
    end
end

@time score_parallel(z)



@time 


