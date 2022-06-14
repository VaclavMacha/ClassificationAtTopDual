# progress
Base.@kwdef mutable struct Progress
    t_init::Float64 = time()
    t_last::Float64 = time()
    t_min::Float64 = 60
    epoch_max::Int = 0
    iter_max::Int = 0
    verbose::Bool = true
end

function add_optionals!(io, optionals::Pair...)
    for (key, val) in optionals
        write(io, "⋅ $(key): $(val) \n")
    end
    return
end

function start!(p::Progress, args...)
    p.t_init = time()
    p.t_last = time()

    # generate log message
    io = IOBuffer()
    write(io, "Training started:  \n")
    add_optionals!(io, args...)
    print_timer(io, TO; sortby=:name)

    # print to logger
    if p.verbose
        @info String(take!(io))
    end
    return
end

function progress!(p::Progress, iter, epoch, args...)
    time() - p.t_last >= p.t_min || return

    all_iter = p.epoch_max * p.iter_max
    finished_iter = iter + (epoch - 1) * p.iter_max

    # generate log message
    io = IOBuffer()
    perc = round(Int, 100 * finished_iter / all_iter)
    write(io, "Training in progress: $(perc)% \n")
    write(io, "⋅ Epoch: $(epoch)/$(p.epoch_max) \n")
    p.iter_max == 1 || write(io, "⋅ Iteration: $(iter)/$(p.iter_max) \n")

    # duration
    p.t_last = time()
    elapsed = p.t_last - p.t_init
    per_iter = elapsed / finished_iter
    per_epoch = p.iter_max * per_iter
    eta = per_iter * (all_iter - finished_iter)

    write(io, "⋅ Elapsed time: $(durationstring(elapsed)) \n")
    write(io, "⋅ Time per epoch: $(speedstring(per_epoch)) \n")
    p.iter_max == 1 || write(io, "⋅ Time per iter: $(speedstring(per_iter)) \n")
    write(io, "⋅ ETA: $(durationstring(eta)) \n")
    add_optionals!(io, args...)
    print_timer(io, TO; sortby=:name)

    # print to logger
    if p.verbose
        @info String(take!(io))
    end
    return
end

function finish!(p::Progress, args...)
    p.t_last = time()
    elapsed = p.t_last - p.t_init

    # generate log message
    io = IOBuffer()
    write(io, "Training finished:  \n")
    write(io, "⋅ Elapsed time: $(durationstring(elapsed)) \n")
    add_optionals!(io, args...)
    print_timer(io, TO; sortby=:name)

    # print to logger
    if p.verbose
        @info String(take!(io))
    end
    return
end
