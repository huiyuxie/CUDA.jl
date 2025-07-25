"""
    @sync [blocking=false] ex

Run expression `ex` and synchronize the GPU afterwards.

The `blocking` keyword argument determines how synchronization is performed. By default,
non-blocking synchronization will be used, which gives other Julia tasks a chance to run
while waiting for the GPU to finish. This may increase latency, so for short operations,
or when benchmaring code that does not use multiple tasks, it may be beneficial to use
blocking synchronization instead by setting `blocking=true`. Blocking synchronization
can also be enabled globally by changing the `nonblocking_synchronization` preference.

See also: [`synchronize`](@ref).
"""
macro sync(ex...)
    # destructure the `@sync` expression
    code = ex[end]
    kwargs = ex[1:end-1]

    # decode keyword arguments
    blocking = false
    for kwarg in kwargs
        Meta.isexpr(kwarg, :(=)) || error("Invalid keyword argument $kwarg")
        key, val = kwarg.args
        if key == :blocking
            isa(val, Bool) ||
                error("Invalid value for keyword argument $kwarg; expected Bool, got $(val)")
            blocking = val
        else
            error("Unknown keyword argument $kwarg")
        end
    end

    quote
        local ret = $(esc(code))
        synchronize(; blocking=$blocking)
        ret
    end
end

function versioninfo(io::IO=stdout)
    @assert functional(true)

    println(io, "CUDA toolchain: ")

    print(io, "- runtime $(runtime_version().major).$(runtime_version().minor), ")
    if local_toolkit
        println(io, "local installation")
    else
        println(io, "artifact installation")
    end
    if has_nvml()
        print(io, "- driver $(NVML.driver_version())")
    else
        print(io, "- unknown driver")
    end
    println(io, " for $(driver_version().major).$(driver_version().minor)")
    println(io, "- compiler $(compiler_version().major).$(compiler_version().minor)")
    println(io)

    println(io, "CUDA libraries: ")
    for lib in (:CUBLAS, :CURAND, :CUFFT, :CUSOLVER, :CUSPARSE)
        mod = getfield(CUDA, lib)
        println(io, "- $lib: ", mod.version())
    end
    println(io, "- CUPTI: $(CUPTI.library_version()) (API $(CUPTI.version()))")
    println(io, "- NVML: ", has_nvml() ? NVML.version() : "missing")
    println(io)

    println(io, "Julia packages: ")
    println(io, "- CUDA: $(Base.pkgversion(CUDA))")
    for name in [:CUDA_Driver_jll, :CUDA_Compiler_jll, :CUDA_Runtime_jll, :CUDA_Runtime_Discovery]
        isdefined(CUDA, name) || continue
        mod = getfield(CUDA, name)
        println(io, "- $(name): $(Base.pkgversion(mod))")
    end
    println(io)

    println(io, "Toolchain:")
    println(io, "- Julia: $VERSION")
    println(io, "- LLVM: $(LLVM.version())")
    println(io)

    env = filter(var->startswith(var, "JULIA_CUDA"), keys(ENV))
    if !isempty(env)
        println(io, "Environment:")
        for var in env
            println(io, "- $var: $(ENV[var])")
        end
        println(io)
    end

    prefs = [
        "nonblocking_synchronization" => Preferences.load_preference(CUDA, "nonblocking_synchronization"),
        "default_memory" => Preferences.load_preference(CUDA, "default_memory"),
        "CUDA_Runtime_jll.version" => Preferences.load_preference(CUDA_Runtime_jll, "version"),
        "CUDA_Runtime_jll.local" => Preferences.load_preference(CUDA_Runtime_jll, "local"),
        "CUDA_Driver_jll.compat" => Preferences.load_preference(CUDA_Driver_jll, "compat"),
    ]
    if any(x->!isnothing(x[2]), prefs)
        println(io, "Preferences:")
        for (key, val) in prefs
            if !isnothing(val)
                println(io, "- $key: $val")
            end
        end
        println(io)
    end

    devs = devices()
    if isempty(devs)
        println(io, "No CUDA-capable devices.")
    elseif length(devs) == 1
        println(io, "1 device:")
    else
        println(io, length(devs), " devices:")
    end
    for (i, dev) in enumerate(devs)
        function query_nvml()
            mig = uuid(dev) != parent_uuid(dev)
            nvml_gpu = NVML.Device(parent_uuid(dev))
            nvml_dev = NVML.Device(uuid(dev); mig)

            str = NVML.name(nvml_dev)
            cap = NVML.compute_capability(nvml_gpu)
            mem = NVML.memory_info(nvml_dev)

            (; str, cap, mem)
        end

        function query_cuda()
            str = name(dev)
            cap = capability(dev)
            mem = device!(dev) do
                # this requires a device context, so we prefer NVML
                (free=free_memory(), total=total_memory())
            end
            (; str, cap, mem)
        end

        str, cap, mem = if has_nvml()
            try
                query_nvml()
            catch err
                if !isa(err, NVML.NVMLError) ||
                   !in(err.code, [NVML.ERROR_NOT_SUPPORTED, NVML.ERROR_NO_PERMISSION])
                    rethrow()
                end
                query_cuda()
            end
        else
            query_cuda()
        end
        println(io, "  $(i-1): $str (sm_$(cap.major)$(cap.minor), $(Base.format_bytes(mem.free)) / $(Base.format_bytes(mem.total)) available)")
    end
end
