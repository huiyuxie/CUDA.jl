# wrappers of low-level functionality

function cufftGetProperty(property::libraryPropertyType)
  value_ref = Ref{Cint}()
  cufftGetProperty(property, value_ref)
  value_ref[]
end

version() = VersionNumber(cufftGetProperty(CUDACore.MAJOR_VERSION),
                          cufftGetProperty(CUDACore.MINOR_VERSION),
                          cufftGetProperty(CUDACore.PATCH_LEVEL))

"""
    cufftMakePlan(output_type::Type{<:cufftNumber}, input_type::Type{<:cufftNumber}, input_size::Dims, region)

low level interface to the CUDA library CuFFT for the function cufftXtMakePlanMany

# Parameters:
* `output_type`: type of the output array
* `input_type`: type of the input array
* `input_size`: size of the array to transform in units of the type
* `region`: dimensions of the array to transform
"""
function cufftMakePlan(output_type::Type{<:cufftNumber}, input_type::Type{<:cufftNumber}, input_size::Dims, region)
    if any(diff(collect(region)) .< 1)
        throw(ArgumentError("FFT region dimensions must be in strictly increasing order; got $region"))
    end
    if any(region .< 1 .|| region .> length(input_size))
        throw(ArgumentError("transformed dims can only refer to valid dimensions"))
    end
    # cuFFT half-precision transforms require all transform dim sizes to be
    # powers of 2 (NVIDIA cuFFT docs, "Half-precision transforms"). Catch this
    # up-front so the user sees a Julia error rather than a bare cuFFT error
    # from inside the wrapper.
    if input_type <: Union{Float16, Complex{Float16}} ||
       output_type <: Union{Float16, Complex{Float16}}
        for d in region
            n = input_size[d]
            ispow2(n) || throw(ArgumentError(
                "cuFFT half-precision transforms require all transform " *
                "dimensions to be powers of 2; got region $region with " *
                "input size $input_size (dim $d has size $n)"))
        end
    end
    nrank = length(region)

    sz = ntuple((d) -> input_size[region[d]], nrank)

    # all sizes which are not part of the dimensions specified by region are batch dimensions.
    num_internal_batches = prod(input_size) ÷ prod(sz)
    cdims = ntuple((d) -> (d==region[1]) ? div(input_size[d], 2) + 1 : input_size[d], length(input_size))

    # make the plan
    worksize_ref = Ref{Csize_t}()
    rsz = length(sz) > 1 ? reverse(sz) : sz
    if nrank > 3
        throw(ArgumentError("only up to three transform dimensions are allowed in one plan"))
    end

    # Plans still execute on dense arrays (including staged views). Use the same
    # stride-to-layout conversion that future direct view plans can build on.
    # Keep the simple-layout path below: NULL embeddings ignore advanced strides.
    if region !== ntuple(identity, nrank)
        internal_batch_dims, _ = get_batch_dims(region, input_size)
        xdims = output_type <: Real ? cdims : input_size
        ydims = input_type <: Real ? cdims : input_size
        input_layout = cufft_layout(xdims, Base.size_to_strides(1, xdims...), region, internal_batch_dims)
        output_layout = cufft_layout(ydims, Base.size_to_strides(1, ydims...), region, internal_batch_dims)
        @assert input_layout !== nothing && output_layout !== nothing
        inembed = Clonglong[input_layout.embed...]
        onembed = Clonglong[output_layout.embed...]
        # The outermost embedding does not affect addresses, but must cover the
        # full transform length even on the shortened complex side of R2C/C2R.
        inembed[1] = max(inembed[1], rsz[1])
        onembed[1] = max(onembed[1], rsz[1])
        num_internal_batches = prod(input_size[collect(internal_batch_dims)])
    end

    # initialize the plan handle
    handle_ref = Ref{cufftHandle}()
    cufftCreate(handle_ref)
    handle = handle_ref[]

    if region === ntuple(identity, nrank)
        # handle simple case, transforming the first nrank dimensions, ... simply! (for robustness)
        # arguments are: plan, rank, transform-sizes, inembed, istride, idist, itype, onembed, ostride, odist, otype, batch
        execution_type = promote_type(input_type, output_type)
        cufftXtMakePlanMany(handle, nrank, Clonglong[rsz...],
                            C_NULL, 1, 1, convert(cudaDataType, input_type),
                            C_NULL, 1, 1, convert(cudaDataType, output_type),
                            num_internal_batches, worksize_ref, convert(cudaDataType, execution_type))
    else
        # External batches are handled when applying the plan; cuFFT only sees
        # the transform dimensions and the selected internal batch dimensions.
        execution_type = promote_type(input_type, output_type)
        cufftXtMakePlanMany(handle, nrank, Clonglong[rsz...],
                            inembed, input_layout.stride, input_layout.dist, convert(cudaDataType, input_type),
                            onembed, output_layout.stride, output_layout.dist, convert(cudaDataType, output_type),
                            num_internal_batches, worksize_ref, convert(cudaDataType, execution_type))
    end

    handle, worksize_ref[]
end


## plan cache

# The cache key includes the full input size (not just the transform/internal-
# batch dims), so plans that differ only in *external* batch dim sizes are not
# shared. For typical workloads the cache stays small; if a caller iterates
# over many distinct batch dim sizes, consider keying on a normalised shape
# instead.
const cufftHandleCacheKey = Tuple{CuContext, Type, Type, Dims, Any}
function handle_ctor((ctx, args...))
    context!(ctx) do
        # make the plan
        handle, worksize = cufftMakePlan(args...)

        # NOTE: we currently do not use the worksize to allocate our own workarea,
        #       instead relying on the automatic allocation strategy.
        handle
    end
end
function handle_dtor((ctx, args...), handle)
    context!(ctx) do
        cufftDestroy(handle)
    end
end
const idle_handles = HandleCache{cufftHandleCacheKey, cufftHandle}(handle_ctor, handle_dtor)

function cufftGetPlan(args...)
    ctx = context()
    handle = pop!(idle_handles, (ctx, args...))

    cufftSetStream(handle, stream())

    return handle
end
function cufftReleasePlan(plan)
    push!(idle_handles, plan)
end
