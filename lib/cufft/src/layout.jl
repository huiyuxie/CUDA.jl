# One side of cuFFT's advanced data layout, in C dimension order and element units.
struct CuFFTLayout{N}
    embed::NTuple{N,Int}
    stride::Int
    dist::Int
end

"""
    cufft_layout(sz, strides, region, batch_dims)

Describe an array's element strides using cuFFT's embedding, element stride and
internal batch distance. `sz` is the actual shape of this side of the transform,
including the shortened complex dimension for R2C/C2R. `region` and `batch_dims`
must contain valid dimensions in increasing order; external batches are excluded.
The caller must pad `embed[1]` to the full outermost transform size for R2C/C2R.

Return `nothing` when the layout cannot be represented, so callers can stage
through a dense buffer instead. This only checks the address layout; data-type
restrictions, base-address alignment and input preservation are handled separately.
"""
function cufft_layout(sz::NTuple{N,Int}, strides::NTuple{N,Int},
                      region::NTuple{R,Int}, batch_dims) where {N,R}
    R == 0 && return nothing
    all(>(0), sz) && all(>(0), strides) || return nothing

    # A cuFFT batch has a single distance between successive signals. Multiple
    # internal batch dimensions can be flattened only when they are contiguous.
    for i in 2:length(batch_dims)
        prev, dim = batch_dims[i-1], batch_dims[i]
        strides[dim] == strides[prev] * sz[prev] || return nothing
    end
    dist = isempty(batch_dims) ? 1 : strides[first(batch_dims)]

    # Julia's first transform dimension is cuFFT's innermost dimension. Each
    # embedding extent must be an integer and fit that side's logical dimension.
    embed = Vector{Int}(undef, R)
    embed[1] = max(dist, sz[last(region)])
    for i in 1:R-1
        inner, outer = region[i], region[i+1]
        extent, remainder = divrem(strides[outer], strides[inner])
        remainder == 0 && extent >= sz[inner] || return nothing
        embed[R-i+1] = extent
    end
    CuFFTLayout{R}(Tuple(embed), strides[first(region)], dist)
end
