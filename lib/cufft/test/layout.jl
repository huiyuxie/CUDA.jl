@testset "strided signals" begin
    X = reshape(collect(1:8*6*4), 8, 6, 4)
    for (v, region, expected) in (
        (view(X, :, 1, 1), (1,), ((8,), 1, 1)),
        (view(X, 1:2:8, 1, 1), (1,), ((4,), 2, 1)),
        (view(X, 2, :, 1), (1,), ((6,), 8, 1)),
        (view(X, 1:4, :, 1), (1, 2), ((6, 8), 1, 1)),
        (view(X, 2:2:8, 2:5, 1), (1, 2), ((4, 4), 2, 1)),
        (view(X, 1:4, 2:5, :), (1, 2, 3), ((4, 6, 8), 1, 1)),
    )
        layout = cuFFT.cufft_layout(size(v), strides(v), region, ())
        @test (layout.embed, layout.stride, layout.dist) == expected

        # Reconstruct cuFFT's address formula and compare with actual view data.
        @test all(CartesianIndices(v)) do I
            offset = I[last(region)] - 1
            for i in 2:length(region)
                offset = offset * layout.embed[i] + I[region[end-i+1]] - 1
            end
            v[I] == first(v) + offset * layout.stride
        end
    end
end

@testset "batch distances" begin
    # The largest gap is internally batched; other batch dimensions stay external.
    sz, st, region = (4, 8, 6), (2, 16, 128), (2,)
    internal, external = cuFFT.get_batch_dims(region, sz)
    @test internal == (3,)
    @test external == (1,)
    layout = cuFFT.cufft_layout(sz, st, region, internal)
    @test (layout.embed, layout.stride, layout.dist) == ((128,), 16, 128)

    # Several internal dimensions must flatten to a single batch distance.
    layout = cuFFT.cufft_layout((4, 3, 2), (1, 8, 24), (1,), (2, 3))
    @test (layout.embed, layout.stride, layout.dist) == ((8,), 1, 8)
    @test cuFFT.cufft_layout((4, 3, 2), (1, 8, 32), (1,), (2, 3)) === nothing
end

@testset "real and half-spectrum layouts" begin
    # The same helper describes either side of R2C/C2R, in that side's element units.
    for (sz, expected) in (((8, 3, 4), ((8, 24), 1, 8)),
                           ((5, 3, 4), ((5, 15), 1, 5)))
        layout = cuFFT.cufft_layout(sz, Base.size_to_strides(1, sz...), (1, 3), (2,))
        @test (layout.embed, layout.stride, layout.dist) == expected
    end
end

@testset "layouts requiring a dense fallback" begin
    # Negative/zero strides, non-integral embeddings, and overlapping signal dims.
    @test cuFFT.cufft_layout((4, 3), (-1, 8), (1, 2), ()) === nothing
    @test cuFFT.cufft_layout((4, 3), (0, 8), (1, 2), ()) === nothing
    @test cuFFT.cufft_layout((4, 3), (2, 9), (1, 2), ()) === nothing
    @test cuFFT.cufft_layout((4, 3), (2, 4), (1, 2), ()) === nothing
end
