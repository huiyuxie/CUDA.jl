using Interpolations

@inline function calcpoint(blockIdx, blockDim, threadIdx, size)
    i = (blockIdx - 1) * blockDim + threadIdx
    return i, Float32(i)
end
function kernel_texture_warp_native(dst::CuDeviceArray{<:Any,1}, texture::CuDeviceTexture{<:Any,1})
    i, u = calcpoint(blockIdx().x, blockDim().x, threadIdx().x, size(dst)[1])
    @inbounds dst[i] = texture[u]
    return nothing
end
function kernel_texture_warp_native(dst::CuDeviceArray{<:Any,2}, texture::CuDeviceTexture{<:Any,2})
    i, u = calcpoint(blockIdx().x, blockDim().x, threadIdx().x, size(dst)[1])
    j, v = calcpoint(blockIdx().y, blockDim().y, threadIdx().y, size(dst)[2])
    @inbounds dst[i,j] = texture[u,v]
    return nothing
end
function kernel_texture_warp_native(dst::CuDeviceArray{<:Any,3}, texture::CuDeviceTexture{<:Any,3})
    i, u = calcpoint(blockIdx().x, blockDim().x, threadIdx().x, size(dst)[1])
    j, v = calcpoint(blockIdx().y, blockDim().y, threadIdx().y, size(dst)[2])
    k, w = calcpoint(blockIdx().z, blockDim().z, threadIdx().z, size(dst)[3])
    @inbounds dst[i,j,k] = texture[u,v,w]
    return nothing
end

function fetch_all(texture)
    dims = size(texture)
    d_out = CuArray{eltype(texture)}(undef, dims...)

    kernel = @cuda launch=false kernel_texture_warp_native(d_out, texture)
    config = launch_configuration(kernel.fun)

    dim_x, dim_y, dim_z = size(texture, 1), size(texture, 2), size(texture, 3)
    threads_x = min(dim_x, config.threads)
    blocks_x = cld(dim_x, threads_x)

    kernel(d_out, texture; threads=threads_x, blocks=(blocks_x, dim_y, dim_z))
    d_out
end

@testset "CuTextureArray(::CuArray)" begin
    testheight, testwidth, testdepth = 16, 16, 4
    a1D = convert(Array{Float32}, 1:testheight)
    a2D = convert(Array{Float32}, repeat(1:testheight, 1, testwidth) + repeat(0.01 * (1:testwidth)', testheight, 1))
    a3D = convert(Array{Float32}, repeat(a2D, 1, 1, testdepth))
    for k = 1:testdepth; a3D[:,:,k] .+= 0.0001 * k; end
    d_a1D = CuArray(a1D)
    d_a2D = CuArray(a2D)
    d_a3D = CuArray(a3D)

    texarr1D = CuTextureArray(d_a1D)
    tex1D = CuTexture(texarr1D)
    @test fetch_all(tex1D) == d_a1D
    array_mem = texarr1D.mem
    @test startswith(sprint(show, array_mem), "$testheight-element ArrayMemory")

    texarr2D = CuTextureArray(d_a2D)
    tex2D = CuTexture(texarr2D)
    @test fetch_all(tex2D) == d_a2D

    texarr3D = CuTextureArray(d_a3D)
    tex3D = CuTexture(texarr3D)
    @test fetch_all(tex3D) == d_a3D

    # test the underlying ArrayMemory too
    array_mem = texarr3D.mem
    @test_throws ErrorException("Opaque array memory does not have a definite size") sizeof(array_mem)
    @test size(array_mem) == size(d_a3D)
    @test length(array_mem) == length(d_a3D)
    @test ndims(array_mem) == ndims(d_a3D)

    # and its CuArrayPtr
    ptr = convert(CUDA.CuArrayPtr{Float32}, array_mem)
    @test isequal(ptr, ptr)
    @test isless(ptr, 2 + ptr)
    @test isless(ptr - 2, ptr)
    @test ptr == ptr
    @test ptr - 2 < ptr
    @test eltype(ptr) == Float32
    @test convert(UInt64, ptr) == UInt64(UInt(ptr)) 
    @test_throws ArgumentError("cannot convert a GPU array pointer to a CPU pointer") convert(Ptr{Float32}, ptr)
end

@testset "CuTextureArray(::Array)" begin
    testheight, testwidth, testdepth = 16, 16, 4
    a1D = convert(Array{Float32}, 1:testheight)
    a2D = convert(Array{Float32}, repeat(1:testheight, 1, testwidth) + repeat(0.01 * (1:testwidth)', testheight, 1))
    a3D = convert(Array{Float32}, repeat(a2D, 1, 1, testdepth))
    for k = 1:testdepth; a3D[:,:,k] .+= 0.0001 * k; end

    texarr1D = CuTextureArray(a1D)
    copyto!(texarr1D, a1D)
    tex1D = CuTexture(texarr1D)
    @test Array(fetch_all(tex1D)) == a1D
    @test sizeof(texarr1D) == sizeof(a1D)
    @test eltype(texarr1D) == Float32
    h_arr_1D = zeros(Float32, length(a1D))
    Base.unsafe_copyto!(pointer(h_arr_1D), pointer(texarr1D), length(h_arr_1D))
    @test h_arr_1D == a1D
    cu_arr_1D = CUDA.zeros(Float32, length(a1D))
    Base.unsafe_copyto!(pointer(cu_arr_1D), pointer(texarr1D), length(cu_arr_1D))
    @test Array(cu_arr_1D) == a1D

    texarr2D = CuTextureArray(a2D)
    tex2D = CuTexture(texarr2D)
    @test Array(fetch_all(tex2D)) == a2D
    texarr2D_2 = CuTextureArray(texarr2D)
    tex2D_2 = CuTexture(texarr2D_2)
    @test Array(fetch_all(tex2D_2)) == a2D
    texarr2D_3 = CuTextureArray{Float32, 2}(texarr2D)
    tex2D_2 = CuTexture(texarr2D_3)
    @test Array(fetch_all(tex2D_2)) == a2D
    copyto!(texarr2D_3, texarr2D_2)
    tex2D_2 = CuTexture(texarr2D_3)
    @test Array(fetch_all(tex2D_2)) == a2D
    @test sizeof(texarr2D) == sizeof(a2D)
    @test eltype(texarr2D) == Float32

    tex2D_dir = CuTexture(CuTextureArray(a2D))
    @test Array(fetch_all(tex2D_dir)) == a2D

    texarr3D = CuTextureArray(a3D)
    tex3D = CuTexture(texarr3D)
    @test Array(fetch_all(tex3D)) == a3D
    texarr3D_2 = CuTextureArray(texarr3D)
    tex3D_2 = CuTexture(texarr3D_2)
    @test Array(fetch_all(tex3D_2)) == a3D
    texarr3D_3 = CuTextureArray{Float32, 3}(texarr3D)
    copyto!(texarr2D_3, texarr2D_2)
    tex3D_3 = CuTexture(texarr3D_3)
    @test Array(fetch_all(tex3D_3)) == a3D
    @test sizeof(texarr3D) == sizeof(a3D)
    @test eltype(texarr3D) == Float32
end

@testset "CuTexture(::CuArray)" begin
    testheight, testwidth, testdepth = 16, 16, 4
    a1D = convert(Array{Float32}, 1:testheight)
    a2D = convert(Array{Float32}, repeat(1:testheight, 1, testwidth) + repeat(0.01 * (1:testwidth)', testheight, 1))
    d_a1D = CuArray(a1D)
    d_a2D = CuArray(a2D)

    # NOTE: tex1D is not supported for linear memory

    # This works as long as d_a2D is well pitched
    texwrap2D = CuTexture(d_a2D)
    @test fetch_all(texwrap2D) == d_a2D
end

@testset "type support" begin
    for T in (Int32, UInt32, Int16, UInt16, Int8, UInt8, Float32, Float16)
        testheight, testwidth, testdepth = 32, 32, 4
        a2D = rand(T, testheight, testwidth)
        d_a2D = CuArray(a2D)

        # Using CuTextureArray
        tex_2D = CuTexture(d_a2D)
        @test fetch_all(tex_2D) == d_a2D

        # Wrapping CuArray
        # This works as long as d_a2D is well pitched
        texwrap_2D = CuTexture(d_a2D)
        @test fetch_all(texwrap_2D) == d_a2D
    end
end

@testset "multiple channels" begin
    testheight, testwidth, testdepth = 16, 16, 4
    a2D = [(Int32(i), Int32(j)) for i = 1:testheight, j = 1:testwidth]
    d_a2D = CuArray(a2D)
    texarr2D = CuTextureArray(d_a2D)
    tex2D = CuTexture(texarr2D)
    @test fetch_all(tex2D) == d_a2D

    testheight, testwidth, testdepth = 16, 16, 4
    a2D = [(Int16(i), Int16(j), Int16(i + j), Int16(i - j)) for i = 1:testheight, j = 1:testwidth]
    d_a2D = CuArray(a2D)
    texarr2D = CuTextureArray(d_a2D)
    tex2D = CuTexture(texarr2D)
    @test fetch_all(tex2D) == d_a2D
end

@testset "interpolations" begin
    @testset "$interpolate $T" for T in (Float16, Float32,)
        @testset "$(N)D" for N in 1:3
            cpu_src = rand(T, fill(10, N)...)
            cpu_idx = [tuple(rand(1:0.1:10, N)...) for _ in 1:10]

            gpu_src = CuTextureArray(CuArray(cpu_src))
            gpu_idx = CuArray(cpu_idx)

            @testset "nearest neighbour" begin
                cpu_dst = similar(cpu_src, size(cpu_idx))
                cpu_int = interpolate(cpu_src, BSpline(Constant()))
                broadcast!(cpu_dst, cpu_idx, Ref(cpu_int)) do idx, int
                    int(idx...)
                end

                gpu_dst = CuArray{T}(undef, size(cpu_idx))
                gpu_tex = CuTexture(gpu_src; interpolation=CUDA.NearestNeighbour())
                broadcast!(gpu_dst, gpu_idx, Ref(gpu_tex)) do idx, tex
                    tex[idx...]
                end

                @test cpu_dst ≈ Array(gpu_dst)
            end

            @testset "linear interpolation" begin
                cpu_dst = similar(cpu_src, size(cpu_idx))
                cpu_int = interpolate(cpu_src, BSpline(Linear()))
                broadcast!(cpu_dst, cpu_idx, Ref(cpu_int)) do idx, int
                    int(idx...)
                end

                gpu_dst = CuArray{T}(undef, size(cpu_idx))
                gpu_tex = CuTexture(gpu_src; interpolation=CUDA.LinearInterpolation())
                broadcast!(gpu_dst, gpu_idx, Ref(gpu_tex)) do idx, tex
                    tex[idx...]
                end

                @test cpu_dst ≈ Array(gpu_dst) rtol=0.01
            end

            N<3 && @testset "cubic interpolation" begin
                cpu_dst = similar(cpu_src, size(cpu_idx))
                cpu_int = interpolate(cpu_src, BSpline(Cubic(Line(OnGrid()))))
                broadcast!(cpu_dst, cpu_idx, Ref(cpu_int)) do idx, int
                    int(idx...)
                end

                gpu_dst = CuArray{T}(undef, size(cpu_idx))
                gpu_tex = CuTexture(gpu_src; interpolation=CUDA.CubicInterpolation())
                broadcast!(gpu_dst, gpu_idx, Ref(gpu_tex)) do idx, tex
                    tex[idx...]
                end

                # FIXME: these results, although they look OK in an image,
                #        do not match the output from Interpolations.jl
                @test_skip cpu_dst ≈ Array(gpu_dst)
            end
        end
    end
end
