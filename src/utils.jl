using Base.Threads
using Interpolations, Parameters
using NLsolve
using Expectations
using Statistics, Distributions
using Optim
using Roots
using LinearAlgebra
using BenchmarkTools

struct PartialInterpolation{T<:AbstractInterpolation}
    itpvec::Vector{T}
end


function partialinterpolate(grid, value, intermethod = Linear(), extramethod = Interpolations.Line())
    # vector intentionally copied so different threads access different interpolator
    itpvec = [extrapolate(interpolate((grid,), value[:, i], Gridded(intermethod)), extramethod) for i in 1:size(value)[2]]
    PartialInterpolation(itpvec)
end

function (f::PartialInterpolation)(x, y)
    return f.itpvec[y](x)
end

function centraldiff(V, agrid)
    Va = similar(V)
    dagrid = diff(agrid)
    na = length(agrid)
    for i in 1:na
        i₋ = max(1, i-1) # when i == 1, it becomes forward difference
        i₊ = min(na, i+1) # when i == na, it becomes backward difference
        Δa = agrid[i₊] - agrid[i₋]
        for j in 1:size(V)[2]
            ΔV = V[i₊, j] - V[i₋, j]
            Va[i, j] = ΔV / Δa
        end               
    end
    return Va
end

function centraldiff!(Va, V, agrid)
    # Va = similar(V)
    dagrid = diff(agrid)
    na = length(agrid)
    for i in 1:na
        i₋ = max(1, i-1) # when i == 1, it becomes forward difference
        i₊ = min(na, i+1) # when i == na, it becomes backward difference
        Δa = agrid[i₊] - agrid[i₋]
        for j in 1:size(V)[2]
            ΔV = V[i₊, j] - V[i₋, j]
            Va[i, j] = ΔV / Δa
        end               
    end
    return Va
end



function sorted_interpolation(x, y, xq)
    yq = similar(xq, eltype(y))
    sorted_interpolation!(yq, x, y, xq)
end

function sorted_interpolation!(yq, x, y, xq)
    xqi = similar(xq, Int64)
    xqpi = similar(xq)
    interpolate_coord!(xqi, xqpi, x, xq)
    apply_coord!(yq, xqi, xqpi, y)
    return yq
end

@fastmath function interpolate_coord!(xqi, xqpi, x, xq)
    xi = 1
    x_low = x[1]
    x_high = x[2]
    nxq, nx = length(xq), length(x)
    @inbounds for xqi_cur in 1:nxq
        xq_cur = xq[xqi_cur]
        while xi < nx - 1
            (x_high >= xq_cur) && break
            xi += 1
            x_low = x_high
            x_high = x[xi+1]
        end
        xqpi[xqi_cur] = (x_high - xq_cur) / (x_high - x_low)
        xqi[xqi_cur] = xi
    end

end

@fastmath function apply_coord!(yq, x_i, x_pi, y)
    nq = length(x_i)
    @inbounds for iq in 1:nq
        y_low = y[x_i[iq]]
        y_high = y[x_i[iq]+1]
        yq[iq] = x_pi[iq] * y_low + (1-x_pi[iq]) * y_high
    end
end


struct IterationResults{T, N}
    zero::Array{T, N}
    f_converged::Bool
    iterations::Int64
    residual::Float64
end

function iterate_function(f!, x0; residualnorm = (x -> norm(x,Inf)), ftol = 1e-11, iterations =1000)
    residual = Inf
    iter = 1
    xnew = copy(x0)
    xold = similar(x0)
    while residual > ftol && iter < iterations
        # This order is important so x0 will be returned if it's already at the ss
        # otherwise it'll jump around, creating problems for solving SS
        xold .= xnew
        f!(xnew, xold)
        residual = residualnorm(xold - xnew)
        iter += 1
    end
    converged = residual <= ftol
    return IterationResults(xold, converged, iter, residual)
end


function discretizeproces(grid, lom, 𝔼 = expectation(Normal(), Gaussian; n = 1000), minp = 1e-8)
    n = length(grid)
    A = zeros(Float64, n, n)
    if n > 1
        gridmid = (grid[1:end-1] + grid[2:end]) / 2
    else
        gridmid = []
    end
    gridmid = [-Inf; gridmid; Inf]
    for i in 1:n
        for j in 1:n
            A[i, j] = 𝔼(ε-> gridmid[j] < lom(grid[i], ε) <= gridmid[j+1])
        end
    end
    A[A.<minp] .= 0
    A .= A ./ sum(A, dims = 2)
    return A
end


function 𝔼markov(f, A::Matrix{T}, iz) where{T}
    𝔼f = zero(T)
    @inbounds for iz′ in 1:size(A)[2]
        if A[iz, iz′] > zero(T)
            𝔼f += A[iz, iz′] * f(iz′)
        end
    end
    return 𝔼f
end