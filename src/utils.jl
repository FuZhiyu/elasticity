using Interpolations, Parameters
using NLsolve
using Expectations
using Statistics, Distributions
using Optim
using Roots
using BenchmarkTools

struct PartialInterpolation{T<:AbstractInterpolation}
    itpvec::Vector{T}
end


function partialinterpolate(grid, value, method = Linear())
    # vector intentionally copied so different threads access different interpolator
    itpvec = [extrapolate(interpolate((grid,), value[:, i], Gridded(method)), Line()) for i in 1:size(value)[2]]
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