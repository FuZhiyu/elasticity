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