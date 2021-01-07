
module TestUtil
include("../src/utils.jl")
using LinearAlgebra, Test

agrid = exp.([0.001:0.001:1;])
ygrid = [0.1:0.1:100;]
V = agrid.^2 .* ygrid'

Vaapprox = centraldiff(V, agrid)
Vareal = 2 * agrid .* ygrid'
error = Vaapprox - Vareal
@test norm(error[2:end-1, :]./Vareal[2:end-1, :]) < 1e-3

@test all(Vaapprox[1, :] .== (V[2, :] - V[1, :]) ./ (agrid[2] - agrid[1]))
@test all(Vaapprox[end, :] .== (V[end, :] - V[end-1, :]) ./ (agrid[end] - agrid[end-1]))



@time v1 = sorted_interpolation(agrid, V[:, end], [1.00:0.001:3;])
itp = extrapolate(interpolate((agrid, ), V[:, end], Gridded(Linear())), Line())
@time v2 = itp.([1.00:0.001:3;])

@test norm(v2 - v1) < 1e-8


end