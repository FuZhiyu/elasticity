
module TestUtil
include("../src/utils.jl")
using LinearAlgebra, Test

agrid = exp.([0.001:0.001:1;])
ygrid = [0.1:0.1:100;]
V = agrid.^2 .* ygrid'

####### central diff ########

Vaapprox = centraldiff(V, agrid)
Vareal = 2 * agrid .* ygrid'
error = Vaapprox - Vareal
@test norm(error[2:end-1, :]./Vareal[2:end-1, :]) < 1e-3

@test all(Vaapprox[1, :] .== (V[2, :] - V[1, :]) ./ (agrid[2] - agrid[1]))
@test all(Vaapprox[end, :] .== (V[end, :] - V[end-1, :]) ./ (agrid[end] - agrid[end-1]))

####### test sorted_interpolation ########

@time v1 = sorted_interpolation(agrid, V[:, end], [1.00:0.001:3;])
itp = extrapolate(interpolate((agrid, ), V[:, end], Gridded(Linear())), Line())
@time v2 = itp.([1.00:0.001:3;])

@test norm(v2 - v1) < 1e-8

####### test Markov ########
grid = [-5:1.0:5; ]
lomiid = (x, ε)-> ε
A = discretizeproces(grid, lomiid, expectation(Normal(), Gaussian, n = 100000))
@test all(diff(A, dims = 1) .== 0.0)
gridmid = [-Inf; (grid[1:end-1] + grid[2:end])/2; Inf]
@test norm(diff(cdf.(Normal(), gridmid)) - A[1, :], Inf) < 0.01
lomnorisk = (x, ε)-> x
A = discretizeproces(grid, lomnorisk, expectation(Normal(), Gaussian, n = 1000))
@test all(A - I .== 0.0)

lommartingale = (x, ε) -> x + ε
A = discretizeproces(grid, lommartingale, expectation(Normal(), Gaussian, n = 100000))
@test norm(diff(cdf.(Normal(), gridmid' .- grid), dims = 2) - A) < 0.01

# test 𝔼markov
f = i -> grid[i]^2
fvec = grid .^ 2
@test norm(A * fvec .- [𝔼markov(f, A, i) for i in 1:length(grid)]) == 0.0



end