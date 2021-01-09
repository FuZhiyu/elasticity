
module testlucasmodel
include("../src/utils.jl")
include("../src/LucasModel.jl")
using .LucasModel: LucasParameters, LucasHouseholds, ψ_markov!, solveR!
using .LucasModel: solveθbyw̃!, θfoc, solvewbyw̃!, cfocrhs, 𝔼cfocrhs
using .LucasModel: iteratepolicy!, interpolatepolicy!, iterationhelper!
using .LucasModel: @unpack_LucasParameters, @unpack_LucasHouseholds
using .LucasModel: solveAandcwratio
using .LucasModel: Rfunc, Ygrowthεy
using LinearAlgebra, Test


#########################################
#       Simpliest Lucas
#########################################
@testset "simple lucas" begin
    p = LucasParameters(
        agrid = [exp(x)-1 for x in 0.01:0.01:5], 
        # ψgrid = [0.1:0.1:0.99;], 
        γ = 4,
        ψgrid = [1.0;], 
        σx=.0, ρx = 1.0,
        σy = 0.1,
        𝔼εy = expectation(Normal(), Gaussian; n = 100)
    )
    solveR!(p)
    hh = LucasHouseholds(p)

    @unpack_LucasParameters p
    @unpack_LucasHouseholds hh

    Avec, cwratio = solveAandcwratio(p)
    𝔼R = [𝔼εy(εy->Rfunc(iψ, iψ, Ygrowthεy(εy, p), p)) for iψ in 1:nψ]
    capitalizedΩ = (1 .- ψgrid) ./ 𝔼R
    wgrid = agrid .+ capitalizedΩ'
    cguess = wgrid .* cwratio'
    V′mat = @.  wgrid^(1-γ) / (1-γ) .* Avec'


    Vamat = centraldiff(V′mat, agrid)
    Vaapproxfunc = partialinterpolate(agrid, Vamat, Linear(), Line())
    Varealfunc(w, iψ) = Avec[iψ] * w^-γ

    # test portfolio choice subproblem

    θfocerror_real = [θfoc(1.0, x, 1, (Vafunc = Varealfunc, ), p) for x in agrid]
    @test norm(θfocerror_real, Inf) < 1e-8

    θfocerror_approx = [θfoc(1.0, x, 1, (Vafunc = Vaapproxfunc, ), p) for x in agrid]
    @test norm(θfocerror_approx, Inf) < 1e-8

    θsol_real = [find_zero(x->θfoc(x, a, 1, (Vafunc = Varealfunc, ), p), 0.5) for a in agrid]
    @test norm(θsol_real .- 1) < 1e-10

    θsol_approx = [find_zero(x->θfoc(x, a, 1, (Vafunc = Vaapproxfunc, ), p), 0.5) for a in agrid]
    @test norm(θsol_approx .- 1) < 1e-10

    solveθbyw̃!(hh.θ̃mat, Vaapproxfunc, p)
    @test norm(hh.θ̃mat .- 1) < 1e-10

    # test consumption
    solvewbyw̃!(wmat, θ̃mat, Varealfunc, p)
    c̃mat = @. wmat + 1 - ψgrid' - agrid
    @test norm(c̃mat ./ wmat .- cwratio) < 1e-10


    c̃mat ./ (wmat .+ capitalizedΩ') .- cwratio'
    # construct Va function from policy function:
    cguessfunc = partialinterpolate(agrid, cguess, Linear(), Line())
    Vafromcfunc(w, iψ) = cguessfunc(w, iψ).^-γ

    solvewbyw̃!(wmat, θ̃mat, Vafromcfunc, p)
    c̃mat = @. wmat + 1 - ψgrid' - agrid
    @test norm(c̃mat ./ wmat .- cwratio, Inf) < 1e-10

    iteratepolicy!(hh, cguess, p)
    @test norm(cmat .- cguess, Inf) < 1e-10

    wrongguess = agrid * (cwratio' .+ 0.01)
    res = fixedpoint((out, x)->iterationhelper!(out, x, hh, p), wrongguess, m = 1, ftol = 1e-9)
    @test norm(res.zero - cguess, Inf) < 1e-8

    # the representative household consumes exactly 1.0
    @test sorted_interpolation(agrid, cmat[:, 1], [p.pdvec[1] + 1;])[1] ≈ 1 atol=1e-8
end

@testset "lucas model with labor income" begin

    p = LucasParameters(
        agrid = [exp(x)-1 for x in 0.01:0.01:5], 
        # ψgrid = [0.1:0.1:0.99;], 
        ψgrid = [0.1:0.1:1.0;], 
        γ = 4,
        σy = 0.1,
        β = 0.94,
        σx=.0, ρx = 1.0,
        𝔼εy = expectation(Normal(), Gaussian; n = 50)
    )
    ψ_markov!(p)
    solveR!(p)
    hh = LucasHouseholds(p)
    @unpack_LucasParameters p
    @unpack_LucasHouseholds hh
    Avec, cwratio = solveAandcwratio(p)
    𝔼R = [𝔼εy(εy->Rfunc(iψ, iψ, Ygrowthεy(εy, p), p)) for iψ in 1:nψ]

    capitalizedΩ = (1 .- ψgrid) .* pdvec + (1 .- ψgrid)
    wgrid = agrid .+ capitalizedΩ'
    c_analytical = wgrid .* cwratio'

    cguess = wgrid .* 0.05
    res = fixedpoint((out, x)->iterationhelper!(out, x, hh, p), cguess, m = 1, ftol = 1e-9)
    @test norm(res.zero - c_analytical, Inf) ≈ 0 atol = 1e-7

    financialwealth = @. [(p.pdvec + 1) * ψgrid;]
    cfunc = partialinterpolate(agrid, cmat)
    θfunc = partialinterpolate(agrid, θmat)
    @test norm(cfunc.(financialwealth, 1:nψ) .- 1, Inf) ≈ 0 atol = 1e-8
    @test norm(θfunc.(financialwealth, 1:nψ) .- 1, Inf) ≈ 0 atol = 1e-8

    Δp = 0.001
    p.pdvec .*= 1 + Δp
    hh2 = LucasHouseholds(p)
    res2 = fixedpoint((out, x)->iterationhelper!(out, x, hh2, p), hh.cmat, m = 1, ftol = 1e-9)
    @assert res2.f_converged
    θ2func = partialinterpolate(agrid, hh2.θmat)
    θ2 = θ2func.(financialwealth, 1:nψ)

    elas = @. (1 - θ2)/Δp

    elas_analytical = @. 1/financialwealth/log(𝔼R/Rfvec)
    @test norm(elas - elas_analytical, Inf) ≈ 0 atol = 1e-1

end

end