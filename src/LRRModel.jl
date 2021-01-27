
module LRRModel
using DataFrames
export LRRParameters, iteratepd!, solve𝔼R!, approximatesolution, simulatemodel
export @unpack_LRRParameters

include("utils.jl")
@with_kw struct LRRParameters
    @deftype Float64
    β = 0.998
    γ = 10.0
    φ = 1.5
    θ = (1-γ)/(1-1/φ)
    ρ = 0.979
    φe = 0.044
    φd = 4.5
    ψd = 3
    μ = 0.0015
    σ̄ = 0.0078
    ν₁ = .987
    σw = 0.23e-5

    maxx = φe * σ̄/sqrt(1-ρ^2) * 1.96
    nx::Int64 = 50
    xgrid::Vector{Float64} = range(-maxx, maxx, length = nx)
    nσ::Int64 = 50
    σmax = sqrt(σw /sqrt(1-ν₁^2) * 1.96 + σ̄^2)
    σmin = sqrt(max(σ̄^2 - σw /sqrt(1-ν₁^2) * 1.96, 0.00005))
    σgrid::Vector{Float64} = range(σmin, σmax, length = nσ)
    𝔼η::IterableExpectation{Array{Float64,1},Array{Float64,1}} = expectation(Normal(), Gaussian; n=10)
    𝔼e::IterableExpectation{Array{Float64,1},Array{Float64,1}} = expectation(Normal(), Gaussian; n=10)
    𝔼w::IterableExpectation{Array{Float64,1},Array{Float64,1}} = expectation(Normal(), Gaussian; n=10)
    # 𝔼::Function = f -> 𝔼η(η->𝔼e(e->𝔼w(w->f(η, e, w))))
    # 𝔼εx::IterableExpectation{Array{Float64, 1}, Array{Float64, 1}} = expectation(Normal(), Gaussian; n = 10)
    Rfmat::Matrix{Float64} = zeros(nx, nσ)
    pdmat::Matrix{Float64} = @. 1/max(exp(-(1-1/φ) * (μ + xgrid) - 0.5 * (1-γ) * (1-1/φ) * σgrid'^2)/β-1, 1e-4)
    𝔼Rmat::Matrix{Float64} = zeros(nx, nσ)
end

function gfunc(x, σ, η, param)
    @unpack_LRRParameters param
    g = μ + x + σ * η
end

function x′func(x, σ, e, param)
    @unpack_LRRParameters param
    x′ = ρ * x + φe * σ * e
end

function σ′func(σ, w, param)
    @unpack_LRRParameters param
    σ′² = σ̄^2 + ν₁ * (σ^2 - σ̄^2) + σw * w
    return sqrt(max(σ′², 0))
end

function Rfunc(pdfunc, x, σ, η, e, w, param)
    @unpack_LRRParameters param
    x′ = x′func(x, σ, e, param)
    σ′ = σ′func(σ, w, param)
    pd′ = pdfunc(x′, σ′)
    pd = pdfunc(x, σ)
    g = gfunc(x, σ, η, param)
    return exp(g) * (pd′ + 1)/pd
end

function gdfunc(x, σ, η, param)
    @unpack_LRRParameters param
    gd = μ + ψd * x + σ * η
end

function Rdfunc(pdfunc, x, σ, η, e, w, param)
    @unpack_LRRParameters param
    x′ = x′func(x, σ, e, param)
    σ′ = σ′func(σ, w, param)
    pd′ = pdfunc(x′, σ′)
    pd = pdfunc(x, σ)
    gd = gdfunc(x, σ, η, param)
    return exp(g) * (pd′ + 1)/pd
end


function Mfunc(pdfunc, x, σ, η, e, w, param)
    @unpack_LRRParameters param
    g = gfunc(x, σ, η, param)
    R = Rfunc(pdfunc, x, σ, η, e, w, param)
    M = β^θ * exp(g)^(-θ/φ) * R^(θ-1)
end

function __pd_inner(pdfunc, x, σ, η, e, w, param)
    @unpack_LRRParameters param
    x′ = x′func(x, σ, e, param)
    σ′ = σ′func(σ, w, param)
    pd′ = pdfunc(x′, σ′)
    g = gfunc(x, σ, η, param)
    return (β * exp(g)^(1-1/φ) * (pd′ + 1))^θ
end
__𝔼w_pd_inner(pdfunc, x, σ, η, e, param) = param.𝔼w(w->__pd_inner(pdfunc, x, σ, η, e, w, param))
__𝔼ew_pd_inner(pdfunc, x, σ, η, param) = param.𝔼e(e->__𝔼w_pd_inner(pdfunc, x, σ, η, e, param))
__𝔼_pd_inner(pdfunc, x, σ, param) = param.𝔼η(η->__𝔼ew_pd_inner(pdfunc, x, σ, η, param))


function iteratepd!(newpdmat, pd′mat, param)
    @unpack_LRRParameters param
    # 𝔼 = f -> 𝔼η(η->𝔼e(e->𝔼w(w->f(η, e, w))))
    pd′func = extrapolate(interpolate((xgrid, σgrid), pd′mat, Gridded(Linear())), Interpolations.Line())
    @threads for iter in CartesianIndices(pdmat)
       ix, iσ = iter[1] , iter[2]
       x, σ = xgrid[ix], σgrid[iσ]
    #    newpdmat[iter] = 𝔼((η, e, w)->__pd_inner(pd′func, x, σ, η, e, w, param))^(1/θ)
       newpdmat[iter] = __𝔼_pd_inner(pd′func, x, σ, param)^(1/θ)
    end
end

function solve𝔼R!(param)
    @unpack_LRRParameters param
    pdfunc = extrapolate(interpolate((xgrid, σgrid), pdmat, Gridded(Linear())), Interpolations.Line())
    𝔼 = f -> 𝔼η(η->𝔼e(e->𝔼w(w->f(η, e, w))))
    for iter in CartesianIndices(pdmat)
       ix, iσ = iter[1] , iter[2]
       x, σ = xgrid[ix], σgrid[iσ]
       𝔼Rmat[iter] = 𝔼((η, e, w)->Rfunc(pdfunc, x, σ, η, e, w, param))
       Rfmat[iter] = 1/𝔼((η, e, w)->Mfunc(pdfunc, x, σ, η, e, w, param))
    end
end

function approximatesolution(param, κ = 0.997, κm = 0.9966)
    @unpack_LRRParameters param
    A1 = (1-1/φ)/(1-κ * ρ)
    A1m = (ψd - 1/φ)/(1-κm * ρ)
    A2 = 0.5 * ((θ - θ/φ)^2 + (θ * A1 * κ * φe)^2)/θ/(1-κ * ν₁)
    B = κ * A1 * φe
    σa2 = (1 + B^2) * σ̄^2 + (A2 * κ * σw)^2
    λmη = -γ
    λme = (1-θ) * B
    λmw = (1-θ) * A2 * κ
    βme = κm * A1m * φe
    Hm = (λmη^2 + (-λme + βme)^2 + φd^2)
    A2m = ((1-θ) * A2 * (1-κ * ν₁) + 0.5 * Hm) / (1-κm * ν₁)
    βmw = κ * A2m
    varrm2 = (βme^2 + φd^2) * σ̄^2 + βmw^2 * σw^2
    rem = βme * λme * σ̄^2 + βmw * λmw * σw^2 - 0.5 * varrm2
    re = -λmη * σ̄^2 + λme * B * σ̄^2 + κ * A2 * λmw * σw^2 - 0.5 * σa2
    rf = -log(β) + 1/φ * μ + (1-θ)/θ * re - 1/2/θ * ((λmη^2 + λme^2) * σ̄^2 + λmw^2 * σw^2)
    return A1, A2, re, rem, rf
end

function simulatemodel(param, T = 100000)
    @unpack_LRRParameters param
    η, e, w = rand(Normal(0, 1), T), rand(Normal(0, 1), T), rand(Normal(0, 1), T)
    xvec, σvec = zeros(T), zeros(T)
    σvec[1] = σ̄
    for i in 2:T
        σvec[i] = σ′func(σvec[i-1], w[i], param)
        xvec[i] = x′func(xvec[i-1], σvec[i], e[i], param)
    end
    dt = DataFrame(t = 1:T, x = xvec, σ = σvec)
    
    Re = (𝔼Rmat - Rfmat)
    Rf = Rfmat
    
    Rffunc = extrapolate(interpolate((xgrid, σgrid), Rf, Gridded(Linear())), Interpolations.Line())
    refunc = extrapolate(interpolate((xgrid, σgrid), log.(𝔼Rmat./Rf), Gridded(Linear())), Interpolations.Line())
    Refunc = extrapolate(interpolate((xgrid, σgrid), Re, Gridded(Linear())), Interpolations.Line())
    pdfunc = extrapolate(interpolate((xgrid, σgrid), pdmat, Gridded(Linear())), Interpolations.Line())

    dt.Rf = Rffunc.(dt.x, dt.σ)
    dt.Re = Refunc.(dt.x, dt.σ)
    dt.re = refunc.(dt.x, dt.σ)
    dt.pd = pdfunc.(dt.x, dt.σ)

    return dt
end




end