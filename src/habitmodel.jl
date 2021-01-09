
module HabitModel

include("utils.jl")
export HabitParameters, HabitHouseholds, solveR!, iteratepolicy!, iterationhelper!
export @unpack_HabitParameters, @unpack_HabitHouseholds

@with_kw struct HabitParameters
    @deftype Float64
    β = 0.93^0.25
    σy = 0.0086/2
    g = 0.022/4 # + 0.5 * σy^2
    γ = 2
    # habit parameters
    φ = 0.89^0.25
    b = 0.011
    S̄ = σy * √(γ/(1-φ-b/γ))
    s̄ = log(S̄)
    sₘₐₓ = s̄ + 1/2 * (1 - S̄^2)
    Sₘₐₓ = exp(sₘₐₓ)
    agrid::Vector{Float64} = [exp(x) - 1 for x in 0.05:0.05:5] # wealth. w in the notes.
    na::Int64 = length(agrid)
    # Sgrid::Vector{Float64} = [[1e-5 * 5; 1e-4 * 5; 1e-4 * ; 1e-3 * 5]; range(0.0, Sₘₐₓ, length = 10)[2:end-1]; Sₘₐₓ .- 1e-4 .- [0.003:-0.001:0;]; Sₘₐₓ]
    nshigh::Int64 = 10
    nslow::Int64 = 10
    slowmin = -20
    sgridhigh::Vector{Float64} =  log.(range(0.0, stop = Sₘₐₓ, length = nshigh))
    sgridlow::Vector{Float64} = range(slowmin, stop = sgridhigh[2], length = nslow)
    sgrid::Vector{Float64} = [sgridlow[1:end-1]; sgridhigh[2:end]]
    Sgrid::Vector{Float64} = exp.(sgrid)
    # sgrid::Vector{Float64} = log.(Sgrid)
    ns::Int64 = length(Sgrid)
    𝔼εy::IterableExpectation{Array{Float64,1},Array{Float64,1}} = expectation(Normal(), Gaussian; n=40)
    Rfvec::Vector{Float64} = zeros(ns)
    pdvec::Vector{Float64} = ones(ns) * 10.0
    𝔼R::Vector{Float64} = zeros(ns)
end

@inline function λfunc(s, param::HabitParameters)
    @unpack s̄, sₘₐₓ, S̄ = param
    if s > sₘₐₓ
        return zero(s)
    end
    return 1/S̄ * √(1-2(s-s̄)) - 1
end


@inline function Δsfunc(s, εy, param)
    @unpack φ, s̄, σy = param
    λ = λfunc(s, param)
    return (1-φ) * (s̄ - s) + λ * σy * εy
end

s′func(s, εy, param) = s + Δsfunc(s, εy, param)

@inline function Δcfunc(εy, param)
    @unpack σy, g = param
    return g + σy * εy
end

@inline function Rfunc(s, εy, param)
    @unpack pdvec, sgrid, Sgrid = param
    s′ = s′func(s, εy, param)
    # pd = linear_interp(sgrid, pdvec, s, true, false)
    # pd′ = linear_interp(sgrid, pdvec, s′, true, false)
    pd = linear_interp(Sgrid, pdvec, exp(s), true, false)
    pd′ = linear_interp(Sgrid, pdvec, exp(s′), true, false)
    Δc = Δcfunc(εy, param)
    return (pd′ + 1) / pd * exp(Δc)
end

@inline function Mfunc(s, εy, param)
    @unpack β, γ = param
    Δs = Δsfunc(s, εy, param)
    Δc = Δcfunc(εy, param)
    return exp(-γ * (Δs+Δc)) * β
end

function 𝔼M(s, param)
    return param.𝔼εy(εy -> Mfunc(s, εy, param))
end

@inline function assetpricing(s, εy, param)
    M = Mfunc(s, εy, param)
    R = Rfunc(s, εy, param)
    return M * R - 1
end

@inline function 𝔼assetpricing(s, param)
    return param.𝔼εy(εy -> assetpricing(s, εy, param))
end

function solvepd_helper(pdguess, param)
    param.pdvec .= pdguess
    return 𝔼assetpricing.(param.sgrid, Ref(param))
end

function pditeration_inner(Fvec, s, εy, param)
    @unpack_HabitParameters param
    M = Mfunc(s, εy, param)
    Δc = Δcfunc(εy, param)
    s′ = s′func(s, εy, param)
    S′ = exp(s′)
    # F = linear_interp(sgrid, Fvec, s′, true, false)    
    # F = linear_interp(Sgrid, Fvec, S′, true, false)
    if S′ < 0.005
        F = linear_interp(sgrid, Fvec, s′, true, false)
    else
        F = linear_interp(Sgrid, Fvec, S′, true, false)
    end
    return M * exp(Δc) * F
end

function 𝔼pditeration(Fvec, s, param)
     return param.𝔼εy(εy -> pditeration_inner(Fvec, s, εy, param))
end

function iterateF!(Fnew, Fvec, param)
    @unpack sgrid = param
    Fnew .= 𝔼pditeration.(Ref(Fvec), sgrid, Ref(param))
    return Fnew
end

function calculatepd(param; iterations = 1000)
    @unpack_HabitParameters param
    Fvec = ones(size(sgrid))
    Fnew = similar(Fvec)
    pdvec .= Fvec
    counter = 0
    while (counter < iterations) && norm(Fvec) > 1e-8
        iterateF!(Fnew, Fvec, param)
        pdvec .+= Fnew
        Fvec .= Fnew
    end
    return pdvec
end

end