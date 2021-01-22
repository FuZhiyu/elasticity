
module HabitModel

include("utils.jl")
export HabitParameters, HabitHouseholds, solveR!, iteratepolicy!, iterationhelper!
export calculatepd!, 𝔼M
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
    ψ = 0.2
    agrid::Vector{Float64} = 1 .+ [exp(x) - 1 for x in 0.0:0.01:5] # starting from 1.0 
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

#===================================================
#       Aggregate functions
===================================================#

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

@inline function Δyfunc(εy, param)
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
    Δy = Δyfunc(εy, param)
    return (pd′ + 1) / pd * exp(Δy)
end

@inline function Mfunc(s, εy, param)
    @unpack β, γ = param
    Δs = Δsfunc(s, εy, param)
    Δy = Δyfunc(εy, param)
    return exp(-γ * (Δs+Δy)) * β
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
    Δy = Δyfunc(εy, param)
    s′ = s′func(s, εy, param)
    S′ = exp(s′)
    # F = linear_interp(sgrid, Fvec, s′, true, false)    
    # F = linear_interp(Sgrid, Fvec, S′, true, false)
    if S′ < 0.005
        F = linear_interp(sgrid, Fvec, s′, true, false)
    else
        F = linear_interp(Sgrid, Fvec, S′, true, false)
    end
    return M * exp(Δy) * F
end

function 𝔼pditeration(Fvec, s, param)
     return param.𝔼εy(εy -> pditeration_inner(Fvec, s, εy, param))
end

function iterateF!(Fnew, Fvec, param)
    @unpack sgrid = param
    Fnew .= 𝔼pditeration.(Ref(Fvec), sgrid, Ref(param))
    return Fnew
end

function calculatepd!(param; iterations = 1000)
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
    for is in 1:ns
        𝔼R[is] = 𝔼εy(εy-> Rfunc(sgrid[is], εy, param))
    end
    Rfvec .= 1 ./ 𝔼M.(sgrid, Ref(param))
    return pdvec
end

#===================================================
#       Bellman equation infrastructure
===================================================#

@with_kw struct HabitHouseholds{T <: Real}
    # the first dimension for the matricies is w, and the second is the state variable z
    Vmat::Matrix{T} # value function
    # V′mat::Matrix{T} # value function
    Vamat::Matrix{T} # value function
    θmat::Matrix{T} # portfolio allocation
    θ̃mat::Matrix{T} # portfolio allocation by w̃
    cmat::Matrix{T} # consumption
    wmat::Matrix{T} # used for endogenous points
    w̃mat::Matrix{T} # used for endogenous points
    # θ̃mat::Matrix{T} # store temporary value
end

function HabitHouseholds(param, T=Float64; initializeHH=true)
    @unpack_HabitParameters param
    mats =  [Matrix{T}(undef, na, ns) for i in 1:length(fieldnames(HabitHouseholds))]
    hh = HabitHouseholds(mats...)
    if initializeHH
        initializeHH!(hh, param)
    end
    return hh
end

@inline @fastmath function θfocεy(θ, w̃, is, εy, funcs, param::HabitParameters)
    @unpack_HabitParameters param
    @unpack Vafunc = funcs
    s = sgrid[is]
    s′ = s′func(s, εy, param)
    Ygrowth = exp(Δyfunc(εy, param))
    R = Rfunc(s, εy, param)
    Rf = Rfvec[is]
    w′ = w̃ * (θ * R + (1 - θ) * Rf) / Ygrowth
    Va = Vafunc(w′, s′)
    return Va * Ygrowth^-γ  *  (R - Rf) * w̃^γ
end

function θfoc(θ, w̃, is, funcs, param::HabitParameters)
    param.𝔼εy(εy -> θfocεy(θ, w̃, is, εy, funcs, param))
end

function solveθbyw̃!(θ̃mat, Vafunc, param)
    @unpack_HabitParameters param
    # Vafunc = partialinterpolate(agrid, Vamat, Linear())
    for iter in CartesianIndices(θ̃mat)
        ia, is = iter[1], iter[2]
        w̃ = agrid[ia]
        try
            θ̃mat[iter] = find_zero(x -> θfoc(x, w̃, is, (Vafunc = Vafunc,), param), (0.0, 30.0), Roots.A42(), tol=1e-5)
        catch
            @show ia, is
            throw(InterruptException())
        end
        # θ̃mat[iter] = find_zero(x -> θfoc(x, w̃, is, (Vafunc = Vafunc,), param), 1.0, tol=1e-5)
    end
    θ̃mat
end

function cfocrhs(θ, w̃, is, εy, funcs, param::HabitParameters)
    @unpack Vafunc = funcs
    @unpack_HabitParameters param
    s = sgrid[is]
    s′ = s′func(s, εy, param)
    Ygrowth = exp(Δyfunc(εy, param))
    Rf = Rfvec[is]
    R = Rfunc(s, εy, param)
    w′ = w̃ * (θ * R + (1 - θ) * Rf) / Ygrowth
    Va = Vafunc(w′, s′)
    return β * Va * Ygrowth^-γ  *  R
end

function 𝔼cfocrhs(θ, w̃, is, funcs, param::HabitParameters)
    𝔼rhs = param.𝔼εy(εy->cfocrhs(θ, w̃, is, εy, funcs, param))
    return 𝔼rhs
end

function solvewbyw̃!(wmat, θ̃mat, Vafunc, param::HabitParameters)
    @unpack_HabitParameters param
    # Vafunc = partialinterpolate(agrid, Vamat, Linear())    
    for iter in CartesianIndices(θ̃mat)
        ia, is = iter[1], iter[2]
        w̃, s, S = agrid[ia], sgrid[is], Sgrid[is]
        θ̃ = θ̃mat[iter]
        𝔼rhs = 𝔼cfocrhs(θ̃, w̃, is, (Vafunc = Vafunc,), param)
        c = 𝔼rhs^(-1 / γ) + 1 - S
        w = w̃ + c + ψ - 1
        wmat[iter] = w
    end
    return wmat
end

function interpolatepolicy!(cmat, θmat, w̃mat, wmat, θ̃mat, param::HabitParameters)
    @unpack_HabitParameters param
    θ̃func = partialinterpolate(agrid, θ̃mat)
    for is in 1:ns
        s = sgrid[is]
        w_w̃ = @view wmat[:, is] # this gives a mapping from w̃grid -> w
        c_w = @view cmat[:, is]
        θ_w = @view θmat[:, is]
        @assert issorted(w_w̃)
        # if this mapping is monotone, we could take the inverse of this mapping:
        w̃_w = @view w̃mat[:, is] # relabel for notational clarity
        ŵgrid, w̃_ŵ = w_w̃, agrid # relabel it for notational clarity. The mapping ŵgrid, w̃_ŵ gives the policy function
        sorted_interpolation!(w̃_w, ŵgrid, w̃_ŵ, agrid)
        θ_w .= θ̃func.(w̃_w, is)
        @. c_w =  agrid + 1 - ψ - w̃_w
    end
end


function iteratepolicy!(hh, c0::Matrix{T}, param::HabitParameters) where {T}
    @unpack_HabitParameters param
    @unpack_HabitHouseholds hh
    cfunc = extrapolate(interpolate((agrid, sgrid), c0, Gridded(Linear())), Line())
    Vafunc = (w, s) -> (cfunc(w, s) + exp(s) - 1)^-γ
    iteratepolicy!(hh, Vafunc, param)
    return cmat
end

function iteratepolicy!(hh, Vafunc, param::HabitParameters)
    @unpack_HabitHouseholds hh
    solveθbyw̃!(θ̃mat, Vafunc, param)
    solvewbyw̃!(wmat, θ̃mat, Vafunc, param)
    interpolatepolicy!(cmat, θmat, w̃mat, wmat, θ̃mat, param)
end

function iterationhelper!(out, x, hh, param::HabitParameters)
    iteratepolicy!(hh, x, param)
    out .= hh.cmat
end



# guess for Lucas models
function guesscwratio(cwratio, ERgamma, param::HabitParameters)
    @unpack_HabitParameters param
    M = cwratio / (1 - cwratio) 
    A = M^-γ / (β * ERgamma)
    A - (M^(1 - γ) + M^-γ) * (1 / (1 + M))^(1 - γ)
end

function solveAandcwratio(param::HabitParameters)
    @unpack_HabitParameters param
    cwratio = zeros(size(sgrid))
    A = zeros(size(sgrid))
    for is in 1:ns
        s = sgrid[is]
        ERgamma = 𝔼εy(εy -> Rfunc(s, εy, param)^(1 - γ))
        cwratio[is] = find_zero(x -> guesscwratio(x, ERgamma, param), (1e-8, 1-1e-8), Roots.A42())
        M = cwratio[is] / (1 - cwratio[is]) 
        A[is] = M^-γ / (β * ERgamma)
    end
    return A, cwratio
end

function initializeHH!(hh, param::HabitParameters)
    @unpack_HabitParameters param
    @unpack_HabitHouseholds hh
    Avec, cwratio = solveAandcwratio(param)
    capitalizedΩ = (1 .- ψ) ./ 𝔼R
    wgrid = agrid .+ capitalizedΩ'
    @. cmat = wgrid * cwratio' + 1 - Sgrid'
    # @. cmat = max.(wgrid * cwratio' , 1 + 1e-3)
end



end