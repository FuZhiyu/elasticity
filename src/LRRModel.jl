
module LLRModel


include("utils.jl")
export LLRParameters, LLRHouseholds, solveR!, iteratepolicy!, iterationhelper!
export @unpack_LLRParameters, @unpack_LLRHouseholds

@with_kw struct LLRParameters
    @deftype Float64
    β = 0.9
    σy = 0.1
    # σD = 0.15
    g = 0.02
    ρ = 0.02 # discounting
    γ = 5
    μx = 2 / 3
    ρx = 0.95
    σx = 0.03
    agrid::Vector{Float64} = [exp(x) - 1 for x in 0.05:0.05:5] # wealth. w in the notes.
    ψgrid::Vector{Float64} = [0.01:0.01:0.99;] # D/Y
    na::Int64 = length(agrid)
    nψ::Int64 = length(ψgrid)
    𝔼εy::IterableExpectation{Array{Float64,1},Array{Float64,1}} = expectation(Normal(), Gaussian; n=20)
    # 𝔼εx::IterableExpectation{Array{Float64, 1}, Array{Float64, 1}} = expectation(Normal(), Gaussian; n = 10)
    Rfvec::Vector{Float64} = zeros(nψ)
    pdvec::Vector{Float64} = zeros(nψ)
    𝔼R::Vector{Float64} = zeros(nψ)
    Aψ::Matrix{Float64}  = zeros(nψ, nψ)
end

function ψ_markov!(param)
    @unpack_LLRParameters param
    # Aψ = zeros(nψ, nψ)
    lom = (ψ, εx) -> ψ′func(ψ, εx, param)
    𝔼 = expectation(Normal(), Gaussian; n=10000)
    A = discretizeproces(ψgrid, lom, 𝔼)
    param.Aψ .= A
end


@inline function ψ′func(ψ, εx, param)
    @unpack_LLRParameters param
    # ρx * x + σx * εx
    # return exp(-σx^2/2 + σx * εx) * x
    if ψ == one(ψ)
        return ψ
    end
    x = log(μx * ψ / (1 - ψ))
    x′ = ρx * x + σx * εx
    return exp(x′) / (μx + exp(x′))
end


function eulerequation(pdvec, iψ, iψ′, param)
    # @unpack_LLRParameters param
    @unpack β, γ, σy, ψgrid, g = param
    ψ, ψ′ = ψgrid[iψ], ψgrid[iψ′]
    pd, pd′ = pdvec[iψ], pdvec[iψ′]
    # ψ′ = ψ′func(ψ, εx, param)
    # pd′ = pdfunc(ψ′)
    return β * ψ′ / ψ * (pd′ + 1) / pd - exp((γ - 1) * g - γ * (γ - 1) / 2 * σy^2)
end


function eulerequation(pdvec, param)
    @unpack Aψ, nψ, ψgrid = param
    # pdfunc = extrapolate(interpolate((ψgrid, ), pd, Gridded(Linear())), Line())
    res = zeros(size(ψgrid))
    for (iψ, ψ) in enumerate(ψgrid)
        for iψ′ in 1:nψ
            if param.Aψ[iψ,iψ′] > 0
                res[iψ] += Aψ[iψ,iψ′] * eulerequation(pdvec, iψ, iψ′, param)
            end
        end
    end
    return res
end


function solveR!(param)
    ψ_markov!(param)
    @unpack_LLRParameters param
    rhs =  exp((γ - 1) * g - γ * (γ - 1) / 2 * σy^2)
    pdconst = 1 / (rhs / param.β - 1)
    res = nlsolve(x -> eulerequation(x, param), pdconst * ones(size(param.ψgrid)), iterations=100, method=:newton)
    pdvec .= res.zero
    calculateRf!(param)
    𝔼R .= [𝔼εy(εy->Rfunc(iψ, iψ, Ygrowthεy(εy, param), param)) for iψ in 1:nψ]
end



function calculateRf!(param)
    @unpack_LLRParameters param
    for iψ in 1:nψ
        ψ′_ψ = 0.0
        for iψ′ in 1:nψ
            if param.Aψ[iψ,iψ′] > 0
                ψ′_ψ += Aψ[iψ,iψ′] * ψgrid[iψ′] / ψgrid[iψ]
            end
        end
        Rfvec[iψ] = 1 / β * exp(γ * g - γ * (γ + 1) / 2 * σy^2)
    end
    return Rfvec
end

#====================================================
#                 Dynamic Programming
====================================================#



@with_kw struct LLRHouseholds{T <: Real}
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

function LLRHouseholds(param, T=Float64; initializeHH=true)
    @unpack_LLRParameters param
    mats =  [Matrix{T}(undef, na, nψ) for i in 1:length(fieldnames(LLRHouseholds))]
    hh = LLRHouseholds(mats...)
    if initializeHH
        initializeHH!(hh, param)
    end
    return hh
end

function initializeHH!(hh, param)
    @unpack_LLRParameters param
    @unpack_LLRHouseholds hh
    Avec, cwratio = solveAandcwratio(param)
    # 𝔼R = [𝔼εy(εy->Rfunc(iψ, iψ, Ygrowthεy(εy, param), param)) for iψ in 1:nψ]
    capitalizedΩ = (1 .- ψgrid) ./ 𝔼R
    wgrid = agrid .+ capitalizedΩ'
    cmat .= wgrid .* cwratio'
    # V′mat = @.  wgrid^(1-γ) / (1-γ) .* Avec'
end

# @fastmath function w′func(w, ψ, c, θ, εy, ψ′, funcs, param)
#     @unpack Rffunc, pdfunc, V′func = funcs
#     @unpack_LLRParameters param
#     Ygrowth = exp(g - 1/2 * σy^2 + σy * εy)
#     Rf = Rffunc(ψ)
#     R = (pdfunc(ψ′) + 1) / pdfunc(ψ) * ψ′ / ψ * Ygrowth
#     R̃ = θ * R + (1-θ) * Rf
#     w′ = 1/Ygrowth * (w + 1 - ψ - c) * R̃
# end

@fastmath function V′εyεx(w′, ψ′, εy, funcs, param)
    @unpack V′func = funcs
    @unpack_LLRParameters param
    Ygrowth = exp(g - 1 / 2 * σy^2 + εy)
    return V′func(w′, ψ′) * Ygrowth.^(1 - γ)
end

function V′εyεx(w, ψ, c, θ, εy, ψ′, funcs, param)
    w′ = w′func(w, ψ, c, θ, εy, ψ′, funcs, param)
    @unpack_LLRParameters param
    V′εyεx(w′, ψ′, εy, funcs, param)
end

# solving foc for θ
@inline @fastmath function Ygrowthεy(εy, param)
    @unpack g, σy = param
    Ygrowth = exp(g - 1 / 2 * σy^2 + σy * εy)
end

@inline @fastmath function Rfunc(iψ, iψ′, Ygrowth, param)
    @unpack pdvec, ψgrid = param
    ψ, ψ′ = ψgrid[iψ], ψgrid[iψ′]
    R = (pdvec[iψ′] + 1) / pdvec[iψ] * ψ′ / ψ * Ygrowth
end

@inline @fastmath function θfocεyεx(θ, w̃, iψ, εy, iψ′, funcs, param)
    @unpack_LLRParameters param
    @unpack Vafunc = funcs
    Ygrowth = Ygrowthεy(εy, param)
    R = Rfunc(iψ, iψ′, Ygrowth, param)
    Rf = Rfvec[iψ]
    w′ = w̃ * (θ * R + (1 - θ) * Rf) / Ygrowth
    Va = Vafunc(w′, iψ′)
    return Va * Ygrowth^-γ  *  (R - Rf) * w̃^γ
end

@inline function θfocεx(θ, w̃, iψ, iψ′, funcs, param)
    return param.𝔼εy(εy -> θfocεyεx(θ, w̃, iψ, εy, iψ′, funcs, param))
end

function θfoc(θ, w̃, iψ, funcs, param)
    𝔼markov(iψ′ -> θfocεx(θ, w̃, iψ, iψ′, funcs, param), param.Aψ, iψ)
end

function cfocrhs(θ, w̃, iψ, εy, iψ′, funcs, param)
    @unpack Vafunc = funcs
    @unpack_LLRParameters param
    Ygrowth = Ygrowthεy(εy, param)
    R = Rfunc(iψ, iψ′, Ygrowth, param)
    Rf = Rfvec[iψ]
    w′ = w̃ * (θ * R + (1 - θ) * Rf) / Ygrowth
    Va = Vafunc(w′, iψ′)
    return β * Va * Ygrowth^-γ  *  R
end

function 𝔼cfocrhs(θ, w̃, iψ, funcs, param)
    𝔼rhs = 𝔼markov(iψ′ -> 
        param.𝔼εy(εy -> (cfocrhs(θ, w̃, iψ, εy, iψ′, funcs, param))), 
        param.Aψ, iψ)
    return 𝔼rhs
end

function solvewbyw̃!(wmat, θ̃mat, Vafunc, param)
    @unpack_LLRParameters param
    # Vafunc = partialinterpolate(agrid, Vamat, Linear())    
    for iter in CartesianIndices(θ̃mat)
        ia, iψ = iter[1], iter[2]
        w̃, ψ = agrid[ia], ψgrid[iψ]
        θ̃ = θ̃mat[iter]
        𝔼rhs = 𝔼cfocrhs(θ̃, w̃, iψ, (Vafunc = Vafunc,), param)
        c = 𝔼rhs^(-1 / γ)
        w = w̃ + c + ψ - 1
        wmat[iter] = w
    end
    return wmat
end

# solving maximization for θ
@fastmath function θobjεyεx(θ, w̃, iψ, εy, iψ′, funcs, param)
    @unpack_LLRParameters param
    ψ, ψ′ = ψgrid[iψ], ψgrid[iψ′]
    @unpack V′func = funcs
    Ygrowth = Ygrowthεy(εy, param)
    R = (pdvec[iψ′] + 1) / pdvec[iψ] * ψ′ / ψ * Ygrowth
    Rf = Rfvec[iψ]
    w′ = w̃ * (θ * R + (1 - θ) * Rf) / Ygrowth
    V′ = V′func(w′, iψ′)
    return V′ * Ygrowth^(1 - γ)
end

function θobjεx(θ, w̃, iψ, iψ′, funcs, param)
    return param.𝔼εy(εy -> θobjεyεx(θ, w̃, iψ, εy, iψ′, funcs, param))
end

function θobj(θ, w̃, iψ, funcs, param)
    res = zero(θ)
    @unpack Aψ = param
    for iψ′ in 1:nψ
        if param.Aψ[iψ,iψ′] > 0
            res += Aψ[iψ,iψ′] * θobjεx(θ, w̃, iψ, iψ′, funcs, param)
        end
    end
    return res
end

function V′εyψ′(θ, w̃, iψ, εy, iψ′, funcs, param)
    @unpack V′func = funcs
    @unpack_LLRParameters param
    Ygrowth = Ygrowthεy(εy, param)
    R = Rfunc(iψ, iψ′, Ygrowth, param)
    Rf = Rfvec[iψ]
    w′ = w̃ * (θ * R + (1 - θ) * Rf) / Ygrowth
    return V′func(w′, iψ′)
end

function 𝔼V′func(θ, w̃, iψ, funcs, param)
    𝔼V′ = 𝔼markov(iψ′ -> 
            𝔼εy(εy -> V′εyψ′(θ, w̃, iψ, εy, iψ′, funcs, param)), 
            param.Aψ, iψ)
    return 𝔼V′
end

function interpolatepolicy!(cmat, θmat, w̃mat, wmat, θ̃mat, param)
    @unpack_LLRParameters param
    θ̃func = partialinterpolate(agrid, θ̃mat)
    for iψ in 1:nψ
        ψ = ψgrid[iψ]
        w_w̃ = @view wmat[:, iψ] # this gives a mapping from w̃grid -> w
        c_w = @view cmat[:, iψ]
        θ_w = @view θmat[:, iψ]
        @assert issorted(w_w̃)
        # if this mapping is monotone, we could take the inverse of this mapping:
        w̃_w = @view w̃mat[:, iψ] # relabel for notational clarity
        ŵgrid, w̃_ŵ = w_w̃, agrid # relabel it for notational clarity. The mapping ŵgrid, w̃_ŵ gives the policy function
        sorted_interpolation!(w̃_w, ŵgrid, w̃_ŵ, agrid) 
        θ_w .= θ̃func.(w̃_w, iψ)
        @. c_w =  agrid + 1 - ψ - w̃_w
    end
end

function solveθbyw̃!(θ̃mat, Vafunc, param)
    @unpack_LLRParameters param
    # Vafunc = partialinterpolate(agrid, Vamat, Linear())
    @threads for iter in CartesianIndices(θ̃mat)
        ia, iψ = iter[1], iter[2]
        w̃ = agrid[ia]
        # θ̃mat[iter] = find_zero(x -> θfoc(x, w̃, iψ, (Vafunc = Vafunc,), param), (0.1, 3.0), Roots.A42(), tol=1e-5)
        θ̃mat[iter] = find_zero(x -> θfoc(x, w̃, iψ, (Vafunc = Vafunc,), param), 1.0, tol=1e-5)
    end
    θ̃mat
end

function iteratepolicy!(hh, c0::Matrix{T}, param) where {T}
    @unpack_LLRParameters param
    @unpack_LLRHouseholds hh
    cfunc = partialinterpolate(agrid, c0, Linear(), Line())
    # Vafunc = (w, iψ) -> max(cfunc(w, iψ), 1e-9)^-γ
    Vafunc = (w, iψ) -> cfunc(w, iψ)^-γ
    # Vamat .= c0.^-γ
    # Vafunc = partialinterpolate(agrid, Vamat, Linear(), Line())
    iteratepolicy!(hh, Vafunc, param)
    return cmat
end

function iteratepolicy!(hh, Vafunc, param)
    @unpack_LLRHouseholds hh
    solveθbyw̃!(θ̃mat, Vafunc, param)
    solvewbyw̃!(wmat, θ̃mat, Vafunc, param)
    interpolatepolicy!(cmat, θmat, w̃mat, wmat, θ̃mat, param)
end

function iteratepolicyfromV!(hh, V0, param)
    @unpack_LLRHouseholds hh
    @unpack_LLRParameters param
    centraldiff!(Vamat, V0, agrid)
    Vafunc = partialinterpolate(agrid, Vamat, Linear())
    iteratepolicy!(hh, Vafunc, param)
end

function iterationhelper!(out, x, hh, param)
    iteratepolicy!(hh, x, param)
    out .= hh.cmat
end

function calculateV!(Vmat, V0, cmat, w̃mat, θmat, param)
    @unpack_LLRHouseholds hh
    @unpack_LLRParameters param
    V′func = partialinterpolate(agrid, V0, Linear())
    for iter in CartesianIndices(Vmat)
        iψ = iter[2]
        θ = θmat[iter]
        w̃ = w̃mat[iter]
        c = cmat[iter]
        𝔼V′ = 𝔼V′func(θ, w̃, iψ, (V′func = V′func,), param)
        Vmat[iter] = c^(1 - γ) / (1 - γ) + β * 𝔼V′
    end
    return Vmat
end

function iteratevalue!(hh, V0, param; additionaliterations=0)
    @unpack_LLRParameters param
    @unpack_LLRHouseholds hh
    iteratepolicyfromV!(hh, V0, param)
    calculateV!(Vmat, V0, cmat, w̃mat, θmat, param)
    for counter in 1:additionaliterations
        V0 .= Vmat
        calculateV!(Vmat, V0, cmat, w̃, θmat, param)
    end
    return Vmat      
end


# helper functions for analytical solution
function guesscwratio(cwratio, ERgamma, param)
    @unpack_LLRParameters param
    M = cwratio / (1 - cwratio) 
    A = M^-γ / (β * ERgamma)
    A - (M^(1 - γ) + M^-γ) * (1 / (1 + M))^(1 - γ)
end

function solveAandcwratio(param)
    @unpack_LLRParameters param
    cwratio = zeros(size(ψgrid))
    A = zeros(size(ψgrid))
    for iψ in 1:nψ
        ERgamma = 𝔼markov(iψ′ -> 
        𝔼εy(εy -> Rfunc(iψ, iψ′, Ygrowthεy(εy, param), param)^(1 - γ)),
            Aψ, iψ)
        cwratio[iψ] = find_zero(x -> guesscwratio(x, ERgamma, param), (1e-8, 1-1e-8), Roots.A42())
        M = cwratio[iψ] / (1 - cwratio[iψ]) 
        A[iψ] = M^-γ / (β * ERgamma)
    end
    return A, cwratio
end




end