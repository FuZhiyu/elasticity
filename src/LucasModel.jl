
module LucasModel

include("utils.jl")
@with_kw struct LucasParameters
    @deftype Float64
    β = 0.98
    σY = 0.08
    σD = 0.15
    g = 0.02
    ρ = 0.02 # discounting
    γ = 5
    μx = 2/3
    ρx = 0.95
    σx = 0.03
    agrid::Vector{Float64} = [exp(x)-1 for x in 0.05:0.05:5] # wealth. w in the notes.
    ψgrid::Vector{Float64} = [0.01:0.01:0.99;] # D/Y
    na::Int64 = length(agrid)
    nψ::Int64 = length(ψgrid)
    𝔼εy::IterableExpectation{Array{Float64, 1}, Array{Float64, 1}} = expectation(Normal(), Gaussian; n = 10)
    # 𝔼εx::IterableExpectation{Array{Float64, 1}, Array{Float64, 1}} = expectation(Normal(), Gaussian; n = 10)
    Rfvec::Vector{Float64} = zeros(nψ)
    pdvec::Vector{Float64} = zeros(nψ)
    Aψ::Matrix{Float64}  = zeros(nψ, nψ)
end

function ψ_markov!(param)
    @unpack_LucasParameters param
    # Aψ = zeros(nψ, nψ)
    if nψ > 1
        ψgridmid = (ψgrid[1:end-1] + ψgrid[2:end]) / 2
    else
        ψgridmid = []
    end
    
    ψgridmid = [-Inf; ψgridmid; Inf]
    𝔼εx = expectation(Normal(), Gaussian; n = 1000)
    for i in 1:nψ
        for j in 1:nψ
            Aψ[i, j] = 𝔼εx(x-> ψgridmid[j] < ψ′func(ψgrid[i], x, param) <= ψgridmid[j+1])
        end
    end
    Aψ[Aψ.<1e-8] .= 0
    Aψ .= Aψ ./ sum(Aψ, dims = 2)
    return Aψ
end



@with_kw struct LucasHouseholds{T <: Real}
    # the first dimension for the matricies is w, and the second is the state variable z
    Vmat::Matrix{T} # value function
    V′mat::Matrix{T} # value function
    θmat::Matrix{T} # portfolio allocation
    θ̃mat::Matrix{T} # portfolio allocation by w̃
    cmat::Matrix{T} # consumption
    wmat::Matrix{T} # used for endogenous points
    w̃mat::Matrix{T} # used for endogenous points
    # θ̃mat::Matrix{T} # store temporary value
end

function LucasHouseholds(param, T = Float64; initializeV = true)
    @unpack_LucasParameters param
    mats =  [Matrix{T}(undef, na, nψ) for i in 1:length(fieldnames(LucasHouseholds))]
    hh = LucasHouseholds(mats...)
    # if initializeV
    #     initializeV!(hh, param)
    # end
    return hh
end


@inline function ψ′func(ψ, εx, param)
    @unpack_LucasParameters param
    # ρx * x + σx * εx
    # return exp(-σx^2/2 + σx * εx) * x
    if ψ == one(ψ)
        return ψ
    end
    x = log(μx * ψ / (1-ψ))
    x′ = ρx * x + σx * εx
    return exp(x′)/(μx + exp(x′))
end


function eulerequation(pdvec, iψ, iψ′, param)
    # @unpack_LucasParameters param
    @unpack β, γ, σY, ψgrid, g = param
    ψ, ψ′ = ψgrid[iψ], ψgrid[iψ′]
    pd, pd′ = pdvec[iψ], pdvec[iψ′]
    # ψ′ = ψ′func(ψ, εx, param)
    # pd′ = pdfunc(ψ′)
    return β * ψ′ / ψ * (pd′ + 1) / pd - exp((γ-1) * g - γ * (γ-1)/2 * σY^2)
end


function eulerequation(pdvec, param)
    @unpack Aψ, nψ, ψgrid = param
    # pdfunc = extrapolate(interpolate((ψgrid, ), pd, Gridded(Linear())), Line())
    res = zeros(size(ψgrid))
    for (iψ, ψ) in enumerate(ψgrid)
        for iψ′ in 1:nψ
            if param.Aψ[iψ,iψ′]>0
                res[iψ] += Aψ[iψ,iψ′] * eulerequation(pdvec, iψ, iψ′, param)
            end
        end
    end
    return res
end


function solveR!(param)
    @unpack_LucasParameters param
    rhs =  exp((γ-1) * g - γ * (γ-1)/2 * σY^2)
    pdconst = 1/(rhs / param.β - 1)
    res = nlsolve(x->eulerequation(x, param), pdconst * ones(size(param.ψgrid)), iterations = 20, method = :newton, show_trace = true)
    pdvec .= res.zero
    calculateRf!(param)
end

function calculateRf!(param)
    @unpack_LucasParameters param
    for iψ in 1:nψ
        ψ′_ψ = 0.0
        for iψ′ in 1:nψ
            if param.Aψ[iψ,iψ′]>0
                ψ′_ψ += Aψ[iψ,iψ′] * ψgrid[iψ′]/ψgrid[iψ]
            end
        end
        Rfvec[iψ] = 1/β * exp(γ * g - γ * (γ+1)/2 * σY^2)
    end
    return Rfvec
end


# @fastmath function w′func(w, ψ, c, θ, εy, ψ′, funcs, param)
#     @unpack Rffunc, pdfunc, V′func = funcs
#     @unpack_LucasParameters param
#     Ygrowth = exp(g - 1/2 * σY^2 + σY * εy)
#     Rf = Rffunc(ψ)
#     R = (pdfunc(ψ′) + 1) / pdfunc(ψ) * ψ′ / ψ * Ygrowth
#     R̃ = θ * R + (1-θ) * Rf
#     w′ = 1/Ygrowth * (w + 1 - ψ - c) * R̃
# end

@fastmath function V′εyεx(w′, ψ′, εy, funcs, param)
    @unpack V′func = funcs
    @unpack_LucasParameters param
    Ygrowth = exp(g - 1/2 * σY^2 + εy)
    return V′func(w′, ψ′) * Ygrowth.^(1-γ)
end

function V′εyεx(w, ψ, c, θ, εy, ψ′, funcs, param)
    w′ = w′func(w, ψ, c, θ, εy, ψ′, funcs, param)
    @unpack_LucasParameters param
    V′εyεx(w′, ψ′, εy, funcs, param)
end

# solving foc for θ
@inline @fastmath function Ygrowthεy(εy, param)
    @unpack g, σY = param
    Ygrowth = exp(g - 1/2 * σY^2 + σY * εy)
end

@inline @fastmath function Rfunc(iψ, iψ′, Ygrowth, param)
    @unpack pdvec, ψgrid = param
    ψ, ψ′ = ψgrid[iψ], ψgrid[iψ′]
    R = (pdvec[iψ′] + 1) / pdvec[iψ] * ψ′ / ψ * Ygrowth
end

@inline @fastmath function θfocεyεx(θ, w̃, iψ, εy, iψ′, funcs, param)
    @unpack_LucasParameters param
    @unpack Vafunc = funcs
    Ygrowth = Ygrowthεy(εy, param)
    R = Rfunc(iψ, iψ′, Ygrowth, param)
    Rf = Rfvec[iψ]
    w′ = w̃ * (θ * R + (1-θ) * Rf) / Ygrowth
    Va = Vafunc(w′, iψ′)
    return Va * Ygrowth^-γ  *  (R - Rf) * w̃^γ
end

@inline function θfocεx(θ, w̃, iψ, iψ′, funcs, param)
    return param.𝔼εy(εy->θfocεyεx(θ, w̃, iψ, εy, iψ′, funcs, param))
end

function θfoc(θ, w̃, iψ, funcs, param)
    @unpack nψ, Aψ = param
    res = zero(θ)
    @inbounds for iψ′ in 1:nψ
        if Aψ[iψ,iψ′] > zero(eltype(Aψ))
            res += Aψ[iψ,iψ′] * θfocεx(θ, w̃, iψ, iψ′, funcs, param)
        end
    end
    return res
end

function cfocrhs(θ, w̃, iψ, εy, iψ′, funcs, param)
    @unpack Vafunc = funcs
    @unpack_LucasParameters param
    Ygrowth = Ygrowthεy(εy, param)
    R = Rfunc(iψ, iψ′, Ygrowth, param)
    Rf = Rfvec[iψ]
    w′ = w̃ * (θ * R + (1-θ) * Rf) / Ygrowth
    Va = Vafunc(w′, iψ′)
    return β * Va * Ygrowth^-γ  *  R
end

function 𝔼cfocrhs(θ, w̃, iψ, funcs, param)
    𝔼rhs = zero(θ)
    @unpack nψ, 𝔼εy = param
    @inbounds for iψ′ in 1:nψ
        if Aψ[iψ,iψ′] > zero(eltype(Aψ))
            𝔼εycfocrhs = 𝔼εy(εy->(cfocrhs(θ, w̃, iψ, εy, iψ′, funcs, param)))
            𝔼rhs += Aψ[iψ,iψ′] * 𝔼εycfocrhs
        end
    end
    return 𝔼rhs
end

function solvewbyw̃!(wmat, θ̃mat, Vafunc, param)
    @unpack_LucasParameters param
    # Vafunc = partialinterpolate(agrid, Vamat, Linear())    
    for iter in CartesianIndices(θ̃mat)
        ia, iψ = iter[1], iter[2]
        w̃, ψ = agrid[ia], ψgrid[iψ]
        θ̃ = θ̃mat[iter]
        𝔼rhs = 𝔼cfocrhs(θ̃, w̃, iψ, (Vafunc = Vafunc, ), param)
        c = 𝔼rhs^(-1/γ)
        w = w̃ + c + ψ - 1
        wmat[iter] = w
    end
    return wmat
end

function interpolatepolicy!(cmat, θmat, w̃mat, wmat, θ̃mat, param)
    @unpack_LucasParameters param
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

# solving maximization for θ
@fastmath function θobjεyεx(θ, w̃, iψ, εy, iψ′, funcs, param)
    @unpack_LucasParameters param
    ψ, ψ′ = ψgrid[iψ], ψgrid[iψ′]
    @unpack V′func = funcs
    Ygrowth = Ygrowthεy(εy, param)
    R = (pdvec[iψ′] + 1) / pdvec[iψ] * ψ′ / ψ * Ygrowth
    Rf = Rfvec[iψ]
    w′ = w̃ * (θ * R + (1-θ) * Rf) / Ygrowth
    V′ = V′func(w′, iψ′)
    return V′ * Ygrowth^(1-γ)
end

function θobjεx(θ, w̃, iψ, iψ′, funcs, param)
    return param.𝔼εy(εy->θobjεyεx(θ, w̃, iψ, εy, iψ′, funcs, param))
end

function θobj(θ, w̃, iψ, funcs, param)
    res = zero(θ)
    @unpack Aψ = param
    for iψ′ in 1:nψ
        if param.Aψ[iψ,iψ′]>0
            res += Aψ[iψ,iψ′] * θobjεx(θ, w̃, iψ, iψ′, funcs, param)
        end
    end
    return res
end

function V′εx(w, ψ, c, θ, ψ′, funcs, param)
    return param.𝔼εy(εy->V′εyεx(w, ψ, c, θ, εy, ψ′, funcs, param))
end

function EV′(w, ψ, c, θ, funcs, param)
    param.𝔼εx(εx->V′εx(w, ψ, c, θ, ψ′func(ψ, εx, param), funcs, param))
end


function solveθbyw̃!(θ̃mat, Vafunc, param)
    @unpack_LucasParameters param
    # Vafunc = partialinterpolate(agrid, Vamat, Linear())
    for iter in CartesianIndices(θ̃mat)
        ia, iψ = iter[1], iter[2]
        w̃ = agrid[ia]
            θ̃mat[iter] = find_zero(x->θfoc(x, w̃, iψ, (Vafunc = Vafunc, ), param), (0.1, 3.0), Roots.A42(), tol = 1e-2)
    end
    θ̃mat
end

function iteratepolicy!(newcmat, cmat, param)
    @unpack_LucasParameters param
    # cfunc = partialinterpolate(agrid, cmat, Linear())
    # Vafunc = (w, iψ) -> max(cfunc(w, iψ), 1e-9)^-γ
    # Vafunc = (w, iψ) -> cfunc(w, iψ)^-γ
    Vafunc = partialinterpolate(agrid, cmat.^-γ, Linear())
    solveθbyw̃!(θ̃mat, Vafunc, param)
    solvewbyw̃!(wmat, θ̃mat, Vafunc, param)
    interpolatepolicy!(newcmat, θmat, w̃mat, wmat, θ̃mat, p)
    return newcmat
end

end