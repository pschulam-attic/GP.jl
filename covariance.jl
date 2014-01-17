abstract CovarianceKernel

function gpcov(kernel::CovarianceKernel,
               v1::Vector{Float64},
               v2::Vector{Float64})
    
    n1 = size(v1, 1)
    n2 = size(v2, 1)
    cv = zeros(n1, n2)

    for i in 1:n1
        for j in 1:n2
            cv[i, j] = gpcov(kernel, v1[i], v2[j])
        end
    end

    cv
end

type SquaredExponential <: CovarianceKernel
    bandwidth::Float64
end

function gpcov(kernel::SquaredExponential, x1::Float64, x2::Float64)
    bw = kernel.bandwidth
    r = x1 - x2
    exp(- r ^ 2 / (2 * bw ^ 2))
end
