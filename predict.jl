type GaussianProcess
    noise::Float64
    kernel::CovarianceKernel
end

type GaussianProcessPrediction
    mean::Vector{Float64}
    sd::Vector{Float64}
end

function predict(gp::GaussianProcess,
                 x_in::Vector{Float64},
                 y_in::Vector{Float64},
                 x_out::Vector{Float64})

    n_in = size(x_in, 1)
    n_out = size(x_out, 1)

    mu_in = zeros(n_in)
    mu_out = zeros(n_out)

    sigma_in = gp_cov_outer(gp.kernel, x_in, x_in)
    sigma_in += gp.noise * eye(n_in)
    
    sigma_out = gp_cov_outer(gp.kernel, x_out, x_out)

    sigma_in_out = gp_cov_outer(gp.kernel, x_in, x_out)
    sigma_out_in = sigma_in_out'

    pred_mean = mu_out
    pred_mean += sigma_out_in * inv(sigma_in) * x_in

    pred_var = sigma_out
    pred_var -= sigma_out_in * inv(sigma_in) * sigma_in_out

    pred_sd = sqrt(diag(pred_var))

    GaussianProcessPrediction(pred_mean, pred_sd)
end
