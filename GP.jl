module GP
export GaussianProcess, SquaredExponential, predict

include("predict.jl")
include("covariance.jl")

end
