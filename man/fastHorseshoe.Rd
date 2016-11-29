\name{fastHorseshoe}
\alias{fastHorseshoe}
\alias{fastHS}
\title{ Fast Bayesian horseshoe regression by the ellipitical slice sampler }
\description{
  The main function of the ellpitical slice sampler for Bayesian shrinakge regression. This function has several built-in priors and user can also provide their own prior function (written as a R function).
}
\usage{
fastHorseshoe(Y, X, beta_hat = NULL, penalize = NULL, prior = "horseshoe", 
  user_prior_function = NULL, sigma = NULL, s2 = 1, kap2 = 1, N = 20000L, 
  burnin = 0L, thinning = 1L, vglobal = 100, verb = FALSE)
}
\arguments{
  \item{Y}{\code{data.frame}, \code{matrix}, or \code{vector} of inputs \code{Y}. Response variable. }
  \item{X}{\code{data.frame}, \code{matrix}, or \code{vector} of inputs \code{X}. Regressors. }
  \item{beta_hat}{\code{matrix}, or \code{vector} of inputs. Initial value of coefficients. The default value is OLS estimator.}
  \item{penalize}{A bool \code{vector} with the length the same as number of regressors. 1 indicates shrink corresponding regressor. The default value is to penalize all coefficients.}
  \item{prior}{Indicating shrinkage prior to use. \code{"horseshoe"} for approximate horseshoe prior (default), \code{"doubleexp"} for double exponential prior, \code{"normal"} for normal prior and \code{"cauchy"} for Cauchy prior.}
  \item{user_prior_function}{User can provide \code{R} log-prior function. See below for examples.}
  \item{sigma}{Initial value of residual standard error. The default value is half of standard error of \code{Y}.}
  \item{s2, kap2}{Parameter of prior over sigma, an inverse gamma prior with rate s2 and shape s2.}
  \item{N}{Number of posterior samples (after burn-in).}
  \item{burnin}{Number of burn-in samples.}
  \item{thinning}{Number of thinnings. \code{thinning = 1} means no thinning.}
  \item{vglobal}{Initial value of global shrinkage parameter.}
  \item{verb}{Bool, if \code{TRUE}, print out sampling progress.}
}
\details{
  The Bayesian lasso model and Gibbs Sampling algorithm is described
  in detail in Park & Casella (2008). 
}
\value{
  \item{loops}{A \code{vector} of number of ellpitical slice sampler loops for each posterior sample.}
  \item{sigma}{A \code{vector} of posterior samples of residual standard error.}
  \item{vglobal}{A \code{vector} of posterior samples of the global shrinkage parameter.}
  \item{beta}{A \code{matrix} of posterior samples of coefficients.}
}
\note{
  When using user-specified prior function, it should take a \code{vector} coefficient and a \code{vector} of global shrinkage parameters as arguments with same length as coefficients vector, and return the evaluation of logrithm of prior. See below for an example.
}

\references{
  Hahn, P. Richard, Jingyu He, and Hedibert Lopes. \emph{Elliptical slice sampling for Bayesian shrinkage regression with applications to causal inference.} (2016).
}

\author{ Jingyu He \email{jingyu.he@chicagobooth.edu} }


\examples{
## follow the lars diabetes example
library(lars)
data(diabetes)
attach(diabetes)

## fast horseshoe
fit1 = fastHorseshoe(y, x, N = 30000, burnin = 5000, thinning = 5)
## posterior mean of beta
colMeans(fit1$beta)
## posterior mean of global shrinakge parameter
mean(fit1$v)
## posterior mean of residual variance
mean(fit1$sigma)
## average loops of ellipitical slice sampler for each sample
mean(fit1$loops)



############################################
## only penalize the first coefficient
fit2 = fastHorseshoe(y, x, penalize = c(1, rep(0, 9)))

\dontrun{
############################################
## use user's own prior function
## approximate horseshoe prior, R code, which is the same as package default
log_horseshoe_approx_prior <- function(beta, v){
    beta = beta / v;
    ll = sum(log(log(1 + 2 / beta^2))) - sum(log(v))
    return(ll)
}

fit3 = fastHorseshoe(y, x, user_prior_function = log_horseshoe_approx_prior)

}
}

% Add one or more standard keywords, see file 'KEYWORDS' in the
% R documentation directory.
\keyword{  linear regression }
