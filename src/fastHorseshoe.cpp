// [[Rcpp::depends(RcppArmadillo)]]
// [[Rcpp::plugins(cpp11)]]
#include <RcppArmadillo.h>
#include <cmath>
#include <time.h>
#include <stdlib.h>
#include <iostream>
using namespace Rcpp;
using namespace arma;
using namespace std;
using namespace R;

inline double log_horseshoe_approx_prior(arma::mat beta, arma::vec v);
inline arma::vec sample_exp(arma::vec lambda);
inline arma::mat scaling(arma::mat x);
inline double log_double_exp_prior(arma::mat beta, arma::vec v);
inline double log_cauchy_prior(arma::mat beta, arma::vec v);
inline double log_normal_prior(arma::mat beta, arma::vec v);
inline double betaprior(arma::mat beta, arma::vec v, int prior, Rcpp::Nullable<Rcpp::Function> user_prior_function);
inline double user_prior_function_wrapper(arma::mat beta, arma::vec v, Rcpp::Function f);

// inline double sampCauchyVar(arma::mat beta, double sigma, int num, double alpha);
// slice sampler for horseshoe regression

/*
    input list :
    Model : Y ~ X, Y is n * 1 and X is n * p matrix
    n : number of observations
    p : number of regressors
    YY : t(Y) %*% Y
    YX : t(Y) %*% X
    XX : t(X) %*% X
    beta_hat : mean of coefficients, OLS estimator
    beta : initial value
    tads_global : tads parameter for global shrinkage parameter
    sigma : scale parameter, 1
    s2, kap2 : parameters of inverse gamma prior
    nsamps : number of samples you want to get
    burn : number of burn - in
    skip : skip in the markov chain. (The true number of samples is nsamps * skip)prior
*/

// [[Rcpp::export]]
List fastHorseshoe(arma::mat Y, arma::mat X, arma::vec beta_hat, arma::vec beta, arma::uvec penalize, int prior_type = 1, Rcpp::Nullable<Rcpp::Function> user_prior_function = R_NilValue, double sigma = 0.5, double s2 = 4, double kap2 = 16,  int nsamps = 10000, int burn = 1000, int skip = 10, double vglobal = 100, bool verb = false){    

    // dimensions
    int n = X.n_rows;
    int p = X.n_cols;

    // compute standard derivation
    double sdy = as_scalar(stddev(Y));

    // scale the initial value of residual variance
    sigma = sigma / sdy;


    arma::mat sdx = stddev(X, 0);

    // scaling matrix by column
    Y = scaling(Y);
    X = scaling(X);
 
    // compute sufficient statistics
    arma::mat YY = trans(Y) * Y;
    arma::mat YX = trans(Y) * X;
    arma::mat XX = trans(X) * X;


    penalize = find(penalize > 0);
    /*
    the input of penalize is (0,1,1,0,1...)
    convert to indeces of 1
    */

    burn = burn + 1;

    double s = sigma;
    double ssq;
    double ly;
    double thetaprop;
    double thetamin;
    double thetamax;
    //double vprop;
    double piprop;
    double picurr;
    arma::mat b;
    arma::vec v(p);
    v.fill(1.0);
    // double vglobal;
    double vgprop;
    //vglobal = 5;
    arma::vec lambda;
    arma::mat S;
    S = XX;
    arma::mat U;
    arma::vec D;
    arma::mat V;
    svd(U, D, V, S);
    arma::mat Linv = U.cols(0, p - 1) * arma::diagmat(1 / sqrt(D));

    //initialize
    arma::mat bsamps(p, nsamps);
    bsamps.fill(0.0);
    arma::vec ssamps(nsamps);
    ssamps.fill(0.0);
    arma::vec vsamps(nsamps);
    vsamps.fill(0.0);
    arma::vec ssq_out(nsamps);
    ssq_out.fill(0.0);
    int loopcount = 0;
    arma::vec loops(nsamps);
    loops.fill(0);
    double u;

    arma::mat nu;
    nu.fill(0.0);

    arma::vec eps(p);
    eps.fill(0.0);

    arma::vec betaprior_values(200);
    arma::vec v_grid(200);
    arma::uvec temp;
    arma::vec all_one;
    arma::mat betaprop;
    arma::vec temp2;
    double priorcomp;
    int iter = 0;

    // clock_t t1, t2;

    double mean_v_grid;
    double var_v_grid;
    double shape_proposal = 1.0;
    double scale_proposal = 1.0;
    int N_v_grid = 1000;


    // t1 = clock();
    int h = 0;

    int count_prior_loop = 0;

    while (h < nsamps)
    {

        if(verb == true && h % 1000 == 0 && h > 1){
            Rprintf("%d\n", h);
        }

        eps = rnorm(Linv.n_cols);

        nu = Linv * eps;

        nu = s * nu;

        u = runif(1, 0, 1)[0];

        priorcomp = betaprior(beta.elem(penalize) + beta_hat.elem(penalize), vglobal * v.elem(penalize), prior_type, user_prior_function);

        ly = priorcomp + log(u);

        thetaprop = runif(1, 0, 2 * M_PI)[0];

        betaprop = beta * cos(thetaprop) + nu * sin(thetaprop);

        thetamin = thetaprop - 2.0 * M_PI;

        thetamax = thetaprop;

        while (betaprior(beta_hat.elem(penalize) + betaprop.elem(penalize), vglobal * v.elem(penalize), prior_type, user_prior_function) < ly)
        {
            loopcount += 1;

            if (thetaprop < 0)
            {
                thetamin = thetaprop;
            } else
            {thetamax = thetaprop;}

            thetaprop = runif(1, thetamin, thetamax)[0];

            betaprop = beta * cos(thetaprop) + nu * sin(thetaprop);

        }



        beta = betaprop;

        b = beta_hat + beta;

        ssq = as_scalar(YY) - 2.0 * as_scalar(YX * (beta + beta_hat)) + as_scalar(trans(beta + beta_hat) * XX * (beta + beta_hat));

        // ssq = ssq + as_scalar(sum(pow((beta + beta_hat) / vglobal, 2)));

        s = 1.0 / sqrt(Rcpp::rgamma(1, (n + kap2) / 2.0,  2.0 / (ssq + s2))[0]);

        if(count_prior_loop < 50){

            count_prior_loop = count_prior_loop + 1;
                   
                    //lambda = abs(Rcpp::rt(p, 1));
            // t1 = clock();
            
            v_grid.ones(N_v_grid);

            for(int i = 0; (unsigned) i < v_grid.n_elem; i++){
            
                v_grid[i] = i + 1;
                    
            }

            v_grid = v_grid / 100 * 0.01;

            betaprior_values.ones(N_v_grid);

            for(int i = 0; (unsigned) i < v_grid.n_elem; i++){

                betaprior_values[i] = betaprior(b.elem(penalize), v_grid[i] * v.elem(penalize), prior_type, user_prior_function) +  R::dnorm(v_grid[i], 0.0, 1.0, 1) ;

            }

            betaprior_values = betaprior_values / sum(betaprior_values);

            mean_v_grid = sum(v_grid % betaprior_values);

            var_v_grid = sum(betaprior_values % pow((v_grid - sum(v_grid % betaprior_values )), 2 ));

            var_v_grid = var_v_grid * 2.0;

            scale_proposal = var_v_grid / mean_v_grid; 

            shape_proposal = mean_v_grid / scale_proposal;

        }

        vgprop = Rcpp::rgamma(1, shape_proposal, scale_proposal)[0];

        piprop = betaprior(b.elem(penalize), vgprop  * v.elem(penalize), prior_type, user_prior_function) + R::dnorm(vgprop, 0.0, 1.0, 1)  + R::dgamma(vglobal, shape_proposal, scale_proposal, 1);
                    
        picurr = betaprior(b.elem(penalize), vglobal * v.elem(penalize), prior_type, user_prior_function) + R::dnorm(vglobal, 0.0, 1.0, 1) + R::dgamma(vgprop, shape_proposal, scale_proposal, 1);
                    
        if (as_scalar(randu(1)) < exp(piprop-picurr)){

            vglobal = vgprop;
                    
        }


        //vglobal = sampCauchyVar(b, vglobal, p, 0.5);


        iter = iter + 1;


        if (iter > burn)
        {
            if (iter % skip == 0)
            {
                bsamps.col(h) = betaprop + beta_hat;
                ssamps(h) = s;
                vsamps(h) = vglobal;
                // ssq_out(h) = ssq;
                loops(h) = loopcount;
                h = h + 1;
            }
        }

        // re-count for the next round.
        loopcount = 0;
    }



    // t2 = clock();
    // float time_elapse =  ((float)t2 - (float)t1);
    // time_elapse = time_elapse / CLOCKS_PER_SEC;
    // cout <<"Running time of the loop is : " << time_elapse << endl;

    // X and Y were scaled at the beginning, rescale estimations
    for(int ll = 0; (unsigned) ll < bsamps.n_cols; ll ++){
        bsamps.col(ll) = bsamps.col(ll) / trans(sdx) * sdy;
    }

    ssamps = ssamps * sdy;


    bsamps = trans(bsamps);


    return List::create(Named("loops") = loops, Named("sigma") = ssamps, Named("vglobal") = vsamps, Named("beta") = bsamps);
}


inline double betaprior(arma::mat beta, arma::vec v, int prior, Rcpp::Nullable<Rcpp::Function> user_prior_function){
    
    double output = 0;

    if(user_prior_function.isNotNull()){

        Function user_prior_function_2(user_prior_function);

        output = user_prior_function_wrapper(beta, v, user_prior_function_2);
    }
    else{
        switch (prior){
            case 1:
                output = log_horseshoe_approx_prior(beta, v);
                break;
            case 2:
                output = log_double_exp_prior(beta, v);
                break;
            case 3:
                output = log_normal_prior(beta, v);
                break;
            case 4:
                output = log_cauchy_prior(beta, v);
                break;
            default:
                Rprintf("Wrong input of prior types.\n");
            }
    }
    return output;
}

inline double user_prior_function_wrapper(arma::mat beta, arma::vec v, Rcpp::Function f)
{
    SEXP result = f(beta, v);
    double output = Rcpp::as<double>(result);
    return output;
}



inline double log_horseshoe_approx_prior(arma::mat beta, arma::vec v)
{
    arma::vec beta2 = conv_to<vec>::from(beta);
    beta2 = beta2 / v;
    arma::vec temp = log(log(1.0 + 2.0 / (pow(beta2, 2.0))));
    double ll;
    ll = sum(temp) - sum(log(v));
    return ll;
}

inline arma::vec sample_exp(arma::vec lambda)
{
    int n = lambda.n_elem;
    arma::vec sample;
    sample = randu<vec>(n);
    sample =  - log(1 - sample) / lambda;
    return (sample);
}

inline arma::mat scaling(arma::mat x){
    // This function normalize a matrix x by column
    int n = x.n_rows;
    // int p = x.n_cols;
    arma::mat x_output;
    arma::mat mean_x;
    arma::mat sd_x;
    // normalize each column
    x_output = x;
    mean_x = mean(x, 0);
    sd_x = stddev(x, 0);
    for(int i = 0; i < n; i++){
        x_output.row(i) = (x.row(i) - mean_x) / sd_x;
    }
    return x_output;
}



inline double log_double_exp_prior(arma::mat beta, arma::vec v){
    // log density of double exponential prior
    arma::vec beta2 = conv_to<vec>::from(beta);
    beta2 = beta2 / v;
    arma::vec temp = (-1.0) * abs(beta2);
    double ll;
    ll = sum(temp) - sum(log(v));
    return ll;
}

inline double log_cauchy_prior(arma::mat beta, arma::vec v){
    // log density of Cauchy prior
    arma::vec beta2 = conv_to<vec>::from(beta);
    beta2 = beta2 / v;
    arma::vec temp = log(1.0 + pow(beta2, 2.0));
    double ll;
    ll = (-1.0) * sum(temp) - sum(log(v));
    return ll;
}

inline double log_normal_prior(arma::mat beta, arma::vec v){
    // log density of normal prior
    arma::vec beta2 = conv_to<vec>::from(beta);
    beta2 = beta2 / v;
    arma::vec temp = pow(beta2, 2.0);
    double ll;
    ll = (- 1.0 / 2.0) * sum(temp) - sum(log(v));
    return ll;
}


