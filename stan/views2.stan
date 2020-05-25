data {
  int<lower=1> no_obs;
//  int<lower=1> num_countries;     // for now just do a single country model
  int<lower=1> max_poly;
  int<lower=1> num_inputs;
  matrix[num_inputs*max_poly,no_obs] u;     // the input matrix
  row_vector[no_obs] y;                     // the output vector

  real<lower=0> beta;                       // for now set these as known
  real<lower=0> alpha;
  real<lower=0,upper=1> b;                  // also set this as initial results seemed to show was identifiable

}
transformed data {
    row_vector[no_obs] y_lin = log(y+1)/alpha;
}
parameters {
    real<lower=0,upper=1> a;        // coefficient for the x state
//    real<lower=0,upper=1> b;        // coefficient for the s state
//    real<lower=0> beta;             // part of the latent discrete state distribution
//    real<lower=0> alpha;            // part of the measurement model
    row_vector[max_poly*num_inputs] c;     // polynomial coefficients for u acting on x
    row_vector[max_poly*num_inputs] d;     // polynomial coefficients for u acting on s
    row_vector<lower=0>[max_poly*num_inputs] c_hyper;    // hyper priors for c
    row_vector<lower=0>[max_poly*num_inputs] d_hyper;    // hyper priors for d
    real<lower=0> shrinkage_param;

    real<lower=0,upper=1> sig_e;            // noise standard deviation

//    states
    row_vector[no_obs+1] x;
    row_vector[no_obs+1] s;
}
transformed parameters{
    // the probability of z = 1
    row_vector<lower=0>[no_obs+1] lambda;
//    lambda = inv_logit(beta * (exp(exp(s-1) - 1)-1));
    lambda = log(1+exp(s));
//    for (i in 1:(no_obs+1)){
//        lambda[i] = fmax(s[i],0.0);
//    }
//    lambda =
}
model {


    // hyper priors
    shrinkage_param ~ cauchy(0.0, 1.0);
    c_hyper ~ cauchy(0.0, 1.0);
    d_hyper ~ cauchy(0.0, 1.0);

    // parameters
    c ~ normal(0.0, c_hyper * shrinkage_param);
    d ~ normal(0.0, d_hyper * shrinkage_param);

    // noise standard deviation
    sig_e ~ cauchy(0.0, 1.0);

    // initial state priors
//    x[1] ~ normal(0, 1 - sig_e*sig_e);
    x[1] ~ normal(0, 1);
    s[1] ~ normal(0, 1);

    // state transition model
    x[2:no_obs+1] ~ normal(a*x[1:no_obs]+c*u, sqrt((1-a*a)*(1-sig_e*sig_e)));
    s[2:no_obs+1] ~ normal(b*s[1:no_obs]+d*u, sqrt(1-b*b));


    y_lin ~ normal(x[2:no_obs+1] .* lambda[2:no_obs+1], sig_e);
    // if yt > 0, then likelihood is just scaled down
    // measurement model
//    for (n in 1:no_obs){ // see https://mc-stan.org/docs/2_20/stan-users-guide/vectorizing-mixtures.html for why cant vectorise
//
//        if (y[n] > 0){
//            // lambda was the probabilty that z = 1
//            target += log(lambda[n+1]) + normal_lpdf(y_lin[n] | x[n+1], sig_e);
//            // + (1-lambda) * 0 percent probability that it came from z = 1
//
//        }
//        else{
//            target += log_mix(lambda[n+1],
//                            normal_lcdf( log(1.5)/alpha | x[n+1], sig_e),
//                            0.0);     // second part is ln(1)=0
//        }
//
//
//    }

}
generated quantities {
    row_vector[no_obs] x_p1 = a*x[1:no_obs]+c*u;
    row_vector[no_obs] s_p1 = b*s[1:no_obs]+d*u;
    row_vector[no_obs] lambda_p1 = inv_logit(beta * (exp(exp(s_p1-1) - 1))-1);


}

