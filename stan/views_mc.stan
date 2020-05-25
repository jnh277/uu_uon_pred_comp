data {
  int<lower=1> no_obs;
  int<lower=1> num_countries;     // for now just do a single country model
  int<lower=1> max_poly;
  int<lower=1> num_inputs;
  matrix[num_countries,num_inputs*max_poly] u [no_obs];     // the input matrix array
  matrix[num_countries,no_obs] y;                     // the output vector
  real<lower=0> beta;                       // for now set these as known
  real<lower=0> alpha;
  real<lower=0,upper=1> b;                  // also set this as initial results seemed to show was identifiable
  real<lower=0,upper=1> a;
  real<lower=0,upper=1> sig_e;            // noise standard deviation
}
transformed data {
    matrix[num_countries,no_obs] y_lin = log(y+1)/alpha;
}
parameters {
//    real<lower=0,upper=1> a;        // coefficient for the x state
//    real<lower=0,upper=1> b;        // coefficient for the s state
//    real<lower=0> beta;             // part of the latent discrete state distribution
//    real<lower=0> alpha;            // part of the measurement model
    vector[max_poly*num_inputs] c;     // polynomial coefficients for u acting on x
    vector[max_poly*num_inputs] d;     // polynomial coefficients for u acting on s
    vector<lower=0>[max_poly*num_inputs] c_hyper;    // hyper priors for c
    vector<lower=0>[max_poly*num_inputs] d_hyper;    // hyper priors for d
    real<lower=0> shrinkage_param;
//    real<lower=0,upper=1> sig_e;            // noise standard deviation

//    states
    matrix[num_countries,no_obs+1] x;
    matrix[num_countries,no_obs+1] s;
}
transformed parameters{
    matrix[num_countries,no_obs] mu1;
    matrix[num_countries,no_obs] mu2;
    // the probability of z = 1
    matrix<lower=0,upper=1>[num_countries,no_obs+1] lambda;
    lambda = inv_logit(beta * (exp(exp(s-1) - 1)-1));

    // one step ahead predictions    for (i in 1:no_obs){
    for (i in 1:no_obs){
        mu1[:,i] = a*x[:,i]+u[i,:,:]*c;
        mu2[:,i] = b*s[:,i]+u[i,:,:]*d;
    }
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
//    sig_e ~ cauchy(0.0, 1.0);

    // initial state priors
//    x[1] ~ normal(0, 1 - sig_e*sig_e);
    x[1,:] ~ normal(0, 1);
    s[1,:] ~ normal(0, 1);

    // state transition model
    to_vector(x[:,2:no_obs+1]) ~ normal(to_vector(mu1), sqrt((1-a*a)*(1-sig_e*sig_e)));
    to_vector(s[:,2:no_obs+1]) ~ normal(to_vector(mu2), sqrt(1-b*b));

    // if yt > 0, then likelihood is just scaled down
    // measurement model
    for (n in 1:no_obs){ // see https://mc-stan.org/docs/2_20/stan-users-guide/vectorizing-mixtures.html for why cant vectorise
        for (i in 1:num_countries){
            if (y[i,n] > 0){
                // lambda was the probabilty that z = 1
                target += log(lambda[i,n+1]) + normal_lpdf(y_lin[i,n] | x[i,n+1], sig_e);
                // + (1-lambda) * 0 percent probability that it came from z = 1

            }
            else{
                target += log_mix(lambda[i,n+1],
                                normal_lcdf( log(1.5)/alpha | x[i,n+1], sig_e),
                                0.0);     // second part is ln(1)=0
            }
        }

    }

}
generated quantities {
//    row_vector[no_obs] x_p1 = a*x[1:no_obs]+c*u;
//    row_vector[no_obs] s_p1 = b*s[1:no_obs]+d*u;
    matrix[num_countries,no_obs] lambda_p1 = inv_logit(beta * (exp(exp(mu2-1) - 1))-1);


}

