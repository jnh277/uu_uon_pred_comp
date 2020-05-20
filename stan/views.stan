data {
  int<lower=1> no_obs;
  int<lower=1> num_countries;
  int<lower=1> max_poly;
  row_vector[no_obs] u_a;
  row_vector[no_obs] u_d;
  row_vector[no_obs] u_p;
  row_vector[no_obs] u_c;
  row_vector[no_obs] u_b;
  row_vector[no_obs] y;

}
transformed data {
}
parameters {
    real<lower=0,upper=1> a;
    real<lower=0,upper=1> b;
    vector[max_poly] c_a;            // polynomial coefficients for u_a
    vector[max_poly] c_d;
    vector[max_poly] c_p;
    vector[max_poly] c_c;
    vector[max_poly] c_b;
    vector[max_poly] c_a_hyper;            // hyper prior for polynomial coefficients for u_a
    vector[max_poly] c_d_hyper;
    vector[max_poly] c_p_hyper;
    vector[max_poly] c_c_hyper;
    vector[max_poly] c_b_hyper;
    vector[max_poly] d_a;            // polynomial coefficients for u_a
    vector[max_poly] d_d;
    vector[max_poly] d_p;
    vector[max_poly] d_c;
    vector[max_poly] d_b;
    vector[max_poly] d_a_hyper;            // hyper prior for polynomial coefficients for u_a
    vector[max_poly] d_d_hyper;
    vector[max_poly] d_p_hyper;
    vector[max_poly] d_c_hyper;
    vector[max_poly] d_b_hyper;
    real<lower=0> shrinkage_param;


    real<lower=0> sig_2;            // noise standard deviation

//    states
    row_vector[no_obs] x;
    row_vector[no_obs] s;
}
transformed parameters{
    row_vector[no_obs_est] yhat;
    yhat[1:max_order] = rep_row_vector(0.0,max_order);
    for (i in max_order+1:no_obs_est){
//        ehat[i] = y_est[i-output_order:i-1]*f_coefs + y_est[i] - u_est[i-input_order+1:i] * b_coefs
//              - ehat[i-output_order:i-1]*f_coefs;
        yhat[i] = u_est[i-input_order+1:i] * b_coefs - y_est[i-output_order:i-1]*f_coefs
                    + ehat[i-output_order:i-1]*f_coefs + ehat[i];
    }


}
model {
    // hyper priors
    shrinkage_param ~ cauchy(0.0, 1.0);
    b_coefs_hyperprior ~ cauchy(0.0, 1.0);
    f_coefs_hyperprior ~ cauchy(0.0, 1.0);

    // parameters
    b_coefs ~ normal(0.0, b_coefs_hyperprior * shrinkage_param);
    f_coefs ~ normal(0.0, f_coefs_hyperprior * shrinkage_param);
//    ehat ~ normal(0.0, r);

    // noise standard deviation
    r ~ cauchy(0.0, 1.0);
    r2 ~ cauchy(0.0, 1.0);

    // measurement likelihood
    ehat ~ normal(0.0, r);     // this includes the e_init prior
    y_est[max_order+1:no_obs_est] ~ normal(yhat[max_order+1:no_obs_est], r2);

}
generated quantities {
    row_vector[no_obs_val] y_hat_val = rep_row_vector(0.0,no_obs_val);
//    row_vector[no_obs_val] e_val = rep_row_vector(0.0,no_obs_val);
    for (i in max_order+1:no_obs_val){ // this isn't the best estimate of y_val as it doesnt have the error terms?
        y_hat_val[i] = u_val[i-input_order+1:i] * b_coefs
                - y_val[i-output_order:i-1] * f_coefs;// -e_val[i-output_order:i-1] * f_coefs;
//        e_val[i] = y_val[i] - y_hat_val[i];
    }
}

