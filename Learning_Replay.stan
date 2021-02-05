functions {
  int oc(int c) { // other choice (input {0,1})
    return (1-c); // return 0 if c==1; or return 1 if c==0
  }
}
data {
  int NS;
  int NT;
  int NSeq;
  int st[NS,NT]; // {1,2,3}
  int c[NS,NT];  // {0,1}
  int r[NS,NT];  // {-1,1}
  real rp[NS, NT, NSeq]; // replay
}
parameters {
  real betam;
  real biasm;
  real alphaDm;
  real alphaRIm;
  real alphaRSm;
  real<lower=0.001> betasd;
  real<lower=0.001> biasd;
  real<lower=0.001> alphaDsd;
  real<lower=0.001> alphaRIsd;
  real<lower=0.001> alphaRSsd;
  real betas[NS];
  real bias[NS];
  real alphaDs[NS];
  real alphaRIs[NS];
  real alphaRSs[NS];
}
model {
  betam ~ normal(0,1);
  biasm ~ normal(0,1);
  alphaDm ~ normal(0,1);
  alphaRIm ~ normal(0,1);
  alphaRSm ~ normal(0,1);
  betasd ~ normal(0,1);
  biasd ~ normal(0,1);
  alphaDsd ~ normal(0,1);
  alphaRIsd ~ normal(0,1);
  alphaRSsd ~ normal(0,1);
  for (s in 1:NS) { // Loop over subjects
    real Q[3, 2] = { { 0.0, 0.0 }, { 0.0, 0.0 }, { 0.0, 0.0 } }; // Initialize Q-values for this subject with zero
    betas[s] ~ normal(betam,betasd);
    bias[s] ~ normal(biasm,biasd);
    alphaDs[s] ~ normal(alphaDm,alphaDsd);
    alphaRIs[s] ~ normal(alphaRIm,alphaRIsd);
    alphaRSs[s] ~ normal(alphaRSm,alphaRSsd);
    for (t in 1:NT) { // Loop over trials
      if (c[s,t] >= 0) { // Only fit if subject entered response
        // Choice (softmax)
        c[s,t] ~ bernoulli_logit(betas[s] * (Q[st[s,t], 2] - Q[st[s,t], 1]) + bias[s]);
        // Q-learning
        if (st[s,t] == 1) {
          Q[1, c[s,t]+1] += Phi_approx(  alphaDs[s]  ) * ( r[s,t] - Q[1, c[s,t]+1] );
          Q[2, c[s,t]+1] += Phi_approx(  (alphaRIs[s] + alphaRSs[s]*( rp[s,t,3+c[s,t]]  ))/sqrt(2)  ) * ( r[s,t] - Q[2, c[s,t]+1] );
          Q[3, c[s,t]+1] += Phi_approx(  (alphaRIs[s] + alphaRSs[s]*( rp[s,t,5+c[s,t]]  ))/sqrt(2)  ) * ( r[s,t] - Q[3, c[s,t]+1] );
        }
        else if (st[s,t] == 2) {
          Q[2, c[s,t]+1] += Phi_approx(  alphaDs[s]  ) * ( r[s,t] - Q[2, c[s,t]+1] );
          Q[1, c[s,t]+1] += Phi_approx(  (alphaRIs[s] + alphaRSs[s]*( rp[s,t,1+c[s,t]]  ))/sqrt(2)  ) * ( r[s,t] - Q[1, c[s,t]+1] );
          Q[3, c[s,t]+1] += Phi_approx(  (alphaRIs[s] + alphaRSs[s]*( rp[s,t,5+c[s,t]]  ))/sqrt(2)  ) * ( r[s,t] - Q[3, c[s,t]+1] );
        }
        else if (st[s,t] == 3) {
          Q[3, c[s,t]+1] += Phi_approx(  alphaDs[s]  ) * ( r[s,t] - Q[3, c[s,t]+1] );
          Q[1, c[s,t]+1] += Phi_approx(  (alphaRIs[s] + alphaRSs[s]*( rp[s,t,1+c[s,t]]  ))/sqrt(2)  ) * ( r[s,t] - Q[1, c[s,t]+1] );
          Q[2, c[s,t]+1] += Phi_approx(  (alphaRIs[s] + alphaRSs[s]*( rp[s,t,3+c[s,t]]  ))/sqrt(2)  ) * ( r[s,t] - Q[2, c[s,t]+1] );
        }
      }
    }
  }
}

generated quantities {
  real alphaDm_phied;
  real alphaRm_phied;
  real alphaDm_phied_minus_alphaRm_phied;
  real alphaDm_minus_alphaRm;
  alphaDm_phied = Phi_approx(alphaDm);
  alphaRm_phied = Phi_approx(alphaRm);
  alphaDm_phied_minus_alphaRm_phied = alphaDm_phied-alphaRm_phied;
  alphaDm_minus_alphaRm = alphaDm-alphaRm;
}
