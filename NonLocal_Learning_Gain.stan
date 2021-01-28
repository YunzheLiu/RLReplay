functions {
  real softsign(real Q) {
    return exp(5.4 * Q) / (1 + exp(5.4 * Q));
  }
  int nl1(int st) { // non-local state 1
    if (st == 1) return 2;
    else if (st == 2) return 1;
    else if (st == 3) return 1;
    else return -1;
  }
  int nl2(int st) { // non-local state 2
    if (st == 1) return 3;
    else if (st == 2) return 3;
    else if (st == 3) return 2;
    else return -1;
  }
  int oc(int c) { // other choice (input {0,1})
    return (1-c); // return 0 if c==1; or return 1 if c==0
  }
}

data {
  int NS;
  int NT;
  int st[NS,NT]; // {1,2,3}
  int c[NS,NT];  // {0,1}
  int r[NS,NT];  // {-1,1}
}

parameters {  
  real betam;
  real alphaDm;
  real alphaHPm;
  real alphaLPm;
  real<lower=0.001> betasd;
  real<lower=0.001> alphaDsd;
  real<lower=0.001> alphaHPsd;
  real<lower=0.001> alphaLPsd;
  real betas[NS];
  real alphaDs[NS];
  real alphaHPs[NS];
  real alphaLPs[NS];
}

model {
  betam ~ normal(0,1);
  alphaDm ~ normal(0,1);
  alphaHPm ~ normal(0,1);
  alphaLPm ~ normal(0,1);
  betasd ~ normal(0,1);
  alphaDsd ~ normal(0,1);
  alphaHPsd ~ normal(0,1);
  alphaLPsd ~ normal(0,1);

  for (s in 1:NS) { // Loop over subjects
    real invTem;
    real alphaD;
    real alphaLP;
    real alphaHP;
    real Q[3, 2]   = { { 0.0, 0.0 }, { 0.0, 0.0 }, { 0.0, 0.0 } }; // Initialize Q-values for this subject with zero
    real Gain_diff = 0;
    real Gain_nl1 = 0;
    real Gain_nl2 = 0;

    betas[s] ~ normal(betam,betasd);
    alphaDs[s] ~ normal(alphaDm,alphaDsd);
    alphaHPs[s] ~ normal(alphaHPm,alphaHPsd);
    alphaLPs[s] ~ normal(alphaLPm,alphaLPsd);

    invTem    = betas[s];
    alphaD = Phi_approx(alphaDs[s]);
    alphaHP = Phi_approx(alphaHPs[s]);
    alphaLP = Phi_approx(alphaLPs[s]);

    for (t in 1:NT) { // Loop over trials
      if (r[s,t] >= -2) { // Only fit if subject entered response

        // Choice (softmax)
        c[s,t] ~ bernoulli_logit(invTem * (Q[st[s,t], 2] - Q[st[s,t], 1]));

        // Q-learning (local)
        Q[st[s,t], c[s,t]+1] += alphaD * ( r[s,t] - Q[st[s,t], c[s,t]+1] );

        // Compute Gain
        Gain_nl1 = ((Q[nl1(st[s,t]),oc(c[s,t])+1]*(r[s,t]-1))/(-2)) - (r[s,t]*(1/(1+exp(Q[nl1(st[s,t]),oc(c[s,t])+1]-Q[nl1(st[s,t]),c[s,t]+1]))) + (1/(1+exp(Q[nl1(st[s,t]),c[s,t]+1]-Q[nl1(st[s,t]),oc(c[s,t])+1])))*Q[nl1(st[s,t]),oc(c[s,t])+1]);
        Gain_nl2 = ((Q[nl2(st[s,t]),oc(c[s,t])+1]*(r[s,t]-1))/(-2)) - (r[s,t]*(1/(1+exp(Q[nl2(st[s,t]),oc(c[s,t])+1]-Q[nl2(st[s,t]),c[s,t]+1]))) + (1/(1+exp(Q[nl2(st[s,t]),c[s,t]+1]-Q[nl2(st[s,t]),oc(c[s,t])+1])))*Q[nl2(st[s,t]),oc(c[s,t])+1]);
        Gain_diff = Gain_nl1 - Gain_nl2;

        // Q-learning (non-local)
        Q[nl1(st[s,t]),c[s,t]+1] += softsign(Gain_diff)   *(alphaHP*(r[s,t]-Q[nl1(st[s,t]),c[s,t]+1]))  +  softsign(Gain_diff*-1)*(alphaLP*(r[s,t]-Q[nl1(st[s,t]),c[s,t]+1]));
        Q[nl2(st[s,t]),c[s,t]+1] += softsign(Gain_diff*-1)*(alphaHP*(r[s,t]-Q[nl2(st[s,t]),c[s,t]+1]))  +  softsign(Gain_diff)*   (alphaLP*(r[s,t]-Q[nl2(st[s,t]),c[s,t]+1]));
      }
    }
  }
}

generated quantities {

  real alphaHP_phied;
  real alphaLP_phied;
  real alphaD_phied;

  real alphaHP_minus_alphaD;
  real alphaHP_minus_alphaLP;
  real alphaD_minus_alphaLP;


  real Qout[NS, NT, 3, 2];

  alphaHP_phied = Phi_approx(alphaHPm);
  alphaLP_phied = Phi_approx(alphaLPm);
  alphaD_phied = Phi_approx(alphaDm);

  alphaHP_minus_alphaD  = alphaHPm-alphaDm; 
  alphaHP_minus_alphaLP = alphaHPm-alphaLPm;
  alphaD_minus_alphaLP  = alphaDm-alphaLPm;


  // re-run the model again to get the final Q value
  for (s in 1:NS) { // Loop over subjects
    real Q[3, 2] = { { 0.0, 0.0 }, { 0.0, 0.0 }, { 0.0, 0.0 } }; // Initialize Q-values for this subject with zero
    real Gain_diff=0;
    real Gain_nl1 = 0;
    real Gain_nl2 = 0;

    real alphaD;
    real alphaLP;
    real alphaHP;

    alphaD  = Phi_approx(alphaDs[s]);
    alphaHP = Phi_approx(alphaHPs[s]);
    alphaLP = Phi_approx(alphaLPs[s]);

    for (t in 1:NT) { // Loop over trials
      if (r[s,t] >= -2) { // Only fit if subject entered response

        // store Q value
        Qout[s, t, , ] = Q;

        // Q-learning (local)
        Q[st[s,t], c[s,t]+1] += alphaD * ( r[s,t] - Q[st[s,t], c[s,t]+1] );

        // Compute Gain
        // Gain_diff = (1/(1+exp(Q[nl2(st[s,t]),oc(c[s,t])+1]-Q[nl2(st[s,t]),c[s,t]+1])))*r[s,t]-(1/(1+exp(Q[nl1(st[s,t]),oc(c[s,t])+1]-Q[nl1(st[s,t]),c[s,t]+1])))*r[s,t]; 

        Gain_nl1 = ((Q[nl1(st[s,t]),oc(c[s,t])+1]*(r[s,t]-1))/(-2)) - (r[s,t]*(1/(1+exp(Q[nl1(st[s,t]),oc(c[s,t])+1]-Q[nl1(st[s,t]),c[s,t]+1]))) + (1/(1+exp(Q[nl1(st[s,t]),c[s,t]+1]-Q[nl1(st[s,t]),oc(c[s,t])+1])))*Q[nl1(st[s,t]),oc(c[s,t])+1]);
        Gain_nl2 = ((Q[nl2(st[s,t]),oc(c[s,t])+1]*(r[s,t]-1))/(-2)) - (r[s,t]*(1/(1+exp(Q[nl2(st[s,t]),oc(c[s,t])+1]-Q[nl2(st[s,t]),c[s,t]+1]))) + (1/(1+exp(Q[nl2(st[s,t]),c[s,t]+1]-Q[nl2(st[s,t]),oc(c[s,t])+1])))*Q[nl2(st[s,t]),oc(c[s,t])+1]);
        Gain_diff = Gain_nl1 - Gain_nl2;

        // Q-learning (non-local)

        Q[nl1(st[s,t]),c[s,t]+1] += softsign(Gain_diff)   *(alphaHP*(r[s,t]-Q[nl1(st[s,t]),c[s,t]+1]))  +  softsign(Gain_diff*-1)*(alphaLP*(r[s,t]-Q[nl1(st[s,t]),c[s,t]+1]));
        Q[nl2(st[s,t]),c[s,t]+1] += softsign(Gain_diff*-1)*(alphaHP*(r[s,t]-Q[nl2(st[s,t]),c[s,t]+1]))  +  softsign(Gain_diff)*   (alphaLP*(r[s,t]-Q[nl2(st[s,t]),c[s,t]+1]));     
      }
    }
  }
} 
