library("poLCA")
cfs = read.csv('/workspace/Eval/cfs_labelenc/adult/cfs_labelenc_adult_neg_6.csv')
probs.start = read.csv("/workspace/Eval/lcr_result/probs_start_adult.csv")
head(cfs)
# get mean and variance
income_mean = mean(cfs$income)
income_var = var(cfs$income)
age_mean = mean(cfs$age)
age_var = var(cfs$age)
hours_per_week_mean = mean(cfs$hours_per_week)
hours_per_week_var = var(cfs$hours_per_week)
# normalize continuous variables
cfs$income = scale(cfs$income)
cfs$age = scale(cfs$age)
cfs$hours_per_week = scale(cfs$hours_per_week)

set.seed(42)

f.income <- cbind(workclass, education, marital_status, occupation, race, gender)~income+age+hours_per_week
nes.income <- poLCA(f.income, cfs, nclass=3, maxiter=3000, nrep=20, probs.start=probs.start)

# save random state
# probs.start = nes.income$probs.start
# write.csv(probs.start, "/workspace/Eval/lcr_result/probs_start_adult.csv")

# visualize the results
# fix age, hours_per_week
# Note: When the regression formula is standardized, the input must be standardized values.
strreps <- cbind(1,seq((0.0-income_mean)/sqrt(income_var), (1.0-income_mean)/sqrt(income_var), length=1000),(63-age_mean)/sqrt(age_var),(60-hours_per_week_mean)/sqrt(hours_per_week_var))
exb.strreps <- exp(strreps %*% nes.income$coeff)
matplot(seq(0.0, 1.0, length=1000),(cbind(1,exb.strreps)/(1+rowSums(exb.strreps))),
        main="Income probability and class membership for age=63, hw=60",
        xlab="High income probability",ylab="Probability of latent class membership",
        ylim=c(0,1),type="l",col=c(3,4,5,6),lwd=3, lty=1:4)

write.csv(cbind(1,exb.strreps)/(1+rowSums(exb.strreps)), "/workspace/Eval/lcr_result/adult_neg_6_age_63_hw_60.csv")
write.csv(nes.income$probs, "/workspace/Eval/lcr_result/adult_neg_6_prob.csv")

# fix income, hours_per_week
# strreps <- cbind(1,(0.2-income_mean)/sqrt(income_var),seq((17-age_mean)/sqrt(age_var), (60-age_mean)/sqrt(age_var), length=44),(40-hours_per_week_mean)/sqrt(hours_per_week_var))
# exb.strreps <- exp(strreps %*% nes.income$coeff) # compute dot product
# matplot(c(17:60),(cbind(1,exb.strreps)/(1+rowSums(exb.strreps))),
#         main="Age and class membership for income=0.6, hw=38",
#         xlab="Age",ylab="Probability of latent class membership",
#         ylim=c(0,1),type="l", lwd=3,lty=1:3)
# legend(5,0.8,c("Class1","Class2","Class3","Class4"),lty=1:4)


# fix income, age
# strreps <- cbind(1,(0.85-income_mean)/sqrt(income_var),(50-age_mean)/sqrt(age_var),seq((1-hours_per_week_mean)/sqrt(hours_per_week_var), (99-hours_per_week_mean)/sqrt(hours_per_week_var), length=99))
# exb.strreps <- exp(strreps %*% nes.income$coeff)
# matplot(c(1:99),(cbind(1,exb.strreps)/(1+rowSums(exb.strreps))),
#         main="Income probability and class membership for income=0.85, age=50",
#         xlab="Hours per week",ylab="Probability of latent class membership",
#         ylim=c(0,1),type="l",col=c(3,4,5,6),lwd=3, lty=1:4)
# legend(25,0.8,c("Class1","Class2","Class3","Class4"),lty=1:4)

# write.csv(cbind(1,exb.strreps)/(1+rowSums(exb.strreps)), "/workspace/Eval/lcr_result/adult_neg1.csv")                  
