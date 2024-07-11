# Data generation
n <- 250000
k <- 20
X <- matrix(rnorm(n * k), ncol = k)
y <- rbinom(n, size = 1, prob = plogis(3 * X[,1] - 2 * X[,2] + 1))
mdata <- list(k = k, n = n, y = y, X = X)

# CPU run

## Prerequisites (one-time setup)
renv::install('cmdstanr', repos = c('https://stan-dev.r-universe.dev', getOption("repos")), prompt=FALSE)

library(cmdstanr)
if (!dir.exists(paste0(Sys.getenv("HOME"),"/.cmdstan/cmdstan-2.34.1"))) {
cmdstanr::install_cmdstan(
  version = "2.34.1", 
  cores = parallelly::availableCores()
  )
}

## Compile the model 
cmdstanr::set_cmdstan_path(paste(Sys.getenv("HOME"),".cmdstan/cmdstan-2.34.1",sep="/"))
mod_cpu <- cmdstanr::cmdstan_model("model.stan",compile=FALSE)
mod_cpu$exe_file("model-cpu")
mod_cpu$compile()


## Run the model
time_cpu<-system.time(fit_cpu <- mod_cpu$sample(data = mdata, chains = 4, parallel_chains = 4, refresh = 0))


# GPU run 

## Prerequisites (one-time setup)
cmdstan_dir = "~/.cmdstan-gpu"

if (!dir.exists(paste0(cmdstan_dir, "/cmdstan-2.34.1"))) {
  cpp_options = list("LDFLAGS+= -lOpenCL")
  
  dir.create(cmdstan_dir)
  cmdstanr::install_cmdstan(
    version = "2.34.1",
    dir = cmdstan_dir,
    cpp_options = cpp_options,
    cores = parallelly::availableCores()
  )
}
cmdstanr::set_cmdstan_path(paste(cmdstan_dir,"cmdstan-2.34.1",sep="/"))

## Compile the model 
cmdstanr::set_cmdstan_path(paste(cmdstan_dir,"cmdstan-2.34.1",sep="/"))
mod_gpu <- cmdstanr::cmdstan_model("model.stan",compile=FALSE)
mod_gpu$exe_file("model-gpu")
mod_gpu$compile(cpp_options = list(stan_opencl = TRUE))


## Run the model
time_gpu<-system.time(fit_gpu <- mod_gpu$sample(data = mdata, chains = 4, parallel_chains = 4, refresh = 0))

# Compare CPU with GPU
time_cpu/time_gpu
