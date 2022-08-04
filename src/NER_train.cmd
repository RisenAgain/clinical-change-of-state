executable = NER_train.sh
getenv     = true
output     = condor_out/NER_train.out
error      = condor_out/NER_train.error
log        = condor_out/NER_train.log
notification = complete
request_GPUs = 1
transfer_executable = false
request_memory = 4*1024
queue
