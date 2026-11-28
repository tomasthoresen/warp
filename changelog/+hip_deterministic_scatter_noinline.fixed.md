Fix an illegal memory access on HIP in deterministic backward kernels that record into many gradient arrays: the record write is no longer inlined at every call site.
