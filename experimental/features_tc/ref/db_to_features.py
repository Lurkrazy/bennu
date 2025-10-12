import tvm
from tvm import meta_schedule as ms
from tvm.tir import Schedule
from tvm.target import Target
from tvm.ir.module import IRModule
from tvm.script import tir as T

# Assume the database already exists and contains data; here, an in-memory database is used as an example
@T.prim_func
def matmul(
    A: T.Buffer((512, 512), "float32"),
    B: T.Buffer((512, 512), "float32"),
    C: T.Buffer((512, 512), "float32"),
) -> None:
    T.func_attr({"global_symbol": "main", "tir.noalias": True})
    for i0, i1, i2 in T.grid(512, 512, 512):
        with T.block("C"):
            i, j, k = T.axis.remap("SSR", [i0, i1, i2])
            T.reads(C[i, j], A[i, k], B[k, j])
            T.writes(C[i, j])
            with T.init():
                C[i, j] = T.float32(0)
            C[i, j] = C[i, j] + A[i, k] * B[k, j]

mod = tvm.IRModule({"main": matmul})
target = Target("llvm")
arg_info = ms.arg_info.ArgInfo.from_prim_func(func=mod["main"])

# Create a database and insert a record
db = ms.database.MemoryDatabase()
sch = Schedule(mod, debug_mask="all")
trace = sch.trace
db.commit_workload(mod)
db.commit_tuning_record(
    ms.database.TuningRecord(
        trace,
        workload=db.commit_workload(mod),
        run_secs=[1.0],
        target=target,
        args_info=arg_info,
    )
)

# Restore Schedule from the database
record = db.query_tuning_record(mod=mod, target=target, workload_name="main")
print("Restored record:", record)
restored_sch = Schedule(record.workload.mod)
record.trace.apply_to_schedule(restored_sch, remove_postproc=False)

# Or restore IRModule
restored_mod = restored_sch.mod

# Use feature_extractor to extract features
extractor = ms.feature_extractor.PerStoreFeature()
tune_ctx = ms.TuneContext(target=target)
candidate = ms.MeasureCandidate(sch=restored_sch, args_info=[])
(features,) = extractor.extract_from(tune_ctx, candidates=[candidate])

print("Features shape:", features.shape)
print("Features:", features.numpy())