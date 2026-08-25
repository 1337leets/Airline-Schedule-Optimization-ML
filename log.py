import time

def load_data(func):
    def wrapper():
        print("\n[>>] Loading data...")
        try:
            r = func()
            print("[i.] Loading successful.")
            return r
        except Exception as e:
            raise SystemExit(f"[i.] Loading failed!\n{e}\n[>|] Terminating.")
    return wrapper


def preprocess(func):
    def wrapper(*args, **kwargs):
        print("\n[>>] Preprocessing...")
        r = func(*args, **kwargs)
        print(f"[i.] Found {len(r["df_transfer_local"])} candidate local connections, {len(r["markets"])} O&D/day markets, {len(r["prog_no_arr"])} inbound & {len(r["prog_no_dep"])} outbound adjustable legs.")
        return r
    return wrapper


def build(func):
    def wrapper(*args, **kwargs):
        print("\n[>>] Building model...")
        m, cand, mkt_c = func(*args, **kwargs)
        print(f"[i.] Built model with {len(cand)} candidate connections across {len(mkt_c)} markets.")
        return m, cand, mkt_c
    return wrapper


def start(func):
    def wrapper(*args, **kwargs):
        print("\n[>>] Warm start in progress...")
        t0 = time.time()
        arr0, dep0, obj0 = func(*args, **kwargs)
        print(f"[i.] Warm start completed in {time.time() - t0:.0f}s, total deviation: {obj0:.0f} min.")
        return arr0, dep0, obj0
    return wrapper


def solve(func):
    def wrapper(*args, **kwargs):
        print("\n[>>] Solving...")
        t0 = time.time()
        res = func(*args, **kwargs)
        print(f"[i.] Solve finished in {time.time() - t0:.0f}s, status: {res.solver.termination_condition}.")
        return res
    return wrapper


def extract(func):
    def wrapper(*args, **kwargs):
        print("\n[>>] Extracting results...")
        r = func(*args, **kwargs)
        print(f"[i.] Extracted {len(r["sel"])} selected connections across {len(r["per_mkt"])} markets, objective: {r["obj"]:.2f}.")
        return r
    return wrapper


def export(func):
    def wrapper(*args, **kwargs):
        print("\n[>>] Exporting results...")
        path = func(*args, **kwargs)
        print(f"[i.] Exported results to {path}.")
        return path
    return wrapper