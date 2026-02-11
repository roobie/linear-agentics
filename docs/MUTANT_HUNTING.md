# Mutant Hunting Strategies

## How mutmut Works

mutmut replaces your source files with instrumented versions in `mutants/`. Each function gets multiple mutant variants that are selected at runtime via the `MUTANT_UNDER_TEST` environment variable and a trampoline dispatcher.

### Key Files
- `mutants/<module>.meta` — JSON with `exit_code_by_key`: maps mutant names to exit codes
- `mutants/<module>.py` — Instrumented source with all mutant variants inline

### Exit Codes
- **1** = killed (test caught the mutation — good)
- **0** = survived (test passed despite mutation — test gap)
- **33** = skipped (no test covers this code path at all)

## Analysis Process

### Step 1: Read `.meta` files for triage
The `.meta` files are the most important artifact. Grep for `: 0` to find survivors:
```bash
grep ': 0' mutants/linear_agentics/<file>.py.meta
```

This gives you a list like `HttpToken.consume__mutmut_1: 0` — meaning mutant 1 of HttpToken.consume survived.

### Step 2: Count survivors by function
Group the survivors by function name to find the worst offenders. A function with 35 surviving mutants is a bigger gap than one with 2. Focus on functions with high survivor counts first.

### Step 3: Categorize survivors

**Category A: Missing functional tests** — The function is never actually called in tests. Often shows as ALL mutants surviving for that function. Example: HttpToken.consume had 35/35 survivors because the only test manually set `_consumed = True` and tested reuse, never calling consume().

**Category B: Weak assertions** — The function is tested, but assertions don't check enough. Example: Budget tests used `pytest.raises(BudgetTimeoutError)` without `match=`, so error message mutations (including `None`) survived.

**Category C: Boundary conditions** — Operator mutations (`>` → `>=`, `+` → `-`) survive because tests don't exercise exact boundaries. Example: `check_timeout` with `>` → `>=` survived because no test checked behavior at exactly the timeout limit.

**Category D: Schema/structure gaps** — Dict-building methods (like `to_tool_definition`) have many survivors because tests only check a few keys. Each mutated dict key or value is a separate mutant.

**Category E: Equivalent mutants** — Some mutations don't change observable behavior. Example: changing a cosmetic description string that nothing should depend on. These are false positives — don't waste time writing tests for them.

### Step 4: Read the mutated source to understand specific mutations
The `mutants/<module>.py` file contains all mutant variants inline. Each variant is named like `xǁClassNameǁmethod__mutmut_N`. Read the specific surviving variants to understand what changed:
- `mutmut_1` might change a default param value (`1` → `2`)
- `mutmut_2` might change an operator (`+=` → `=`)
- `mutmut_3` might change an operator (`+=` → `-=`)
- `mutmut_4` might change a comparison (`>` → `>=`)
- `mutmut_5` might change a string to `None`

### Step 5: Prioritize fixes by ROI

**High ROI** (fix first):
1. Category A (missing functional tests) — one test can kill 20-30 mutants
2. Category C (boundary conditions) — often reveals real bugs or spec ambiguities
3. Category B (weak assertions) — trivial to fix, add `match=` or tighter checks

**Medium ROI**:
4. Category D (schema gaps) — many kills per test, but tedious to write

**Low ROI** (skip or defer):
5. Category E (equivalent mutants) — impossible to kill without changing production code
6. CLI/UI code survivors — mocking stdin/stdout is fragile and low safety value

## Common Mutmut Mutations

| Original | Mutation | How to kill |
|----------|----------|-------------|
| `+=` | `=` or `-=` | Assert cumulative effect after multiple calls |
| `>` | `>=` or `<` | Test exact boundary value |
| `if x is None` | `if x is not None` | Test both None and non-None paths |
| `f"message {var}"` | `None` | Use `match=` in pytest.raises |
| `"string_key"` | `"XXstring_keyXX"` | Assert exact dict keys |
| `return value` | `return None` | Assert return value is not None |
| Default param `=1` | `=2` | Call without explicit param, assert default behavior |
| `param` | `None` | Assert stored attribute equals input |

## Gotchas

- **Large `.meta` files**: tokens.py.meta can be >256KB. Use `grep` or read with offset/limit.
- **Exit code 33 ≠ bug**: Skipped mutants just mean no test exercises that code. Infrastructure functions (http_request, db_query) often have no unit tests because they need real I/O — they're tested indirectly via token-level tests with mocks.
- **mutmut + Hypothesis**: mutmut runs tests from multiple executors, which triggers Hypothesis `differing_executors` HealthCheck. Suppress it with `@settings(suppress_health_check=[HealthCheck.differing_executors])` on tests that use `tempfile.mkdtemp()`.
