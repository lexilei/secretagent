# NaturalPlan Error Analysis

---

# 0-Shot Run: run_20260318_082239

**Run:** `run_20260318_082239` (0-shot, stratified n=50 cal / 50 meet / 48 trip)

---

## 1. Error Type Definitions

| Type | Description |
|------|--------------|
| **exception** | Runtime failure: timeout, tool execution error, or failure to produce final answer |
| **no_parse / no_solution / no_plan** | Output format does not match evaluation requirements; unparseable |
| **time_only** | Correct day of week, wrong time slot |
| **day_only** | Correct time slot, wrong day of week |
| **day_and_time** | Both day and time slot wrong |
| **wrong_plan** | Valid format but incorrect planning logic (meeting route, trip route, durations, etc.) |

---

## 0-Shot: Summary by Task

### Calendar Scheduling (180 errors total)

| Type | Count | % |
|------|-------|---|
| no_parse | 74 | 41% |
| time_only | 56 | 31% |
| day_and_time | 30 | 17% |
| exception | 17 | 9% |
| day_only | 3 | 2% |

### Meeting Planning (217 errors total)

| Type | Count | % |
|------|-------|---|
| wrong_plan | 142 | 65% |
| exception | 66 | 30% |
| no_solution | 9 | 4% |

### Trip Planning (240 errors total)

| Type | Count | % |
|------|-------|---|
| wrong_plan | 104 | 43% |
| no_plan | 76 | 32% |
| exception | 60 | 25% |

---

## 0-Shot: Breakdown by Experiment

| Experiment | Total Errors | Distribution |
|------------|--------------|---------------|
| cal_zs_unstruct | 50 | no_parse: 50 (100%) |
| cal_zs_struct | 20 | time_only: 10, day_and_time: 5, exception: 4, day_only: 1 |
| cal_workflow | 38 | time_only: 16, no_parse: 9, day_and_time: 9, exception: 3, day_only: 1 |
| cal_pot | 42 | no_parse: 13, day_and_time: 12, time_only: 11, exception: 5, day_only: 1 |
| cal_react | 30 | time_only: 19, exception: 5, day_and_time: 4, no_parse: 2 |
| meet_zs_unstruct | 50 | wrong_plan: 50 (100%) |
| meet_zs_struct | 50 | exception: 28, wrong_plan: 22 |
| meet_workflow | 40 | wrong_plan: 27, exception: 13 |
| meet_pot | 40 | wrong_plan: 22, exception: 18 |
| meet_react | 37 | wrong_plan: 21, no_solution: 9, exception: 7 |
| trip_zs_unstruct | 48 | no_plan: 32, wrong_plan: 16 |
| trip_zs_struct | 48 | wrong_plan: 25, no_plan: 23 |
| trip_workflow | 48 | wrong_plan: 23, exception: 18, no_plan: 7 |
| trip_pot | 48 | wrong_plan: 21, exception: 19, no_plan: 8 |
| trip_react | 48 | exception: 23, wrong_plan: 19, no_plan: 6 |

---

## 0-Shot: Main Findings

1. **Format issues**: `cal_zs_unstruct` is 100% no_parse; ~32% of Trip errors are no_plan.
2. **Logic errors**: ~50% of Calendar errors are time_only or day_and_time; Meeting and Trip are dominated by wrong_plan.
3. **Exceptions**: ~25–30% of Meeting and Trip errors are exceptions (tool chain or execution failures).
4. **Method differences**: `cal_react` is mostly time_only (63%), so it outputs valid format but makes reasoning mistakes; `cal_zs_unstruct` never produces parseable output.

---

# 5-Shot Run: run_20260318_004100

**Run:** `run_20260318_004100` (5-shot, stratified n=50 cal / 50 meet / 48 trip)

---

## 5-Shot: Summary by Task

### Calendar Scheduling (131 errors total)

| Type | Count | % |
|------|-------|---|
| time_only | 49 | 37% |
| exception | 34 | 26% |
| no_parse | 25 | 19% |
| day_and_time | 19 | 15% |
| day_only | 4 | 3% |

### Meeting Planning (193 errors total)

| Type | Count | % |
|------|-------|---|
| wrong_plan | 95 | 49% |
| exception | 83 | 43% |
| no_solution | 15 | 8% |

### Trip Planning (206 errors total)

| Type | Count | % |
|------|-------|---|
| wrong_plan | 143 | 69% |
| exception | 44 | 21% |
| no_plan | 19 | 9% |

---

## 5-Shot: Breakdown by Experiment

| Experiment | Total Errors | Distribution |
|------------|--------------|---------------|
| cal_zs_unstruct | 17 | no_parse: 7, time_only: 4, day_and_time: 3, exception: 2, day_only: 1 |
| cal_zs_struct | 31 | time_only: 18, day_and_time: 7, exception: 5, day_only: 1 |
| cal_workflow | 16 | time_only: 6, day_and_time: 4, exception: 4, no_parse: 2 |
| cal_pot | 40 | exception: 15, no_parse: 14, time_only: 6, day_and_time: 4, day_only: 1 |
| cal_react | 27 | time_only: 15, exception: 8, no_parse: 2, day_only: 1, day_and_time: 1 |
| meet_zs_unstruct | 31 | wrong_plan: 31 (100%) |
| meet_zs_struct | 45 | exception: 35, wrong_plan: 10 |
| meet_workflow | 42 | exception: 28, wrong_plan: 14 |
| meet_pot | 40 | wrong_plan: 21, exception: 18, no_solution: 1 |
| meet_react | 35 | wrong_plan: 19, no_solution: 14, exception: 2 |
| trip_zs_unstruct | 43 | wrong_plan: 39, no_plan: 4 |
| trip_zs_struct | 35 | wrong_plan: 33, exception: 2 |
| trip_workflow | 37 | wrong_plan: 29, exception: 7, no_plan: 1 |
| trip_pot | 45 | wrong_plan: 21, exception: 19, no_plan: 5 |
| trip_react | 46 | wrong_plan: 21, exception: 16, no_plan: 9 |

---

## 5-Shot: Main Findings

1. **Fewer format issues**: `cal_zs_unstruct` drops from 100% no_parse (0-shot) to 41% no_parse (5-shot); Trip no_plan drops from 32% to 9%.
2. **More exceptions in Meeting**: 5-shot Meeting has 43% exceptions vs 30% in 0-shot; `meet_zs_struct` and `meet_workflow` are dominated by exceptions.
3. **Calendar**: time_only remains the largest category (37%); exception share rises to 26% vs 9% in 0-shot.
4. **Trip**: wrong_plan dominates (69%); fewer no_plan and exception than 0-shot.
