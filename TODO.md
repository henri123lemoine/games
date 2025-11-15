# TODO: Project Status and Remaining Tasks

## ✅ COMPLETED: Core Algorithm Fixes (Latest Update)

### 1. Fixed MCCFR Regret Computation ✅
**What was broken**: Both vanilla MCCFR and Deep MCCFR had fundamentally broken regret updates
- **Issue**: Regret computation used `utility - strategy @ [utility, utility]` which always equals 0
- **Impact**: Regrets never accumulated, strategies never improved
- **Fix**: Implemented proper outcome sampling MCCFR:
  - For action taken: use observed utility
  - For actions not taken: use 0 as baseline (conservative)
  - Regret = value_estimate[action] - expected_value_under_strategy
- **Files Changed**:
  - `twentyone-rl/src/twentyone_rl/agents/mccfr.py` (lines 142-164)
  - `twentyone-rl/src/twentyone_rl/agents/deep_mccfr.py` (lines 392-433)

### 2. Added Vanilla MCCFR to Evaluation Suite ✅
**What was missing**: Tournament system only had HeuristicAgent, no proper MCCFR baseline
- **Fix**: Added MCCFRAgent wrapper class for evaluation
- **Benefit**: Can now properly compare Deep MCCFR vs Vanilla MCCFR vs Heuristics
- **Files Changed**: `twentyone-rl/src/twentyone_rl/evaluation/tournament.py` (added MCCFRAgent class)

### 3. Improved Observation Encoding ✅
**What was limited**: 10-dimensional encoding lacked strategic information
- **Old**: Basic state features only (10 dims)
- **New**: Added 6 derived strategic features (16 dims total):
  - Distance to 21
  - Hearts differential (winning/losing)
  - Round pressure
  - Relative position vs opponent
  - Safe draw indicator (total <= 10)
  - Danger zone indicator (17-21)
- **Files Changed**: `twentyone-rl/src/twentyone_rl/agents/deep_mccfr.py` (lines 185-239)

### 4. Fixed Build System ✅
**What was broken**: Hardcoded absolute path in pyproject.toml
- **Issue**: `twentyone @ file:///Users/henrilemoine/...`
- **Fix**: Changed to relative path `twentyone @ file://../twentyone-py`
- **Files Changed**: `twentyone-rl/pyproject.toml` (line 8)

---

## 🔄 NEXT STEPS: Validation and Testing

### 1. Train and Evaluate Fixed Agents
**Priority**: HIGH
- Train vanilla MCCFR with fixed algorithm (suggest 10,000 iterations)
- Train Deep MCCFR with fixed algorithm (suggest 50,000 iterations)
- Run tournament comparing all agents:
  - Heuristic(threshold=17)
  - Vanilla MCCFR (fixed)
  - Deep MCCFR (fixed)
- Document win rates and compare to previous broken results

### 2. Hyperparameter Tuning for Deep MCCFR
**Priority**: MEDIUM
- Current learning rate: 3e-4 (may need adjustment)
- Current architecture: 256→256→256→128 encoder (may be overkill)
- Batch size: 64 (may need tuning)
- Experience buffer size: 100,000 (seems reasonable)
- Consider: Add learning rate scheduling, adjust dropout rates

### 3. Extended Features (Optional)
**Priority**: LOW
- Save/load regret tables for vanilla MCCFR (currently not implemented)
- Add more sophisticated value baselines (e.g., running average)
- Implement CFR+ or Linear CFR variants for faster convergence
- Add tensorboard logging for training visualization
