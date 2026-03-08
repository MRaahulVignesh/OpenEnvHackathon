# 🚀 QUICK START - Baseline Evaluation

## Run This in Visual Studio Code Terminal

### 1️⃣ Open VS Code
```
Open Visual Studio Code
File → Open Folder → /Users/vijay/Desktop/OpenEnvHackathon-baseline
```

### 2️⃣ Open Terminal
```
Terminal → New Terminal
(or press Ctrl + ` on keyboard)
```

### 3️⃣ Download Model (First Time Only)
```bash
python training/download_model.py
```
⏱️ **Takes**: 5-10 minutes (downloads ~6GB)

### 4️⃣ Run Baseline Evaluation
```bash
python training/baseline_eval.py
```
⏱️ **Takes**: 15-30 minutes (evaluates 22 scenarios)

### 5️⃣ Check Results
Open the file: `baseline_results.json`

---

## ✅ Expected Output

You should see something like:

```
============================================================
BASELINE — Qwen/Qwen2.5-3B-Instruct
============================================================
  Scenarios:     22
  Avg score:     66.1%
  Pass@1 (100%): 13.6%

  BANKING       72.5%  (8 scenarios)
  CONSULTING    65.7%  (7 scenarios)
  LAW           62.9%  (7 scenarios)

✓ Saved to baseline_results.json
```

---

## 🆘 If Something Goes Wrong

### Error: "CUDA out of memory"
**Run this first to clear GPU:**
```bash
python -c "import torch; torch.cuda.empty_cache(); print('GPU cleared')"
```

### Error: "Model not found"
**Download the model:**
```bash
python training/download_model.py
```

### Error: "No module named 'transformers'"
**Install dependencies:**
```bash
pip install transformers torch accelerate python-dotenv huggingface_hub
```

### Error: "Scenarios not found"
**Make sure you're in the right directory:**
```bash
cd /Users/vijay/Desktop/OpenEnvHackathon-baseline
python training/baseline_eval.py
```

---

## 📂 Files You'll Create

After running, you'll have:

```
OpenEnvHackathon-baseline/
├── baseline_results.json    ← Your evaluation results
└── models/
    └── base_model/
        └── Qwen2.5-3B-Instruct/  ← Downloaded model (~6GB)
```

---

## 🎯 Next Steps After Baseline

1. **Note your baseline score** (e.g., 66.1%)

2. **Train the model:**
   ```bash
   # Terminal 1
   python main.py --port 8001

   # Terminal 2
   python training/train_grpo.py
   ```

3. **Evaluate trained model:**
   Update `.env`:
   ```
   MODEL_NAME=Qwen2.5-3B-Instruct_v0
   ```

   Then run:
   ```bash
   python training/baseline_eval.py
   ```

4. **Compare results:**
   - Baseline: 66.1%
   - After training: ???%
   - Improvement: ??? percentage points

---

## 💡 Pro Tips

- **Save your baseline results** before training
- **GPU memory**: Make sure you have 50GB+ free
- **Check GPU**: Run `nvidia-smi` to monitor usage
- **Patience**: Evaluation takes 15-30 minutes - don't interrupt!

---

## 📊 What Gets Evaluated?

The script tests your model on:

- **8 Banking scenarios** (M&A, valuation, risk assessment)
- **7 Consulting scenarios** (strategy, operations, pricing)
- **7 Law scenarios** (contracts, compliance, IP)

Each scenario has:
- Multiple files to analyze
- A specific task
- A rubric with 5 criteria
- Binary scoring (0 or 1 per criterion)

---

## ✨ That's It!

Just run:
```bash
python training/baseline_eval.py
```

And wait for your results! 🎉
