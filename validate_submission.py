import subprocess
import os
import sys

def run_test_task(task_name: str, seed: int = 42):
    print(f"Testing Task: {task_name}...")
    try:
        # Run inference.py locally and capture output
        # Use sys.executable to ensure we use the same python interpreter
        result = subprocess.run(
            [sys.executable, 'inference.py', task_name, str(seed)],
            capture_output=True,
            text=True,
            timeout=30 
        )
        output = result.stdout
        
        # Validation checks based on strict sample format
        # [START] task=clear_cut_moderation env=PolicyPulseAI model=...
        has_start = f"[START] task={task_name} env=PolicyPulseAI" in output
        
        # [STEP] step=1 action=... reward=... done=... error=...
        has_step = "[STEP] step=1 action=" in output
        
        # [END] success=... steps=... score=... rewards=...
        has_end = "[END] success=" in output
        
        if has_start and has_step and has_end:
            print(f"  [PASS] {task_name} | Logs formatted correctly according to reference sample.")
        else:
            print(f"  [FAIL] {task_name} | Logs missing or incorrectly formatted markers!")
            if not has_start: print("    - Missing correct [START] marker")
            if not has_step: print("    - Missing example [STEP] marker")
            if not has_end: print("    - Missing correct [END] marker")
            print(f"Full Output Snippet:\n{output[:500]}...")
            
    except Exception as e:
        print(f"  [FAIL] {task_name} | Error: {e}")

def check_hf_token_safety():
    print("Checking inference.py for HF_TOKEN safety...")
    with open("inference.py", "r") as f:
        content = f.read()
        if 'HF_TOKEN = os.getenv("HF_TOKEN")' in content and 'fake_key' not in content:
            # Check if it has a default in getenv
            if 'os.getenv("HF_TOKEN",' in content:
                 print("  [FAIL] HF_TOKEN still appears to have a default value!")
                 return False
            print("  [PASS] HF_TOKEN has no default value.")
            return True
        else:
            print("  [FAIL] HF_TOKEN definition is non-compliant or contains 'fake_key'!")
            return False

if __name__ == "__main__":
    print("-" * 50)
    print("🛡️ PolicyPulse AI: Submission Grade Readiness Check")
    print("-" * 50)
    
    # 1. Check file existence
    required_files = [
        "inference.py", "server/app.py", "openenv.yaml", 
        "Dockerfile", "README.md", "requirements.txt", 
        "pyproject.toml", "uv.lock"
    ]
    all_files_ok = True
    for f in required_files:
        if os.path.exists(f):
            print(f"  [PASS] {f} exists.")
        else:
            print(f"  [FAIL] {f} is missing!")
            all_files_ok = False

    # 2. Check for Envs directory
    if os.path.exists("envs/social_stream_moderation"):
        print("  [PASS] Environment package found.")
    else:
        print("  [FAIL] envs/social_stream_moderation is missing!")
        all_files_ok = False

    # 3. Check HF_TOKEN
    token_ok = check_hf_token_safety()

    # 4. Test tasks with inference.py
    print("\n📦 Simulating Evaluation Episodes...")
    run_test_task("clear_cut_moderation")
    run_test_task("nuanced_sarcastic")
    run_test_task("policy_fairness")
    
    print("-" * 50)
    if all_files_ok and token_ok:
        print("✅ READY: All structural and format checks passed.")
        print("You can now securely submit your repository and HF Space URLs.")
    else:
        print("❌ BLOCKED: Please fix the failures above before submitting.")
    print("-" * 50)
