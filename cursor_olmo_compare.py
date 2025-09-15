# cursor_olmo_compare.py
# Requires: pip install ai2-olmo transformers accelerate torch

from hf_olmo import OLMoForCausalLM, OLMoTokenizerFast
import torch
import os

def check_disk_space():
    """Check available disk space in GB"""
    statvfs = os.statvfs('/')
    free_gb = (statvfs.f_frsize * statvfs.f_bavail) / (1024**3)
    return free_gb

def query_model(model_name: str, question: str, n: int = 10, max_new_tokens: int = 20):
    print(f"Loading {model_name}...")
    tokenizer = OLMoTokenizerFast.from_pretrained(model_name)
    model = OLMoForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16).to("cuda" if torch.cuda.is_available() else "cpu")

    outputs = []
    for i in range(n):
        print(f"  Running trial {i+1}/{n}...")
        inputs = tokenizer(question, return_tensors="pt").to(model.device)
        with torch.no_grad():
            resp = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,      # sampling to let variation show
                top_p=0.95,
                temperature=0.7,
            )
        text = tokenizer.decode(resp[0], skip_special_tokens=True)
        outputs.append(text)

    return outputs


if __name__ == "__main__":
    # Check disk space first
    free_gb = check_disk_space()
    print(f"Available disk space: {free_gb:.1f} GB")
    
    if free_gb < 30:
        print("⚠️  WARNING: Low disk space detected!")
        print("OLMo-7B requires ~28GB. Consider using smaller models or freeing up space.")
        print("\nAlternative smaller models to try:")
        print("- allenai/OLMo-1B (requires ~2GB)")
        print("- allenai/OLMo-1.7-1B (requires ~2GB)")
        print("- allenai/OLMo-1.7-3B (requires ~6GB)")
        
        # Use smaller models instead
        question = 'Is a "baby stroller" a "vehicle"?'
        
        # Smaller OLMo models that should fit
        models_to_test = {
            "OLMo-1B": "allenai/OLMo-1B",
            "OLMo-1.7-1B": "allenai/OLMo-1.7-1B", 
        }
        
        results = {}
        for model_name, model_path in models_to_test.items():
            try:
                print(f"\n🔄 Testing {model_name}...")
                results[model_name] = query_model(model_path, question, n=5)  # Reduced trials
            except Exception as e:
                print(f"❌ Failed to load {model_name}: {e}")
                continue
        
        # Log responses
        for model_name, answers in results.items():
            print(f"\n=== {model_name} ===")
            for i, ans in enumerate(answers, 1):
                print(f"Run {i}: {ans}")
                
    else:
        # Original code for when there's enough space
        question = 'Is a "baby stroller" a "vehicle"?'

        # Early OLMo-7B (main branch)
        olmo_7b = "allenai/OLMo-7B"
        # Newer OLMo-1.7-7B (April 2024 release)
        olmo_17 = "allenai/OLMo-1.7-7B"

        results = {
            "OLMo-7B (early)": query_model(olmo_7b, question),
            "OLMo-1.7-7B (Apr 2024)": query_model(olmo_17, question),
        }

        # Log responses
        for model_name, answers in results.items():
            print(f"\n=== {model_name} ===")
            for i, ans in enumerate(answers, 1):
                print(f"Run {i}: {ans}")
