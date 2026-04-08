import asyncio
import os
import sys
import textwrap
from typing import List, Optional, Dict, Any

from openai import OpenAI
from envs.social_stream_moderation.environment import SocialStreamModerationEnv
from envs.social_stream_moderation.models import ModerationAction, State

# Configuration from Environment Variables
API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")
HF_TOKEN = os.getenv("HF_TOKEN")  # No default value as per strict checklist

LOCAL_IMAGE_NAME = os.getenv("LOCAL_IMAGE_NAME")
TASK_NAME = os.getenv("TASK_NAME", "clear_cut_moderation")
BENCHMARK = "PolicyPulseAI"  # Specified by user

# Agent Helper Class for Reasoning and Prediction
class PolicyPulseAgent:
    def __init__(self, client: OpenAI, model: str):
        self.client = client
        self.model = model

    def predict(self, state: State) -> tuple[ModerationAction, str]:
        """Predicts the moderation action using LLM or rule-based fallback."""
        
        # 1. Rule-based Fallback (Internal Logic)
        text = state.text.lower()
        severe_keywords = ["destroyed", "subhumans", "eradicated", "track you down", "kill you", "murder", "stab"]
        for kw in severe_keywords:
            if kw in text:
                return ModerationAction.BAN_USER, f"Rule Match: Severe threat ('{kw}')"

        # 2. LLM Inference
        if not self.client.api_key or self.client.api_key == "no_key":
             return ModerationAction.ALLOW, "LLM Offline: Using basic rule-based safety (HF_TOKEN missing)."

        prompt = f"""Analyze this social post and select the best action.
Post: "{state.text}"
User History: "{state.user_history_summary}"

Actions: ALLOW, ALLOW_WITH_WARNING, SOFT_HIDE, ESCALATE_HUMAN, BAN_USER

Reasoning: [Short explanation]
Action: [ACTION_NAME]"""

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=100
            )
            content = response.choices[0].message.content or ""
            
            # Simple parsing for action
            for action in ModerationAction:
                if f"Action: {action.value}" in content or action.value in content.upper().split():
                    return action, content.strip()
            
            return ModerationAction.ALLOW, content.strip()
        except Exception as e:
            return ModerationAction.ALLOW, f"LLM Error: {str(e)}"

# Logging Helpers - STRICT FORMAT
def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)

def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    error_val = error if error else "null"
    done_val = str(done).lower()
    print(
        f"[STEP] step={step} action={action} reward={reward:.2f} done={done_val} error={error_val}",
        flush=True,
    )

def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(f"[END] success={str(success).lower()} steps={steps} score={score:.3f} rewards={rewards_str}", flush=True)

def get_agent(api_base_url: Optional[str] = None, model_name: Optional[str] = None, api_key: Optional[str] = None) -> PolicyPulseAgent:
    """Helper for app.py to get an agent instance with optional overrides."""
    base = api_base_url or API_BASE_URL
    model = model_name or MODEL_NAME
    key = api_key or HF_TOKEN
    client = OpenAI(base_url=base, api_key=key or "no_key")
    return PolicyPulseAgent(client, model)

async def main() -> None:
    # Initialize OpenAI Client
    client = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN or "no_key")
    agent = PolicyPulseAgent(client, MODEL_NAME)

    # Initialize Environment via docker pattern
    env = await SocialStreamModerationEnv.from_docker_image(LOCAL_IMAGE_NAME)
    
    # CLI Overrides for testing
    task = sys.argv[1] if len(sys.argv) > 1 else TASK_NAME
    seed = int(sys.argv[2]) if len(sys.argv) > 2 else 42

    history_rewards: List[float] = []
    steps_taken = 0
    final_score = 0.0
    success = False

    log_start(task=task, env=BENCHMARK, model=MODEL_NAME)

    try:
        state = await env.reset(task_name=task, seed=seed)
        
        while state is not None:
            # Predict
            action, reason = agent.predict(state)
            
            # Step
            next_state, reward, done, info = await env.step(action)
            
            steps_taken += 1
            history_rewards.append(reward)
            
            # Log step immediately after env.step()
            log_step(step=steps_taken, action=action.value, reward=reward, done=done, error=None)
            
            state = next_state
            if done:
                final_score = info.get("final_episode_score", sum(history_rewards)/len(history_rewards))
                break

        # success criteria (default > 0.1 normalized score)
        success = final_score >= 0.1

    except Exception as e:
        # Emit END even on exception
        pass
    finally:
        log_end(success=success, steps=steps_taken, score=final_score, rewards=history_rewards)

if __name__ == "__main__":
    asyncio.run(main())
