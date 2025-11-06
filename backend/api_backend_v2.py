import asyncio
import torch
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn

from behavioral_core import BehavioralSelfPredictor
from behavior_space_visualizer import BehaviorSpaceVisualizer

app = FastAPI(title="Interactivity Maximization API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global State: Simplified to only core agent and config
agent_instance = None
BEHAVIOR_SIZE = 64
HORIZON = 10

class ConfigUpdate(BaseModel):
    horizon: int = 10
    behavior_size: int = 64

class ManualAction(BaseModel):
    action_type: str
    direction: int

@app.get("/")
async def root():
    return {
        "service": "Interactivity Maximization API",
        "version": "2.0",
        "paper": "The World Is Bigger: Big World Hypothesis",
        "endpoints": {
            "config": "POST /config - Update agent configuration",
            "websocket": "WS /ws/agent - Real-time agent updates"
        }
    }

@app.post("/config")
async def update_config(config: ConfigUpdate):
    global HORIZON, BEHAVIOR_SIZE
    HORIZON = config.horizon
    BEHAVIOR_SIZE = config.behavior_size
    return {
        "status": "success",
        "horizon": HORIZON,
        "behavior_size": BEHAVIOR_SIZE
    }

@app.get("/metrics")
async def get_metrics():
    if agent_instance is None:
        return {"error": "No agent running"}
    
    return agent_instance.get_full_state()

@app.websocket("/ws/agent")
async def websocket_agent(websocket: WebSocket):
    await websocket.accept()
    
    global agent_instance
    
    print("\n" + "="*60)
    print(" NEW WEBSOCKET CONNECTION")
    print("="*60 + "\n")
    
    
    agent_instance = BehaviorSpaceVisualizer(
        behavior_size=BEHAVIOR_SIZE,
        horizon=HORIZON
    )
    
    
    step_count = 0
    i_score_sum = 0.0
    
    try:
        while True:
            
            result = agent_instance.step()
            step_count += 1
            i_score_sum += result['i_score']
            
            
            message = _format_behavior_space_message(result, step_count, i_score_sum)
            
            
            await websocket.send_json(message)
            
            
            
            if step_count % 50 == 0:
                _log_progress(result, step_count, i_score_sum)
            
            
            await asyncio.sleep(0.05) 
            
    except WebSocketDisconnect:
        print(f"\n WebSocket disconnected after {step_count} steps")
        print(f" Final I-Score: {result['i_score']:.4f}")
        print(f" Average I-Score: {i_score_sum/step_count:.4f}\n")
        agent_instance = None
        
    except Exception as e:
        print(f"\n WebSocket error: {e}")
        import traceback
        traceback.print_exc()
        await websocket.close(code=1011)
        agent_instance = None


def _format_behavior_space_message(result: dict, step_count: int, i_score_sum: float) -> dict:
    return {
        "algorithm": {
            "i_score": result['i_score'],
            "conditional_complexity": result['conditional_complexity'],
            "semiconditional_complexity": result['semiconditional_complexity'],
            "learning_loss": result['learning_loss'],
            "behavior_vector": result['behavior_vector'].flatten().round(decimals=5).tolist(),
            "value_prediction": result['value_prediction'].flatten().round(decimals=5).tolist(),
            "i_score_history": result['metrics']['i_score_history'],
            "conditional_history": result['metrics']['conditional_history'],
            "semiconditional_history": result['metrics']['semiconditional_history'],
            "i_score_trend": result['metrics']['i_score_trend'],
            "average_i_score": i_score_sum / step_count,
        },
        "visualization": {
            "type": "behavior_space",
            
            
            "behavior_vector": result['behavior_vector'].flatten().round(decimals=5).tolist(),
            "old_prediction": result['old_prediction_vector'].flatten().round(decimals=5).tolist(),
            "current_prediction": result['current_prediction_vector'].flatten().round(decimals=5).tolist(),
            
            
            "distance_old_to_actual": result['distance_old_to_actual'],
            "distance_current_to_actual": result['distance_current_to_actual'],
            "distance_difference": result['distance_difference'],
            
            
            "trajectory_2d": result['trajectory_2d'],
            "old_prediction_2d": result['old_prediction_2d'],
            "current_prediction_2d": result['current_prediction_2d'],
            "actual_position_2d": result['actual_position_2d'],
            "pca_ready": result['pca_ready'],
        },
        
        
        "meta": {
            "step": step_count,
            "timestamp": asyncio.get_event_loop().time()
        }
    }


def _log_progress(result: dict, step_count: int, i_score_sum: float):
    print(f"\n STEP {step_count}")
    print(f" I-Score: {result['i_score']:.4f}")
    print(f" Avg I-Score: {i_score_sum/step_count:.4f}")
    print(f" Conditional: {result['conditional_complexity']:.4f}")
    print(f" Semi-conditional: {result['semiconditional_complexity']:.4f}")
    print(f" Distance (old→actual): {result['distance_old_to_actual']:.4f}")
    print(f" Distance (now→actual): {result['distance_current_to_actual']:.4f}")
    print(f" Geometric I-Score: {result['distance_difference']:.4f}")


@app.post("/experiment/freeze_learning")
async def freeze_learning_experiment():
    if agent_instance is None:
        return {"error": "No agent running. Start WebSocket first."}
    
    print("\n" + "="*60)
    print(" FREEZE LEARNING EXPERIMENT")
    print("="*60)
    
    
    baseline_metrics = agent_instance.core.get_metrics()
    baseline_i_score = baseline_metrics['current_i_score']
    agent_to_freeze = agent_instance.core
    
    
    original_zero_grad = agent_to_freeze.optimizer.zero_grad
    original_step = agent_to_freeze.optimizer.step
    
    agent_to_freeze.optimizer.zero_grad = lambda: None
    agent_to_freeze.optimizer.step = lambda: None
    
    print(f" Learning frozen")
    print(f" Baseline I-Score: {baseline_i_score:.4f}")
    print(" Running 100 more steps with frozen parameters...")
    
    frozen_i_scores = []
    for i in range(100):
        result = agent_instance.step()
        frozen_i_scores.append(result['i_score'])
        
        if i % 20 == 0:
            print(f" Step {i}: I-Score = {result['i_score']:.4f}")
    
    avg_frozen_i_score = sum(frozen_i_scores) / len(frozen_i_scores)
    drop = baseline_i_score - avg_frozen_i_score
    
    agent_to_freeze.optimizer.zero_grad = original_zero_grad
    agent_to_freeze.optimizer.step = original_step
    
    print(f" Average I-Score after freeze: {avg_frozen_i_score:.4f}")
    print(f" Drop: {drop:.4f} ({(drop/max(baseline_i_score, 0.0001))*100:.1f}%)")
    print(" Learning restored")
    print("="*60 + "\n")
    
    return {
        "experiment": "freeze_learning",
        "baseline_i_score": baseline_i_score,
        "frozen_i_scores": frozen_i_scores,
        "average_frozen_i_score": avg_frozen_i_score,
        "drop": drop,
        "drop_percentage": (drop/max(baseline_i_score, 0.0001))*100,
        "conclusion": f"I-Score dropped by {drop:.4f} ({(drop/max(baseline_i_score, 0.0001))*100:.1f}%), proving agent must keep learning (Theorem 2)"
    }


@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "agent_running": agent_instance is not None
    }

@app.get("/status")
async def get_status():
    if agent_instance is None:
        return {
            "agent_running": False,
            "message": "No agent instance. Connect via WebSocket to start."
        }
    
    metrics = agent_instance.core.get_metrics()
    return {
        "agent_running": True,
        "current_turn": agent_instance.turn,
        "current_i_score": metrics['current_i_score'],
        "i_score_trend": metrics['i_score_trend']
    }


if __name__ == "__main__":
    print("\n" + "="*60)
    print("STARTING INTERACTIVITY MAXIMIZATION SERVER")
    print("="*60)
    print("\n Endpoints:")
    print(" • REST API:      http://localhost:8000")
    print(" • Docs:          http://localhost:8000/docs")
    print(" • WebSocket:     ws://localhost:8000/ws/agent")
    print(" • Health:        http://localhost:8000/health")
    print("\n" + "="*60 + "\n")
    
    uvicorn.run(
        "api_backend_v2:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )