import torch
import torch.nn as nn
import numpy as np
from typing import Tuple, Dict, List
from collections import deque
from .behavioral_core import BehavioralSelfPredictor

class SimplePCA:
    
    def __init__(self, n_components: int = 2):
        self.n_components = n_components
        self.mean = None
        self.components = None
        
    def fit(self, X: np.ndarray):
        self.mean = np.mean(X, axis=0)
        X_centered = X - self.mean
        
        cov_matrix = np.cov(X_centered.T)
        
        eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)
        
        idx = eigenvalues.argsort()[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]
        
        self.components = eigenvectors[:, :self.n_components].T
        
        return self
    
    def transform(self, X: np.ndarray) -> np.ndarray:
        if self.mean is None or self.components is None:
            raise ValueError("PCA must be fitted before transform")
        
        X_centered = X - self.mean
        return np.dot(X_centered, self.components.T).real
    
    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        """Fit and transform in one step."""
        self.fit(X)
        return self.transform(X)

class BehaviorSpaceVisualizer:
    def __init__(self, behavior_size: int = 64, horizon: int = 20):
        self.core = BehavioralSelfPredictor(
            behavior_size=behavior_size,
            horizon=horizon
        )
        
        self.behavior_size = behavior_size
        self.horizon = horizon
        
        self.behavior_history = deque(maxlen=200)
        
        self.pca = SimplePCA(n_components=2)
        self.pca_fitted = False
        
        self.previous_behavior = torch.randn(behavior_size) * 0.3
        
        self.prediction_history = deque(maxlen=horizon + 5)
        
        self.step_count = 0
        
    def step(self) -> Dict:
        core_result = self.core.step(self.previous_behavior)
        
        actual_behavior_vector = core_result['behavior']
        current_prediction_vector = core_result['value_prediction']
        i_score = core_result['i_score']
        
        self.behavior_history.append(actual_behavior_vector.detach().numpy())
        self.prediction_history.append(current_prediction_vector.detach().numpy())
        
        distances = self._calculate_prediction_distances(
            actual_behavior_vector,
            current_prediction_vector
        )
        
        trajectory_2d = None
        old_pred_2d = None
        current_pred_2d = None
        actual_2d = None
        
        if len(self.behavior_history) >= 50:
            trajectory_2d, old_pred_2d, current_pred_2d, actual_2d = \
                self._project_to_2d(actual_behavior_vector, distances)
        
        loss = core_result['learning_loss']
        
        self.previous_behavior = actual_behavior_vector.detach()
        self.step_count += 1
        
        return {
            'i_score': i_score,
            'conditional_complexity': core_result['conditional_complexity'],
            'semiconditional_complexity': core_result['semiconditional_complexity'],
            'learning_loss': loss,
            'behavior_vector': actual_behavior_vector.detach().numpy(),
            'value_prediction': current_prediction_vector.detach().numpy(),
            'distance_old_to_actual': distances['old_to_actual'],
            'distance_current_to_actual': distances['current_to_actual'],
            'distance_difference': distances['difference'],
            'old_prediction_vector': distances['old_prediction'],
            'current_prediction_vector': distances['current_prediction'],
            'actual_vector': actual_behavior_vector.detach().numpy(),
            'trajectory_2d': trajectory_2d,
            'old_prediction_2d': old_pred_2d,
            'current_prediction_2d': current_pred_2d,
            'actual_position_2d': actual_2d,
            'metrics': self.core.get_metrics(),
            'step': self.step_count,
            'pca_ready': self.pca_fitted
        }
    
    def _calculate_prediction_distances(
        self,
        actual_behavior: torch.Tensor,
        current_prediction: torch.Tensor
    ) -> Dict:
        if len(self.prediction_history) >= self.horizon:
            old_prediction = self.prediction_history[-self.horizon]
        else:
            old_prediction = current_prediction.detach().numpy()
        
        actual = actual_behavior.detach().numpy()
        current = current_prediction.detach().numpy()
        
        distance_old = np.linalg.norm(old_prediction - actual)
        distance_current = np.linalg.norm(current - actual)
        
        difference = distance_old - distance_current
        
        return {
            'old_prediction': old_prediction,
            'current_prediction': current,
            'old_to_actual': float(distance_old),
            'current_to_actual': float(distance_current),
            'difference': float(difference)
        }
    
    def _project_to_2d(
        self,
        actual_behavior: torch.Tensor,
        distances: Dict
    ) -> Tuple:
        """
        Project 64D behavior space to 2D using PCA for visualization.
        """
        
        if not self.pca_fitted:
            if len(self.behavior_history) >= 50:
                history_array = np.array(list(self.behavior_history))
                self.pca.fit(history_array)
                self.pca_fitted = True
            else:
                return None, None, None, None
        
        
        history_array = np.array(list(self.behavior_history))
        trajectory_2d = self.pca.transform(history_array)
        
        old_pred_2d = self.pca.transform(np.array([distances['old_prediction']]))[0]
        current_pred_2d = self.pca.transform(np.array([distances['current_prediction']]))[0]
        actual_2d = self.pca.transform(np.array([actual_behavior.detach().numpy()]))[0]
        
        return (
            trajectory_2d.tolist(),
            old_pred_2d.tolist(),
            current_pred_2d.tolist(),
            actual_2d.tolist()
        )
    
    def get_full_state(self) -> Dict:
        """Get complete state for frontend."""
        metrics = self.core.get_metrics()
        
        latest_behavior = None
        if len(self.behavior_history) > 0:
            latest_behavior = list(self.behavior_history)[-1].tolist()
        
        return {
            'behavior_vector': latest_behavior,
            'i_score': metrics['current_i_score'],
            'i_score_history': metrics['i_score_history'],
            'conditional_history': metrics['conditional_history'],
            'semiconditional_history': metrics['semiconditional_history'],
            'i_score_trend': metrics['i_score_trend'],
            'step': self.step_count,
            'pca_ready': self.pca_fitted
        }


if __name__ == "__main__":
    visualizer = BehaviorSpaceVisualizer(behavior_size=64, horizon=20)
    
    for step in range(200):
        result = visualizer.step()
        
        if (step + 1) % 20 == 0:
            print(f"Step {step+1:3d}")
            print(f"  I-Score:                 {result['i_score']:7.4f}")
            print(f"  Avg Trend:               {result['metrics']['i_score_trend']:7.4f}")
            print(f"  Distance (old → actual): {result['distance_old_to_actual']:7.4f}")
            print(f"  Distance (now → actual): {result['distance_current_to_actual']:7.4f}")
            print(f"  Difference (geometric):  {result['distance_difference']:7.4f}")
            
            if result['pca_ready']:
                print(f"  2D Position:             ({result['actual_position_2d'][0]:6.2f}, "
                      f"{result['actual_position_2d'][1]:6.2f})")
            print()
    
    print(f"Final Statistics:")
    print(f"  I-Score:        {result['i_score']:.4f}")
    print(f"  Average trend:  {result['metrics']['i_score_trend']:.4f}")
    print(f"  Steps run:      {result['step']}")
    if result['trajectory_2d']:
      print(f"  Trajectory has: {len(result['trajectory_2d'])} points")