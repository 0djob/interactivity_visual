# Behavioral Self-Prediction: Teaching AI to Learn Forever

## What is this project?

Imagine trying to predict what you'll think tomorrow. The catch: the act of trying to predict changes what you'll actually think. This creates a paradox where you can never fully predict yourself because you're constantly learning and adapting.

This project explores exactly that. I built an AI agent that tries to predict its own future behavior, creating a system that must keep learning forever because it can never quite catch up to itself.

## The Big Idea

Most AI systems learn until they "solve" a problem, then stop. But what if the problem can never be fully solved? What if the world is always bigger and more complex than the AI's ability to understand it?

This project implements the "big world hypothesis": an intelligent agent should always be learning because the world is fundamentally too complex to ever fully understand.

Think of it like this. A chess AI can "solve" chess (or get close) because chess has fixed rules. But a human can never "solve" life because life keeps changing. I want AI that's more like the second one.

## How To Run It

Click on the live link under the actions tab in this GitHub project. It will open a site and automatically reconnect to the backend.
Note: The steps on the live link version of this project will run much slower, as it is hosted on a free plan, so if you require a faster test, I would advise running it on your own PC

## How It Works

### The Self-Referential Loop

The agent exists in a loop where it observes its own previous behavior (a 64-dimensional vector), predicts what it will do next, acts based on its prediction, learns from how wrong its prediction was, and repeats forever.

The interesting part? The agent can never predict itself perfectly because by the time it makes a prediction, it has already changed (it learned something). It can't see its own "brain" (the neural network weights), and the world responds to its actions in complex ways.

This creates what the paper calls an "implicit constraint": the agent is naturally limited, not because I artificially constrained it, but because of the fundamental structure of the problem.

### The I-Score: Measuring Learning Value

The key metric is called the I-Score (Interactivity Score). It measures how much memory helps predict the future.

Without recent memory: How well can I predict my next 20 steps using only old information?
With recent memory: How well can I predict using my recent experience?
I-Score equals the difference.

If I-Score is high (like 0.27 in my results), it means memory is valuable, the agent is learning useful things, and the world is complex enough to be interesting.

If I-Score drops to zero, the agent has fallen into a "dark room": it found a boring pattern it can perfectly predict, stopped exploring, and stopped learning.

### Fighting the Dark Room

The biggest challenge is preventing the agent from getting lazy. Without intervention, it naturally drifts toward simple, predictable behaviors it can easily predict, like sitting in a dark room doing nothing.

I prevent this with aggressive exploration bonuses:

Curiosity Bonus (weight: 0.5) rewards the agent for being surprised. If you couldn't predict this, that's good. Learn from it.

Novelty Bonus (weight: 0.3) rewards behaviors different from what it's done recently. Don't just repeat yourself.

Entropy Bonus (weight: 0.1) rewards diverse actions. Don't fall into predictable patterns.

Complexity Floor (minimum I-Score: 0.015) punishes the agent if I-Score gets too low. If learning becomes worthless, you're doing something wrong.

Auto-Reset (after 50 low steps): if stuck in a rut, inject chaos and start fresh.

These bonuses are 5-10x stronger than typical exploration in RL, because I need to fight the natural pull toward simplicity.

## The Technical Architecture

### Neural Networks

The agent has three networks:

LSTM (64 hidden units) maintains memory of recent behaviors and allows the agent to use temporal context.

Policy Network decides what action to take, mapping behavior to action probabilities.

Value Network predicts what the next behavior will be. This is the self-prediction part.

### Target Network Trick

I use a target network (a frozen copy) to stabilize learning. The online network learns continuously while the target network updates slowly (soft update with tau=0.01). This prevents the agent from "chasing its own tail" too aggressively.

### The Learning Algorithm

Every 20 steps (the "horizon"), the system collects experiences in a buffer, recomputes predictions for the whole horizon (with fresh gradients), and calculates I-Score:

```
conditional_complexity = error using recent memory
semiconditional_complexity = error using old memory
i_score = (semiconditional - conditional) / horizon
```

Then it optimizes to maximize:

```
Loss = -I-Score - curiosity_bonus - novelty_bonus - entropy_bonus + floor_penalty
```

Finally, it updates networks with gradient descent.

## What Success Looks Like

### Results (900 steps)

```
Step 100: I-Score = 0.183
Step 200: I-Score = 0.255
Step 400: I-Score = 0.278
Step 900: I-Score = 0.268

Average: ~0.27 (maintained for 900 steps)
```

This is good because I-Score doesn't collapse (no dark room), the agent keeps learning for 900 steps, the trend is stable (not diverging), and both complexities stay bounded but separated.

### What the Numbers Mean

Conditional Complexity (~1.7): How hard it is to predict future behavior with recent memory. Lower means memory is helping.

Semi-conditional Complexity (~2.0): How hard it is to predict without recent memory. Higher means the future is complex.

The Gap (~0.27): How much memory reduces prediction error. This is the value of learning.

Prediction Distances (2.5-6.5): L2 distance between predicted and actual behavior. Greater than zero means can't perfectly predict (which is good). Memory helps slightly (old distance greater than current distance).

## Why This Matters

### Philosophically

This project touches on deep questions. Can intelligence exist without curiosity? Is learning just compression, or something more? What makes an agent "keep going" when there's no external goal?

### Practically

The techniques here could help build AI that explores naturally without external rewards, adapts continuously to changing environments, never stops learning (lifelong learning), and avoids local optima (dark rooms).

### Scientifically

I've demonstrated stable behavioral self-prediction, sustained interactivity for 900 steps, implicit capacity constraint (can't perfectly self-predict), continual adaptation (agent doesn't converge), and anti-dark-room mechanisms that work.

## Problems Encountered

Problem 1: Gradient Errors
Issue: "Modified by in-place operation" errors
Solution: Recompute forward passes with fresh gradients instead of storing TD errors

Problem 2: Dark Room Collapse
Issue: I-Score dropping to zero after 100 steps
Solution: Aggressive exploration bonuses (5x normal strength)

Problem 3: 20-Step Oscillations
Issue: I-Score crashing every 20 steps
Solution: Soft target updates (tau=0.01) instead of hard sync

Problem 4: Slow Convergence
Issue: I-Score slowly declining despite bonuses
Solution: Adaptive weights + auto-reset mechanism

Each problem taught me something about the nature of self-prediction and continual learning.

The agent will start with high variance (exploring), stabilize around I-Score = 0.25-0.28, occasionally reset itself if stuck, and maintain learning for as long as you run it.

## Files in This Project

Core Implementation:
- behavioral_core_AGGRESSIVE.py (the main agent)
- behavior_space_visualizer.py (visualization wrapper with PCA, trajectories)
- behavioral_core_GRADIENT_FIX.py (previous version with gradient bug fix)

Documentation:
- DARK_ROOM_SOLUTION.md (complete guide to the dark room problem)
- OSCILLATION_EXPLAINED.md (why I-Score oscillates every 20 steps)
- SOLUTION_COMPARISON.md (comparison of all solution attempts)
- EXACT_FIXES.md (step-by-step patches and fixes)

Experiments:
- SOFT_TARGET_UPDATE_PATCH.py (smooths oscillations)
- behavioral_core_enhanced.py (version with cellular automaton environment)

## The Paper This Implements

"The World Is Bigger: A Computationally-Embedded Perspective on the Big World Hypothesis"
Lewandowski et al., 2024

My implementation closely follows Section 5-6: Maximizing interactivity with RL and Behavioral self-prediction benchmark.

I've achieved more stable results than the paper's Figure 4 by adding aggressive exploration mechanisms.

## Future Directions

What could make this better:

Richer Environment: Add cellular automaton boundary (like Conway's Game of Life). Agent interacts with complex external dynamics. Should increase I-Score to 0.4-0.5+.

Hierarchical Behaviors: Multiple timescales (fast and slow behaviors). Should capture more complex patterns.

Scaling Up: Larger networks (128, 256 hidden units). Longer horizons (50, 100 steps). Should maintain higher I-Score.

Meta-Learning: Learn exploration strategies themselves. Adapt bonuses automatically.

Multi-Agent: Multiple agents predicting each other. Social dynamics emerge.

## License

MIT License. Feel free to use this code for research or learning.

## Acknowledgments

Original paper: Lewandowski, Ramesh, Meyer, Schuurmans, Machado (2024)
Inspiration: Active inference, intrinsic motivation, continual learning research
Built with: PyTorch and a lot of debugging

## Final Thoughts

This project shows that you can build an AI that wants to keep learning, not because you programmed a goal, but because the structure of the problem makes learning inherently valuable.

The agent is always chasing something it can never quite catch: itself. And in that eternal chase, it must continually adapt, explore, and grow.
