# Behavioral Self-Prediction: Teaching AI to Learn Forever

## What is this project?

Imagine you're trying to predict what you'll think tomorrow. But here's the catch: the act of *trying* to predict changes what you'll actually think! This creates a beautiful paradox—you can never fully predict yourself because you're constantly learning and adapting.

That's what this project explores. We've built an AI agent that tries to predict its own future behavior, creating a system that *must* keep learning forever because it can never quite catch up to itself.

## The Big Idea (In Plain English)

Most AI systems learn until they "solve" a problem, then stop. But what if the problem can never be fully solved? What if the world is always bigger and more complex than the AI's ability to understand it?

This project implements the **"big world hypothesis"**—the idea that an intelligent agent should *always* be learning because the world is fundamentally too complex to ever fully understand.

Think of it like this:
- A chess AI can "solve" chess (or get close) because chess has fixed rules
- But a human can never "solve" life because life keeps changing
- We want AI that's more like the second one

## How It Works

### The Self-Referential Loop

Our agent exists in a loop where:

1. **It observes** its own previous behavior (a 64-dimensional vector)
2. **It predicts** what it will do next
3. **It acts** based on its prediction
4. **It learns** from how wrong its prediction was
5. Repeat forever

The beautiful part? The agent can never predict itself perfectly because:
- By the time it makes a prediction, it has already changed (it learned something!)
- It can't see its own "brain" (the neural network weights)
- The world responds to its actions in complex ways

This creates what the paper calls an **"implicit constraint"**—the agent is naturally limited, not because we artificially constrained it, but because of the fundamental structure of the problem.

### The I-Score: Measuring Learning Value

The key metric is called the **I-Score** (Interactivity Score). It measures:

**"How much does memory help me predict the future?"**

Here's the math in plain English:
- **Without recent memory:** How well can I predict my next 20 steps using only old information?
- **With recent memory:** How well can I predict using my recent experience?
- **I-Score = The difference**

If I-Score is high (like 0.27 in our results), it means:
- Memory is valuable ✅
- The agent is learning useful things ✅
- The world is complex enough to be interesting ✅

If I-Score drops to zero, the agent has fallen into a "dark room":
- It found a boring pattern it can perfectly predict ❌
- It stopped exploring ❌
- It stopped learning ❌

### Fighting the Dark Room

The biggest challenge is preventing the agent from getting lazy. Without intervention, it naturally drifts toward simple, predictable behaviors it can easily predict—like sitting in a dark room doing nothing.

We prevent this with **aggressive exploration bonuses**:

1. **Curiosity Bonus** (weight: 0.5)
   - Rewards the agent for being surprised
   - "If you couldn't predict this, that's good! Learn from it!"

2. **Novelty Bonus** (weight: 0.3)
   - Rewards behaviors different from what it's done recently
   - "Don't just repeat yourself—try something new!"

3. **Entropy Bonus** (weight: 0.1)
   - Rewards diverse actions
   - "Don't fall into predictable patterns!"

4. **Complexity Floor** (minimum I-Score: 0.015)
   - Punishes the agent if I-Score gets too low
   - "If learning becomes worthless, you're doing something wrong!"

5. **Auto-Reset** (after 50 low steps)
   - If stuck in a rut, inject chaos and start fresh
   - "Can't escape? Let's shake things up!"

These bonuses are 5-10x stronger than typical exploration in RL, because we need to fight the natural pull toward simplicity.

## The Technical Architecture

### Neural Networks

The agent has three networks:

1. **LSTM (64 hidden units)**
   - Maintains memory of recent behaviors
   - Allows the agent to use temporal context

2. **Policy Network**
   - Decides what action to take
   - Maps: behavior → action probabilities

3. **Value Network**
   - Predicts: "What will my next behavior be?"
   - This is the self-prediction part!

### Target Network Trick

We use a **target network** (a frozen copy) to stabilize learning:
- **Online network:** Learns continuously
- **Target network:** Updates slowly (soft update with tau=0.01)

This prevents the agent from "chasing its own tail" too aggressively.

### The Learning Algorithm

Every 20 steps (the "horizon"):

1. **Collect experiences** in a buffer
2. **Recompute predictions** for the whole horizon (with fresh gradients)
3. **Calculate I-Score:**
   ```
   conditional_complexity = error using recent memory
   semiconditional_complexity = error using old memory
   i_score = (semiconditional - conditional) / horizon
   ```
4. **Optimize** to maximize:
   ```
   Loss = -I-Score - curiosity_bonus - novelty_bonus - entropy_bonus + floor_penalty
   ```
5. **Update networks** with gradient descent

## What Success Looks Like

### Our Results (900 steps)

```
Step 100: I-Score = 0.183  ✅ Starting well
Step 200: I-Score = 0.255  ✅ Increasing!
Step 400: I-Score = 0.278  ✅ Stabilizing
Step 900: I-Score = 0.268  ✅ Sustained!

Average: ~0.27 (maintained for 900 steps)
```

This is **excellent** because:
- I-Score doesn't collapse (no dark room!)
- Agent keeps learning for 900 steps
- The trend is stable (not diverging)
- Both complexities stay bounded but separated

### What the Numbers Mean

**Conditional Complexity (~1.7):**
- How hard it is to predict future behavior *with* recent memory
- Lower = memory is helping

**Semi-conditional Complexity (~2.0):**
- How hard it is to predict *without* recent memory
- Higher = future is complex

**The Gap (~0.27):**
- How much memory reduces prediction error
- This is the value of learning!

**Prediction Distances (2.5-6.5):**
- L2 distance between predicted and actual behavior
- > 0 = can't perfectly predict (good!)
- Memory helps slightly (old > now)

## Why This Matters

### Philosophically

This project touches on deep questions:
- Can intelligence exist without curiosity?
- Is learning just compression, or something more?
- What makes an agent "keep going" when there's no external goal?

### Practically

The techniques here could help build AI that:
- **Explores naturally** without external rewards
- **Adapts continuously** to changing environments
- **Never stops learning** (lifelong learning)
- **Avoids local optima** (dark rooms)

### Scientifically

We've demonstrated:
- ✅ Stable behavioral self-prediction (unlike paper's Figure 4)
- ✅ Sustained interactivity for 900 steps
- ✅ Implicit capacity constraint (can't perfectly self-predict)
- ✅ Continual adaptation (agent doesn't converge)
- ✅ Anti-dark-room mechanisms that actually work

## The Journey (How We Got Here)

### Problem 1: Gradient Errors ❌→✅
**Issue:** "Modified by in-place operation" errors  
**Solution:** Recompute forward passes with fresh gradients instead of storing TD errors

### Problem 2: Dark Room Collapse ❌→✅
**Issue:** I-Score dropping to zero after 100 steps  
**Solution:** Aggressive exploration bonuses (5x normal strength)

### Problem 3: 20-Step Oscillations ❌→✅
**Issue:** I-Score crashing every 20 steps  
**Solution:** Soft target updates (tau=0.01) instead of hard sync

### Problem 4: Slow Convergence ❌→✅
**Issue:** I-Score slowly declining despite bonuses  
**Solution:** Adaptive weights + auto-reset mechanism

Each problem taught us something about the nature of self-prediction and continual learning.

## Running the Code

### Quick Start

```python
from behavioral_core_AGGRESSIVE import AggressiveAntiDarkRoomAgent

# Create agent
agent = AggressiveAntiDarkRoomAgent(
    behavior_size=64,    # Dimensionality of behavior space
    hidden_size=64,      # LSTM hidden units
    horizon=20           # How far ahead to predict
)

# Training loop
behavior = torch.randn(64) * 0.5  # Random initial behavior

for step in range(1000):
    result = agent.step(behavior)
    behavior = result['behavior']  # Close the loop
    
    if step % 50 == 0:
        print(f"Step {step}: I-Score = {result['i_score']:.4f}")
```

### What You'll See

```
Step 0:   I-Score = 0.0235
Step 50:  I-Score = 0.1224
Step 100: I-Score = 0.1832
...
🔄 Step 321: Resetting (I-Score stuck low)  ← Auto-reset in action!
...
Step 900: I-Score = 0.2679
```

The agent will:
- Start with high variance (exploring)
- Stabilize around I-Score = 0.25-0.28
- Occasionally reset itself if stuck
- Maintain learning for as long as you run it

## Files in This Project

### Core Implementation
- `behavioral_core_AGGRESSIVE.py` - The main agent (what you should use)
- `behavior_space_visualizer.py` - Visualization wrapper (PCA, trajectories)
- `behavioral_core_GRADIENT_FIX.py` - Previous version (gradient bug fix)

### Documentation
- `DARK_ROOM_SOLUTION.md` - Complete guide to the dark room problem
- `OSCILLATION_EXPLAINED.md` - Why I-Score oscillates every 20 steps
- `SOLUTION_COMPARISON.md` - Comparison of all solution attempts
- `EXACT_FIXES.md` - Step-by-step patches and fixes

### Experiments
- `SOFT_TARGET_UPDATE_PATCH.py` - Smooths oscillations
- `behavioral_core_enhanced.py` - Version with cellular automaton environment

## Key Parameters (Tuning Guide)

### If I-Score is too low (<0.01):
```python
agent.curiosity_weight = 1.0      # Increase from 0.5
agent.novelty_weight = 0.6        # Increase from 0.3
agent.min_i_score = 0.02          # Raise floor from 0.015
```

### If agent diverges (NaN):
```python
agent.optimizer = Adam(lr=0.001)  # Lower from 0.002
clip_grad_norm_(params, 0.5)      # Stronger clipping from 1.5
```

### If oscillations are too large:
```python
horizon = 5                        # Smaller from 20
tau = 0.01                        # Softer updates (already set)
```

## The Paper This Implements

**"The World Is Bigger: A Computationally-Embedded Perspective on the Big World Hypothesis"**  
Lewandowski et al., 2024

Our implementation closely follows Section 5-6:
- Section 5: Maximizing interactivity with RL
- Section 6: Behavioral self-prediction benchmark

We've achieved more stable results than the paper's Figure 4 by adding aggressive exploration mechanisms.

## Future Directions

### What Could Make This Even Better?

1. **Richer Environment**
   - Add cellular automaton boundary (like Conway's Game of Life)
   - Agent interacts with complex external dynamics
   - Should increase I-Score to 0.4-0.5+

2. **Hierarchical Behaviors**
   - Multiple timescales (fast and slow behaviors)
   - Should capture more complex patterns

3. **Scaling Up**
   - Larger networks (128, 256 hidden units)
   - Longer horizons (50, 100 steps)
   - Should maintain higher I-Score

4. **Meta-Learning**
   - Learn exploration strategies themselves
   - Adapt bonuses automatically

5. **Multi-Agent**
   - Multiple agents predicting each other
   - Social dynamics emerge

## Contributing

This is a research project exploring continual learning and self-prediction. If you:
- Find bugs or improvements
- Have ideas for extensions
- Want to discuss the theory

Feel free to open issues or reach out!

## License

MIT License - Feel free to use this code for research or learning.

## Acknowledgments

- Original paper: Lewandowski, Ramesh, Meyer, Schuurmans, Machado (2024)
- Inspiration: Active inference, intrinsic motivation, continual learning research
- Built with: PyTorch, love, and a lot of debugging

## Final Thoughts

This project shows that you can build an AI that *wants* to keep learning, not because you programmed a goal, but because the structure of the problem makes learning inherently valuable.

The agent is always chasing something it can never quite catch—itself. And in that eternal chase, it must continually adapt, explore, and grow.

Maybe that's not so different from us.

---

**Questions?** Read the documentation files or just run the code and watch it explore! 🚀
