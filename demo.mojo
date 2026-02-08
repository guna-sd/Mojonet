"""
MojoNet Demo: Linear Regression with Tnet Module.

This demo shows the complete deep learning pipeline using the Tnet module:
Tensor → Linear Layer → Autograd → SGD → Checkpoint → Inference

Usage:
    pixi shell
    mojo run demo.mojo
"""

from Tnet import Tensor, Linear, SGD, save, load_into
from Tnet.autodemo import autodemo


fn print_separator():
    print("=" * 60)


fn simple_demo() raises:
    print_separator()
    print("MojoNet Demo: Linear Regression with Tnet")
    print_separator()
    print()

    # ═══════════════════════════════════════════════════════════════════════
    # 1. Create Synthetic Data: y = 2*x + 1
    # ═══════════════════════════════════════════════════════════════════════
    print("[1] Creating synthetic data (y = 2*x + 1)...")

    # Training data: 5 samples [batch=5, features=1]
    var x_train = Tensor(5, 1)
    var y_train = Tensor(5, 1)
    for i in range(5):
        x_train[i] = Float32(i + 1)  # [1, 2, 3, 4, 5]
        y_train[i] = Float32(2 * (i + 1) + 1)  # [3, 5, 7, 9, 11]

    print("  x_train:", x_train)
    print("  y_train:", y_train)
    print()

    # ═══════════════════════════════════════════════════════════════════════
    # 2. Create Model: Linear(1 -> 1)
    # ═══════════════════════════════════════════════════════════════════════
    print("[2] Creating model: Linear(1 -> 1)...")

    var model = Linear(1, 1, has_bias=True)
    print("  Initial weight:", model.weight[0])
    print("  Initial bias:", model.bias[0])
    print("  Parameters:", model.num_parameters())
    print()

    # ═══════════════════════════════════════════════════════════════════════
    # 3. Create Optimizer: SGD with lr=0.05
    # ═══════════════════════════════════════════════════════════════════════
    print("[3] Creating optimizer: SGD(lr=0.05)...")
    var optimizer = SGD(lr=0.05)
    print()

    # ═══════════════════════════════════════════════════════════════════════
    # 4. Training Loop
    # ═══════════════════════════════════════════════════════════════════════
    print("[4] Training for 200 steps...")
    print()

    var n: Float32 = 5.0

    for step in range(200):
        # Zero gradients
        model.zero_grad()

        # Forward pass: pred = x @ W.T + b
        var pred = model.forward(x_train)

        # Compute MSE loss: loss = mean((pred - target)^2)
        var diff = pred - y_train
        var squared = diff ** Float32(2.0)
        var loss = squared.mean()

        # Compute gradient of MSE: d_loss/d_pred = 2*(pred - target)/n
        var grad_pred = diff * (Float32(2.0) / n)

        # Backward pass through linear layer
        var grad_input = model.backward(grad_pred)

        # Update parameters
        optimizer.step(model)

        # Print progress every 20 steps
        if step % 20 == 0 or step == 199:
            print("  Step", step, ": Loss =", loss)

    print()
    print("  Learned weight:", model.weight[0], "(expected: 2.0)")
    print("  Learned bias:", model.bias[0], "(expected: 1.0)")
    print()

    # ═══════════════════════════════════════════════════════════════════════
    # 5. Save Checkpoint
    # ═══════════════════════════════════════════════════════════════════════
    print("[5] Saving checkpoint...")

    save(model, "./checkpoint.txt")
    print("  Saved to: ./checkpoint.txt")
    print()

    # ═══════════════════════════════════════════════════════════════════════
    # 6. Load Checkpoint into New Model
    # ═══════════════════════════════════════════════════════════════════════
    print("[6] Loading checkpoint into new model...")

    var model2 = Linear(1, 1, has_bias=True)
    load_into(model2, "./checkpoint.txt")

    print("  Loaded weight:", model2.weight[0])
    print("  Loaded bias:", model2.bias[0])
    print()

    # ═══════════════════════════════════════════════════════════════════════
    # 7. Inference on Test Data
    # ═══════════════════════════════════════════════════════════════════════
    print("[7] Running inference...")

    var x_test = Tensor(3, 1)
    x_test[0] = 6.0
    x_test[1] = 7.0
    x_test[2] = 8.0

    var y_pred = model2.forward(x_test)

    print("  Test inputs: [6.0, 7.0, 8.0]")
    print("  Predictions: [", y_pred[0], ",", y_pred[1], ",", y_pred[2], "]")
    print("  Expected:    [13.0, 15.0, 17.0]")
    print()

    print_separator()
    print("Demo complete!")
    print("  Pipeline: Tensor -> Linear -> Autograd -> SGD -> Checkpoint")
    print_separator()


fn main() raises:
    simple_demo()
    autodemo()